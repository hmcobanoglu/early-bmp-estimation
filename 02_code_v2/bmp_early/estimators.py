"""Features available at day t and the provisional BMP estimators.

Names in the manuscript: P0 last observed net yield; P1 class-completion ratio;
P2 lag-adjusted first-order fit; P3 residual-corrected kinetic model;
C1 cellulose-adjusted ratio.
"""
import numpy as np
from scipy.optimize import least_squares

MIN_OBS_P2 = 4      # observations in (0, t]
MIN_AFTER_LAG = 3   # points at or after the lag end
MAX_AGE = 4.0       # d, staleness limit when the criterion is not yet met


def curve_features(tt, yy, t):
    """Features of a truncated case curve (times tt <= t, values yy, tt[0] = 0)."""
    f = {"y_t": np.nan, "t_last": np.nan, "age": np.nan, "n_obs": 0,
         "tau": np.nan, "y_tau": np.nan, "rate_ratio": np.nan, "crit_met": 0,
         "imax_last": 1}
    pos = tt > 0
    f["n_obs"] = int(pos.sum())
    if f["n_obs"] == 0:
        return f
    f["y_t"], f["t_last"] = yy[-1], tt[-1]
    f["age"] = t - tt[-1]
    if len(tt) < 3:
        return f
    dt = np.diff(tt)
    r = np.diff(yy) / np.where(dt > 0, dt, np.nan)
    j = int(np.nanargmax(r))                     # interval j spans tt[j]..tt[j+1]
    f["tau"], f["y_tau"] = tt[j], yy[j]          # end of interval preceding max rate
    f["imax_last"] = int(j == len(r) - 1)
    rmax = r[j]
    f["rate_ratio"] = r[-1] / rmax if rmax > 0 else np.nan
    # 1% net 3 d visible within 0..t: relative daily rate < 1% over a window >= 3 d
    rel = r / np.where(yy[1:] > 0, yy[1:], np.nan)
    ok = rel < 0.01
    k = len(ok) - 1
    while k >= 0 and ok[k]:
        k -= 1
    if tt[-1] - tt[k + 1] >= 3 and k < len(ok) - 1:
        f["crit_met"] = 1
    return f


def p0(f):
    return f["y_t"] if f["y_t"] > 0 else np.nan


def p1(f, fbar_c):
    """fbar_c: callable giving training mean completion at time t_last."""
    if not (f["y_t"] > 0):
        return np.nan
    fr = fbar_c(f["t_last"])
    return f["y_t"] / fr if fr > 0.05 else np.nan


SETTINGS = {"form": "first_order", "tbmp_cap": None}   # sensitivity switches


def p2(tt, yy, f, kappa, t, tbmp=np.nan):
    """Lag-adjusted first-order fit y = y(tau) + A[1 - exp(-k (t - tau))].
    Upper bound on A from kappa * Y_t (training-derived, composition-free); the
    composition-informed sensitivity caps the asymptote at 1.10 * TBMP instead."""
    out = {"pred": np.nan, "A": np.nan, "k": np.nan, "at_bound": 0}
    if f["n_obs"] < MIN_OBS_P2 or not (f["y_t"] > 0) or f["imax_last"]:
        return out
    if f["age"] > MAX_AGE and not f["crit_met"]:
        return out
    if SETTINGS["form"] == "gompertz":
        return _gompertz(tt, yy, f, kappa, tbmp)
    if SETTINGS["tbmp_cap"] and np.isfinite(tbmp):
        kappa = max(SETTINGS["tbmp_cap"] * tbmp / f["y_t"], 1.0 + 1e-6)
    sel = tt >= f["tau"]
    x, y = tt[sel] - f["tau"], yy[sel]
    if len(x) < MIN_AFTER_LAG:
        return out
    a_max = max(kappa * f["y_t"] - f["y_tau"], 1e-6)
    a0 = min(max(f["y_t"] - f["y_tau"], 1e-3) * 1.5, a_max * 0.99)

    def res(p):
        return f["y_tau"] + p[0] * (1 - np.exp(-p[1] * x)) - y
    try:
        sol = least_squares(res, x0=[a0, 0.15], bounds=([0, 0.01], [a_max, 1.5]))
    except Exception:
        return out
    if not sol.success:
        return out
    A, k = sol.x
    pred = f["y_tau"] + A
    if not (pred > 0):
        return out
    out.update(pred=pred, A=A, k=k,
               at_bound=int(A > 0.995 * a_max or k < 0.0101 or k > 1.49))
    return out


def _gompertz(tt, yy, f, kappa, tbmp):
    """Modified Gompertz (Zwietering et al., 1990) fitted to all points in (0, t]:
    y = A exp{-exp[R e (lambda - t)/A + 1]}; returns A; k reported as R/A."""
    out = {"pred": np.nan, "A": np.nan, "k": np.nan, "at_bound": 0}
    a_max = kappa * f["y_t"]
    if SETTINGS["tbmp_cap"] and np.isfinite(tbmp):
        a_max = max(SETTINGS["tbmp_cap"] * tbmp, f["y_t"] * (1 + 1e-6))
    x, y = tt, yy

    def res(p):
        A, R, lam = p
        return A * np.exp(-np.exp(R * np.e * (lam - x) / A + 1)) - y
    a0 = min(f["y_t"] * 1.3, a_max * 0.99)
    try:
        sol = least_squares(res, x0=[a0, max(f["y_t"] / max(f["t_last"], 1), 1.0), 0.5],
                            bounds=([f["y_t"], 0.0, 0.0], [a_max, np.inf, f["t_last"]]))
    except Exception:
        return out
    if not sol.success:
        return out
    A, R, lam = sol.x
    out.update(pred=A, A=A, k=R / A, at_bound=int(A > 0.995 * a_max))
    return out


P3_FEATURES = ["log_yt_over_p2", "rate_ratio", "k", "t_minus_tau", "age", "isr",
               "crit_met", "at_bound", "cel_ratio", "cel_missing", "cls_feed", "cls_straw"]


def p3_design(f, p2o, cel_ratio, isr, cls):
    cel_missing = int(not (np.isfinite(cel_ratio) and cel_ratio > 0))
    return [np.log(f["y_t"] / p2o["pred"]), f["rate_ratio"], p2o["k"],
            f["t_last"] - f["tau"], f["age"], isr if np.isfinite(isr) else np.nan,
            f["crit_met"], p2o["at_bound"],
            np.log(cel_ratio) if not cel_missing else 0.0, cel_missing,
            int(cls == "feed"), int(cls == "straw")]
