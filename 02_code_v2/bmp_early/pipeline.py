"""Case assembly, model fitting and prediction shared by both archives."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from . import estimators as E
from .cases import case_curve, base_material

DAYS = (5, 7, 10, 14, 21)
GRID = np.arange(0, 61, 0.5)
ALPHAS = np.logspace(-2, 3, 11)


def assemble(bottles, refs, cel_name, class_map, days=DAYS, bottles_at=None):
    """One row per case x day with features available at that day.
    refs: DataFrame indexed by (lab, test, material) with ystar, tstar, status.
    bottles_at: optional callable t -> bottle curves rebuilt from raw data cut at t
    (IIS); otherwise curves are truncated with the time and blank-time rule (IL)."""
    rows, curves = [], {}
    keyed = {k: d for k, d in bottles.groupby(["lab", "test", "material"])}
    cut = {}
    for t in days:
        bt = bottles_at(t) if bottles_at else bottles[(bottles.time <= t) & (bottles.blank_time <= t)]
        cut[t] = {k: d for k, d in bt.groupby(["lab", "test", "material"])}
    for key, d in keyed.items():
        if key not in refs.index:
            continue
        ref = refs.loc[key]
        tt, yy = case_curve(d)
        curves[key] = (tt, yy)
        celkeys = sorted(k for k in keyed if k[0] == key[0] and k[1] == key[1]
                         and base_material(k[2]) == cel_name)
        celkey = celkeys[0] if celkeys else None
        for t in days:
            dt_ = cut[t].get(key)
            tt_t, yy_t = case_curve(dt_, t) if dt_ is not None else (np.array([0.0]), np.array([0.0]))
            f = E.curve_features(tt_t, yy_t, t)
            ycel = np.nan
            if base_material(key[2]) != cel_name and celkey in cut[t]:
                ct, cy = case_curve(cut[t][celkey], t)
                if len(ct) > 1 and ct[-1] > 0:
                    ycel = cy[-1]
            rows.append(dict(lab=key[0], test=key[1], material=key[2], cls=class_map.get(base_material(key[2])),
                             base=base_material(key[2]), round=ref.get("round", np.nan),
                             day=t, ystar=ref.ystar, tstar=ref.tstar, status=ref.status,
                             status_relaxed=ref.get('status_relaxed', ref.status),
                             isr=ref.isr, y_cel_t=ycel, tt=tt_t, yy=yy_t, **f))
    return pd.DataFrame(rows), curves


def _fbar(train, curves):
    """Mean completion curve Y(t)/Y* per class from training validated cases."""
    out = {}
    for c, d in train.drop_duplicates(["lab", "test", "material"]).groupby("cls"):
        vals = []
        for _, r in d.iterrows():
            tt, yy = curves[(r.lab, r.test, r.material)]
            v = np.interp(GRID, tt, yy) / r.ystar
            v[GRID > tt.max()] = np.nan
            vals.append(v)
        m = np.nanmean(np.vstack(vals), axis=0)
        ok = np.isfinite(m)
        out[c] = (GRID[ok], m[ok])
    return out


def fit(train, curves, day):
    """Fit all components on validated training cases for one prediction day."""
    tr = train[(train.status == "validated") & (train.day == day)].copy()
    model = {"day": day, "fbar": _fbar(tr, curves)}
    tr = tr[tr.y_t > 0]
    model["kappa"] = {c: 1.1 * float(np.nanmax(d.ystar / d.y_t)) for c, d in tr.groupby("cls")}
    cel = tr.loc[tr.base.isin(["CEL", "Cellulose"]), "y_t"]
    model["cel_mean_yt"] = float(cel.mean()) if len(cel) else np.nan
    # gamma for the cellulose-adjusted ratio, non-cellulose cases only
    p1v = np.array([_p1(model, r) for _, r in tr.iterrows()])
    rel = tr.y_cel_t.to_numpy() / model["cel_mean_yt"]
    ok = np.isfinite(p1v) & np.isfinite(rel) & (rel > 0) & (p1v > 0)
    x, y = np.log(rel[ok]), np.log(tr.ystar.to_numpy()[ok] / p1v[ok])
    model["gamma"] = float((x @ y) / (x @ x)) if ok.sum() >= 5 else 0.0
    # Ridge on log residuals of the first-order fit
    X, Y, G = [], [], []
    for _, r in tr.iterrows():
        o = _p2(model, r)
        if np.isfinite(o["pred"]):
            X.append(E.p3_design(r, o, r.y_cel_t / model["cel_mean_yt"], r.isr, r.cls))
            Y.append(np.log(r.ystar / o["pred"]))
            G.append(r.lab)
    X, Y, G = np.array(X, float), np.array(Y), np.array(G)
    model["impute"] = np.nanmean(X, axis=0) if len(X) else None
    if len(X) >= 20 and len(np.unique(G)) >= 5:
        X = np.where(np.isfinite(X), X, model["impute"])
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1
        Z = (X - mu) / sd
        gkf = GroupKFold(n_splits=min(5, len(np.unique(G))))
        cv = []
        for a in ALPHAS:
            err = []
            for itr, ite in gkf.split(Z, Y, G):
                m = Ridge(alpha=a).fit(Z[itr], Y[itr])
                err.append(np.mean(np.abs(m.predict(Z[ite]) - Y[ite])))
            cv.append(np.mean(err))
        a = ALPHAS[int(np.argmin(cv))]
        model.update(ridge=Ridge(alpha=a).fit(Z, Y), mu=mu, sd=sd, alpha=a)
    else:
        model["ridge"] = None
    return model


def _p1(model, r):
    fb = model["fbar"].get(r.cls)
    if fb is None:
        return np.nan
    return E.p1(r, lambda t: float(np.interp(t, fb[0], fb[1])))


def _p2(model, r):
    kap = model["kappa"].get(r.cls, np.nan)
    if not np.isfinite(kap):
        return {"pred": np.nan, "A": np.nan, "k": np.nan, "at_bound": 0}
    return E.p2(r.tt, r.yy, r, kap, r.day, r.get("tbmp", np.nan))


def predict(model, df):
    """All estimators and the locked procedure (P3 -> P2 -> P1 -> P0) per row."""
    out = []
    for _, r in df.iterrows():
        p0 = E.p0(r)
        p1 = _p1(model, r)
        o2 = _p2(model, r)
        p2 = o2["pred"]
        rel = r.y_cel_t / model["cel_mean_yt"] if np.isfinite(model["cel_mean_yt"]) else np.nan
        c1 = p1 * rel ** model["gamma"] if (np.isfinite(p1) and np.isfinite(rel) and rel > 0) else p1
        p3 = np.nan
        if np.isfinite(p2) and model.get("ridge") is not None:
            x = np.array(E.p3_design(r, o2, rel, r.isr, r.cls), float)
            x = np.where(np.isfinite(x), x, model["impute"])
            try:
                p3 = p2 * float(np.exp(model["ridge"].predict(((x - model["mu"]) / model["sd"])[None])[0]))
            except Exception:
                p3 = np.nan
        for name, v in (("P3", p3), ("P2", p2), ("P1", p1), ("P0", p0)):
            if np.isfinite(v) and v > 0:
                locked, step = v, name
                break
        else:
            locked, step = np.nan, "none"
        out.append(dict(P0=p0, P1=p1, P2=p2, P3=p3, C1=c1, locked=locked, step=step,
                        A=o2["A"], k=o2["k"], at_bound=o2["at_bound"]))
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def lab_weighted(df, col, fn=lambda e: np.abs(e)):
    """Laboratory-weighted mean of fn(log error) over rows with a valid estimate."""
    d = df[np.isfinite(df[col]) & (df[col] > 0)]
    e = fn(np.log(d[col] / d.ystar))
    per_lab = e.groupby(d.lab).mean()
    return float(per_lab.mean()), int(per_lab.size), int(len(d)), per_lab


def bootstrap_ci(per_lab, n=2000, seed=1):
    rng = np.random.default_rng(seed)
    v = per_lab.to_numpy()
    if len(v) < 3:
        return np.nan, np.nan
    bs = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
