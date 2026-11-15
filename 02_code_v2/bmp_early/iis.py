"""Load the IIS-BMP 2018-19 archive and recalculate reference BMP (Y*).

Data: github.com/sashahafner/BMP-kinetics-paper-2022 (MIT), commit 55715f3.
Gas volumes in biogas.csv are already dry, 0 degC, 1 atm (README); they are not
normalised again here.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "01_data_v2" / "raw" / "IIS-BMP_Hafner2022"

BLANK = "BK"
CELLULOSE = "CEL"
MATERIAL_CLASS = {  # a priori, from material type (plan section 6.3)
    "CEL": "readily_degradable",
    "SA": "composite_feed",
    "SB": "composite_feed",
    "SC": "composite_feed",
    "SD": "lignocellulosic",
}
# Theoretical maximum BMP, NmL CH4 g-1 VS (Hafner et al. 2020, Table 1)
TBMP = {"SA": 440.0, "SB": 448.0, "SC": 606.0, "SD": 481.0, "CEL": 414.0}

CEL_RANGE = (340.0, 395.0)   # Standard BMP Methods doc 101 v1.0
CEL_RSD_MAX = 6.0            # %


def load():
    tests = pd.read_csv(RAW / "tests.csv")
    bottles = pd.read_csv(RAW / "bottles.csv")
    gas = pd.read_csv(RAW / "biogas.csv")
    smp = pd.read_csv(RAW / "SMP.csv")
    bottles = bottles.merge(tests, on="test.id", how="left")
    return tests, bottles, gas, smp


def blank_rate_per_inoc(bottles, gas):
    """Mean cumulative blank CH4 per g inoculum (fresh mass), per test and time."""
    bk = bottles.loc[bottles["substrate"] == BLANK, ["test.id", "bottle.id", "m.inoc"]]
    g = gas.merge(bk, on="bottle.id")
    g["spec"] = g["cvCH4"] / g["m.inoc"]
    return g[["test.id", "bottle.id", "time.d", "spec"]]


def _interp_within(x_obs, y_obs, x_new):
    """Linear interpolation inside the observed range only (NaN outside)."""
    x_obs = np.asarray(x_obs, float)
    y_obs = np.asarray(y_obs, float)
    out = np.interp(x_new, x_obs, y_obs)
    out[(x_new < x_obs.min()) | (x_new > x_obs.max())] = np.nan
    return out


def net_smp_bottle(bottles, gas):
    """Net SMP (NmL CH4 g-1 VS) per substrate bottle at its own measurement times."""
    bk = blank_rate_per_inoc(bottles, gas)
    sub = bottles.loc[bottles["substrate"] != BLANK]
    rows = []
    for test_id, bt in sub.groupby("test.id"):
        bks = bk[bk["test.id"] == test_id]
        if bks.empty:
            continue
        # mean blank curve: interpolate each blank onto union grid, then average
        grid = np.unique(bks["time.d"].to_numpy())
        curves = [
            _interp_within(b["time.d"], b["spec"], grid)
            for _, b in bks.groupby("bottle.id")
        ]
        blank_mean = np.nanmean(np.vstack(curves), axis=0) if len(curves) > 1 else curves[0]
        ok = ~np.isnan(blank_mean)
        for _, b in bt.iterrows():
            gb = gas[gas["bottle.id"] == b["bottle.id"]].sort_values("time.d")
            if gb.empty or not np.isfinite(b["m.sub.vs"]) or b["m.sub.vs"] <= 0:
                continue
            bl = _interp_within(grid[ok], blank_mean[ok], gb["time.d"].to_numpy())
            net = (gb["cvCH4"].to_numpy() - bl * b["m.inoc"]) / b["m.sub.vs"]
            rows.append(pd.DataFrame({
                "test.id": test_id, "lab.id": b["lab.id"], "study": b["study"],
                "substrate": b["substrate"], "bottle.id": b["bottle.id"],
                "time.d": gb["time.d"].to_numpy(), "smp_net": net,
            }))
    return pd.concat(rows, ignore_index=True)


def unit_curve(bottle_curves, source_col="smp_net"):
    """Triplicate-mean net SMP on an integer-day grid, interpolating each bottle
    only within its own observed range (no extrapolation)."""
    tmax = bottle_curves.groupby("bottle.id")["time.d"].max().min()
    grid = np.arange(0, int(np.floor(tmax)) + 1, dtype=float)
    curves = []
    for _, b in bottle_curves.groupby("bottle.id"):
        b = b.dropna(subset=[source_col]).sort_values("time.d")
        if len(b) < 2:
            continue
        curves.append(_interp_within(b["time.d"], b[source_col], grid))
    if not curves:
        return grid, np.full_like(grid, np.nan), 0
    return grid, np.nanmean(np.vstack(curves), axis=0), len(curves)


def termination_1pct_3d(grid, y):
    """First day at which daily net production was <1% of the accumulated net
    production on 3 consecutive days (Holliger et al. 2021; Koch et al. 2026).
    Returns (day, Y*) or (nan, nan) if not reached."""
    run = 0
    for i in range(1, len(grid)):
        if not (np.isfinite(y[i]) and np.isfinite(y[i - 1])) or y[i] <= 0:
            run = 0
            continue
        run = run + 1 if (y[i] - y[i - 1]) / y[i] < 0.01 else 0
        if run == 3:
            return grid[i], y[i]
    return np.nan, np.nan
