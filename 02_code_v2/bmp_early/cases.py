"""Case tables and causal net SMP curves for the two inter-laboratory archives.

A case is laboratory x test x material (triplicate). Curves are returned per bottle
with a column `blank_ok_until` giving, for every substrate observation, the time of
the latest blank observation needed to compute it; truncation at day t keeps an
observation only if both its own time and that blank time are <= t.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "01_data_v2" / "raw"
PROC = ROOT / "01_data_v2" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

CLASS_IIS = {"CEL": "readily", "SA": "feed", "SB": "feed", "SC": "feed", "SD": "straw"}
CLASS_IL = {"Cellulose": "readily", "Whey": "readily", "Rabbit feed": "feed",
            "Bird feed": "feed", "Sunflower seeds": "feed", "Wheat straw": "straw"}
CEL_IIS, CEL_IL = "CEL", "Cellulose"
NOMINAL_ISR = np.array([1.0, 2.0, 4.0])


def nominal_isr(x):
    """Nearest of 1, 2 or 4 on the log scale (conditions tested within one assay)."""
    x = np.asarray(x, float)
    out = NOMINAL_ISR[np.argmin(np.abs(np.log(x[:, None]) - np.log(NOMINAL_ISR)[None, :]), axis=1)]
    return np.where(np.isfinite(x) & (x > 0), out, 2.0)


def cond_label(material, isr):
    return f"{material}|{int(isr)}"


def base_material(cond):
    return cond.split("|")[0]


def _blank_mean(bk_times, bk_curves):
    """Mean blank curve on the union grid; each blank interpolated inside its range."""
    grid = np.unique(np.concatenate(bk_times))
    vals = []
    for tt, yy in zip(bk_times, bk_curves):
        v = np.interp(grid, tt, yy)
        v[(grid < tt.min()) | (grid > tt.max())] = np.nan
        vals.append(v)
    return grid, np.nanmean(np.vstack(vals), axis=0)


_IIS_RAW = {}


def _iis_raw():
    if not _IIS_RAW:
        _IIS_RAW["b"] = pd.read_csv(RAW / "IIS-BMP_Hafner2022" / "bottles.csv")
        _IIS_RAW["t"] = pd.read_csv(RAW / "IIS-BMP_Hafner2022" / "tests.csv")
        _IIS_RAW["g"] = pd.read_csv(RAW / "IIS-BMP_Hafner2022" / "biogas.csv")
    return _IIS_RAW["b"].copy(), _IIS_RAW["t"], _IIS_RAW["g"]


def iis_bottle_curves(t_cut=np.inf):
    """IIS: net SMP per substrate bottle from normalised cumulative CH4.
    With t_cut, all gas records after t_cut (substrate and blank bottles) are removed
    before any calculation, so every value uses only data available at t_cut."""
    b, t, g = _iis_raw()
    g = g[g["time.d"] <= t_cut]
    b = b.merge(t, on="test.id", how="left")
    b["ISR"] = b["ISR"].replace([np.inf, -np.inf], np.nan)
    out = []
    for tid, bt in b.groupby("test.id"):
        bk = bt[bt["substrate"] == "BK"]
        if bk.empty:
            continue
        bk_t, bk_y = [], []
        for _, r in bk.iterrows():
            gb = g[g["bottle.id"] == r["bottle.id"]].sort_values("time.d")
            if len(gb) >= 2:
                bk_t.append(gb["time.d"].to_numpy())
                bk_y.append(gb["cvCH4"].to_numpy() / r["m.inoc"])
        if not bk_t:
            continue
        grid, bmean = _blank_mean(bk_t, bk_y)
        ok = np.isfinite(bmean)
        grid, bmean = grid[ok], bmean[ok]
        for _, r in bt[bt["substrate"] != "BK"].iterrows():
            if not (np.isfinite(r["m.sub.vs"]) and r["m.sub.vs"] > 0):
                continue
            gb = g[g["bottle.id"] == r["bottle.id"]].sort_values("time.d")
            s = gb["time.d"].to_numpy()
            inside = (s >= grid.min()) & (s <= grid.max())
            s, cv = s[inside], gb["cvCH4"].to_numpy()[inside]
            idx = np.searchsorted(grid, s, side="left")          # upper bracket
            idx = np.clip(idx, 0, len(grid) - 1)
            blank_t = grid[idx]
            net = (cv - np.interp(s, grid, bmean) * r["m.inoc"]) / r["m.sub.vs"]
            out.append(pd.DataFrame({
                "archive": "IIS", "lab": str(r["lab.id"]), "test": int(tid),
                "material": cond_label(r["substrate"], nominal_isr([r["ISR"]])[0]),
                "bottle": int(r["bottle.id"]),
                "time": s, "smp": net, "blank_time": blank_t,
                "isr": r["ISR"], "study": r["study"], "method": r["method.biogas"]}))
    return pd.concat(out, ignore_index=True)


def il_bottle_curves():
    """IL: published net SMP (organiser calculation). blank_time = time itself when a
    blank of the same test was measured at that time (synchronous), else the next
    blank observation time (asynchronous; may lie after the prediction day)."""
    s = pd.read_csv(RAW / "IL-BMP_Hafner2026" / "setup.csv")
    smp = pd.read_csv(RAW / "IL-BMP_Hafner2026" / "SMP.csv")
    gas_t = pd.read_csv(RAW / "IL-BMP_Hafner2026" / "biogas.csv", usecols=["bottle_id", "time_d"])
    s = s.assign(cond=[cond_label(m, i) for m, i in zip(s.substrate, nominal_isr(s.ISR))])
    smp = smp[["bottle_id", "time_d", "smp"]].merge(
        s[["bottle_id", "test", "lab_id", "file_id", "substrate", "cond", "ISR"]], on="bottle_id")
    gt = gas_t.merge(s[["bottle_id", "file_id", "substrate"]], on="bottle_id")
    bk_times = {f: np.unique(d["time_d"].round(4)) for f, d in gt[gt["substrate"] == "Blank"].groupby("file_id")}
    rows = []
    for f, d in smp[smp["substrate"] != "Blank"].groupby("file_id"):
        bt = bk_times.get(f, np.array([]))
        tt = d["time_d"].round(4).to_numpy()
        if len(bt):
            j = np.clip(np.searchsorted(bt, tt, side="left"), 0, len(bt) - 1)
            bl = np.where(np.isin(tt, bt), tt, bt[j])
            bl = np.where(tt > bt.max(), np.inf, bl)
        else:
            bl = np.full(len(tt), np.inf)
        rows.append(pd.DataFrame({
            "archive": "IL", "lab": d["lab_id"].to_numpy(), "test": d["file_id"].to_numpy(),
            "round": d["test"].to_numpy(),
            "material": d["cond"].to_numpy(), "bottle": d["bottle_id"].to_numpy(),
            "time": d["time_d"].to_numpy(), "smp": d["smp"].to_numpy(), "blank_time": bl,
            "isr": d["ISR"].to_numpy(), "file": f}))
    return pd.concat(rows, ignore_index=True)


def case_curve(bottles, t=np.inf):
    """Triplicate-mean net SMP using only observations available at day t.
    Each bottle is interpolated inside its own observed range onto the union of
    observation times; times where any bottle is undefined are dropped."""
    b = bottles[(bottles["time"] <= t) & (bottles["blank_time"] <= t)]
    b = b[np.isfinite(b["smp"])]
    if b.empty:
        return np.array([0.0]), np.array([0.0])
    grid = np.unique(np.concatenate([[0.0], b["time"].to_numpy()]))
    vals = []
    for _, d in b.groupby("bottle"):
        d = d.sort_values("time")
        tt = np.concatenate([[0.0], d["time"].to_numpy()]) if d["time"].min() > 0 else d["time"].to_numpy()
        yy = np.concatenate([[0.0], d["smp"].to_numpy()]) if d["time"].min() > 0 else d["smp"].to_numpy()
        v = np.interp(grid, tt, yy)
        v[grid > tt.max()] = np.nan
        vals.append(v)
    m = np.vstack(vals)
    keep = np.all(np.isfinite(m), axis=0)
    return grid[keep], m[:, keep].mean(axis=0)
