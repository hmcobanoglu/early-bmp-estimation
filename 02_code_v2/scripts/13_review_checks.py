"""Checks requested in the pre-submission reviews (descriptive; no model refitting).

A. IL: recompute net SMP from cumulative CH4 and blank bottles and compare with the
   organisers' SMP curves (is the published curve a plain blank-corrected cumulative value?).
B. IIS: validated conditions without a day-10 estimate and the reason.
C. Condition flow for both archives (counts by step).
D. Reference definition: mean of bottle values at their own criterion day vs bottle mean at the
   condition completion day (latest bottle day).
E. Day-10 cellulose ratio with the training mean computed without the held-out laboratory;
   condition- and test-level tables; same grouping for the class-completion ratio.
F. IL primary set: per-material and per-laboratory errors, raw and laboratory-weighted shares.
G. Errors for conditions still running on each day, by class.

Run: python -I 02_code_v2/scripts/13_review_checks.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bmp_early import cases as C      # noqa: E402
from bmp_early import pipeline as P   # noqa: E402

R = C.ROOT / "03_outputs_v2"
OUT = R / "results"
RAW = C.RAW / "IL-BMP_Hafner2026"
res = {}
VSCOL = "c_vs_sub_f"  # organisers normalised by the fixed reference VS (checked against SMP.csv)


def male(d, col):
    e = np.abs(np.log(d[col] / d.ystar))
    return float(e.groupby(d.lab).mean().mean())


# ---------------- A. IL organiser SMP vs recomputed net SMP ----------------
def check_il_smp():
    s = pd.read_csv(RAW / "setup.csv")
    g = pd.read_csv(RAW / "biogas.csv")
    smp = pd.read_csv(RAW / "SMP.csv")
    out = []
    for f, st in s.groupby("file_id"):
        bk = st[st.substrate == "Blank"]
        gf = g[g.file_id == f]
        if bk.empty:
            continue
        bt, by = [], []
        for _, r in bk.iterrows():
            gb = gf[gf.bottle_id == r.bottle_id].sort_values("time_d")
            if len(gb) >= 2:
                bt.append(gb.time_d.to_numpy())
                by.append(gb.cvCH4.to_numpy() / r.mass_inoc)
        if not bt:
            continue
        grid = np.unique(np.concatenate(bt))
        vals = []
        for tt, yy in zip(bt, by):
            v = np.interp(grid, tt, yy)
            v[(grid < tt.min()) | (grid > tt.max())] = np.nan
            vals.append(v)
        bm = np.nanmean(np.vstack(vals), axis=0)
        ok = np.isfinite(bm)
        grid, bm = grid[ok], bm[ok]
        btimes = set(np.round(np.concatenate(bt), 4))
        for _, r in st[st.substrate != "Blank"].iterrows():
            gb = gf[gf.bottle_id == r.bottle_id].sort_values("time_d")
            mvs = r.mass_sub * r[VSCOL]
            if gb.empty or not np.isfinite(mvs) or mvs <= 0:
                continue
            tt = gb.time_d.to_numpy()
            inside = (tt >= grid.min()) & (tt <= grid.max())
            net = (gb.cvCH4.to_numpy() - np.interp(tt, grid, bm) * r.mass_inoc) / mvs
            out.append(pd.DataFrame({"bottle_id": r.bottle_id, "time_d": tt, "net": np.where(inside, net, np.nan),
                                     "sync": [round(x, 4) in btimes for x in tt]}))
    rc = pd.concat(out).merge(smp[["bottle_id", "time_d", "smp"]], on=["bottle_id", "time_d"])
    rc = rc[np.isfinite(rc.net) & (rc.smp > 20)]
    rel = 100 * (rc.net / rc.smp - 1)
    sy = rel[rc.sync]
    by_bottle = (rel.abs() > 1).groupby(rc.bottle_id).any()
    res["A_IL_smp_recomputed"] = dict(points=int(len(rc)), median_abs_pct=round(float(rel.abs().median()), 4),
                                      p90_abs_pct=round(float(rel.abs().quantile(.9)), 3), p95_abs_pct=round(float(rel.abs().quantile(.95)), 3),
                                      sync_points=int(len(sy)), sync_median_abs_pct=round(float(sy.abs().median()), 4),
                                      sync_p95_abs_pct=round(float(sy.abs().quantile(.95)), 3),
                                      bottles=int(rc.bottle_id.nunique()),
                                      bottles_any_point_gt1pct=int(by_bottle.sum()))


# ---------------- B. IIS conditions without a day-10 estimate ----------------
def check_missing():
    d = pd.read_csv(R / "iis" / "iis_lolo_predictions.csv", dtype={"lab": str})
    v = d[(d.status == "validated")]
    m = v[(v.day == 10) & (v.step == "none")]
    res["B_IIS_no_estimate_day10"] = m[["lab", "test", "material", "y_t", "t_last", "n_obs"]].to_dict(orient="records")
    labs_any = v[v.step != "none"].groupby("day").lab.nunique().to_dict()
    res["B_IIS_labs_with_estimates_by_day"] = {int(k): int(x) for k, x in labs_any.items()}
    res["B_IIS_conditions_with_estimates_by_day"] = {int(k): int(x) for k, x in
                                                     v[v.step != "none"].groupby("day").size().items()}
    res["B_IIS_P2_available_by_day"] = {int(k): [int(x.P2.notna().sum()), int(x[x.P2.notna()].lab.nunique())]
                                        for k, x in v.groupby("day")}


# ---------------- C. flow ----------------
def flow():
    ri = pd.read_csv(R / "iis" / "iis_references.csv", dtype={"lab": str})
    rl = pd.read_csv(R / "il" / "il_references.csv", dtype={"lab": str})
    f = {"IIS": dict(labs=int(ri.lab.nunique()), tests=int(ri.test.nunique()), conditions=len(ri),
                     status=ri.status.value_counts().to_dict(),
                     screened_labs=int(ri[ri.status == "validated"].lab.nunique()),
                     screened_tests=int(ri[ri.status == "validated"].test.nunique()))}
    rl["ref_lab"] = ~rl.lab.str.startswith("E")
    cel = rl[rl.material.str.startswith(C.CEL_IL + "|")].copy()
    cel = cel.assign(pref=(cel.material != C.CEL_IL + "|2").astype(int)).sort_values(["test", "pref"]).drop_duplicates("test")
    cel["in_range"] = cel.ystar.between(340, 395)
    cel["pass"] = cel.complete & cel.in_range & (cel.rsd <= 6)
    il = {}
    for (rd, ref), x in rl.groupby(["round", "ref_lab"]):
        c = cel[(cel["round"] == rd) & (cel.ref_lab == ref)]
        key = f"round{rd}_{'reference' if ref else 'other'}"
        il[key] = dict(labs=int(x.lab.nunique()), files=int(x.test.nunique()), conditions=len(x),
                       status=x.status.value_counts().to_dict(),
                       cel_tests=len(c), cel_mean_in_range=int(c.in_range.sum()), cel_screen_pass=int(c["pass"].sum()),
                       cel_incomplete=int((~c.complete).sum()),
                       screened_labs=int(x[x.status == "validated"].lab.nunique()),
                       screened_noncel=int(((x.status == "validated") & ~x.material.str.startswith(C.CEL_IL)).sum()),
                       screened_cel=int(((x.status == "validated") & x.material.str.startswith(C.CEL_IL)).sum()))
    p = pd.read_csv(R / "il" / "il_predictions.csv", dtype={"lab": str})
    p["eval_lab"] = p.lab.str.startswith("E")
    t1 = p[(p.day == 10) & (p["round"] == 1) & p.eval_lab & (p.status == "validated") & (p.base != C.CEL_IL)]
    il["round1_other_primary_path"] = dict(screened_noncel=len(t1), labs=int(t1.lab.nunique()),
                                           sync=int(t1.sync.sum()), sync_labs=int(t1[t1.sync].lab.nunique()),
                                           async_labs=sorted(t1[~t1.sync].lab.unique().tolist()),
                                           with_estimate=int(t1[t1.sync].locked.notna().sum()))
    f["IL"] = il
    res["C_flow"] = f


# ---------------- D. reference definition ----------------
def check_reference_definition():
    ri = pd.read_csv(R / "iis" / "iis_references.csv", dtype={"lab": str})
    v = ri[ri.status == "validated"].set_index(["test", "material"])
    bc = C.iis_bottle_curves()
    vals = []
    for (tid, mat), x in bc.groupby(["test", "material"]):
        if (tid, mat) not in v.index:
            continue
        ts = v.loc[(tid, mat), "tstar"]
        yb = [np.interp(ts, d.time, d.smp) for _, d in x.groupby("bottle") if d.time.max() >= ts]
        if yb:
            vals.append(100 * (np.mean(yb) / v.loc[(tid, mat), "ystar"] - 1))
    vals = pd.Series(vals)
    res["D_IIS_common_day_vs_own_day_pct"] = dict(n=len(vals), median=round(float(vals.median()), 3),
                                                  p90_abs=round(float(vals.abs().quantile(.9)), 3), p95_abs=round(float(vals.abs().quantile(.95)), 3), n_gt1=int((vals.abs() > 1).sum()),
                                                  max=round(float(vals.max()), 3))


# ---------------- E. cellulose ratio, held-out-laboratory training mean ----------------
def check_cellulose():
    d = pd.read_csv(R / "iis" / "iis_lolo_predictions.csv", dtype={"lab": str})
    d10 = d[d.day == 10]
    celv = d10[(d10.base == "CEL") & (d10.status == "validated")]
    obs = d10[d10.status.isin(["validated", "outside_qc"]) & (d10.base != "CEL")].copy()
    obs["ref_cel"] = [celv[celv.lab != l].y_t.mean() for l in obs.lab]
    obs["cel_ratio"] = obs.y_cel_t / obs.ref_cel
    obs["grp"] = pd.cut(obs.cel_ratio, [0, 0.9, 1.1, 9], labels=["<0.9", "0.9-1.1", ">1.1"])
    tab = {}
    for col in ("locked", "P1"):
        x = obs.dropna(subset=["grp"])
        x = x[np.isfinite(x[col]) & (x[col] > 0)].copy()
        x["e"] = np.log(x[col] / x.ystar)
        tab[col] = {str(k): dict(n=int(len(g)), labs=int(g.lab.nunique()),
                                 male=round(float(np.abs(g.e).groupby(g.lab).mean().mean()), 4),
                                 bias=round(float(g.e.groupby(g.lab).mean().mean()), 4),
                                 gmr=round(float(np.exp(g.e.groupby(g.lab).mean().mean())), 3),
                                 share_e_pos=round(float((g.e > 0).mean()), 3),
                                 share_outside=round(float((g.status == "outside_qc").mean()), 3))
                    for k, g in x.groupby("grp", observed=True)}
    res["E_cellulose_groups_heldout_mean"] = tab
    res["E_no_cellulose_on_day10"] = int(obs.cel_ratio.isna().sum())
    # test level
    ri = pd.read_csv(R / "iis" / "iis_references.csv", dtype={"lab": str})
    tst = obs.groupby("test").agg(cel_ratio=("cel_ratio", "first"), lab=("lab", "first")).dropna()
    ri_t = ri.groupby("test").status.apply(lambda s: "validated" if (s == "validated").any() else "other")
    tst["screen"] = tst.index.map(ri_t)
    tst["low"] = tst.cel_ratio < 0.9
    res["E_test_level"] = {f"{'low' if lo else 'not_low'}_{sc}": int(n)
                           for (lo, sc), n in tst.groupby(["low", "screen"]).size().items()}
    obs[["lab", "test", "material", "cls", "status", "cel_ratio", "ystar", "locked", "P1"]].to_csv(
        OUT / "cellulose_ratio_day10.csv", index=False)


# ---------------- F. IL primary set detail ----------------
def check_il_primary():
    p = pd.read_csv(R / "il" / "il_predictions.csv", dtype={"lab": str})
    p["eval_lab"] = p.lab.str.startswith("E")
    pr = p[(p.status == "validated") & (p.base != C.CEL_IL) & (p["round"] == 1) & p.eval_lab & p.sync]
    out = {}
    for day in (10, 14):
        x = pr[pr.day == day].copy()
        x["e"] = np.log(x.locked / x.ystar)
        x["e1"] = np.log(x.P1 / x.ystar)
        within = (np.abs(x.locked / x.ystar - 1) <= 0.10)
        out[day] = dict(n=len(x), labs=int(x.lab.nunique()),
                        by_material={b: dict(n=len(g), labs=int(g.lab.nunique()), male=round(male(g, "locked"), 4),
                                             male_P1=round(male(g, "P1"), 4)) for b, g in x.groupby("base")},
                        by_lab={l: dict(n=len(g), male=round(float(g.e.abs().mean()), 4),
                                        male_P1=round(float(g.e1.abs().mean()), 4)) for l, g in x.groupby("lab")},
                        within10_raw=f"{int(within.sum())}/{len(x)}",
                        within10_labweighted=round(float(within.groupby(x.lab).mean().mean()), 3),
                        pi_covered=f"{int(((x.ystar >= x.pi_lo) & (x.ystar <= x.pi_up)).sum())}/{int(x.pi_lo.notna().sum())}",
                        geo_width_ratio=round(float(np.exp(np.log(x.pi_up / x.pi_lo).mean())), 3))
        # leave one laboratory out of the summary
        out[day]["drop_one_lab_range"] = [round(min(male(x[x.lab != l], "locked") for l in x.lab.unique()), 4),
                                          round(max(male(x[x.lab != l], "locked") for l in x.lab.unique()), 4)]
    res["F_IL_primary"] = out


# ---------------- G. running conditions by class ----------------
def check_running():
    d = pd.read_csv(R / "iis" / "iis_lolo_predictions.csv", dtype={"lab": str})
    v = d[(d.status == "validated") & (d.tstar > d.day)]
    out = {}
    for day in (10, 14, 21):
        x = v[(v.day == day) & v.locked.notna()]
        out[day] = {"all": [len(x), int(x.lab.nunique()), round(male(x, "locked"), 4), round(male(x, "P1"), 4)]}
        for c, g in x.groupby("cls"):
            out[day][c] = [len(g), int(g.lab.nunique()), round(male(g, "locked"), 4), round(male(g, "P1"), 4)]
    res["G_running_n_labs_male_locked_P1"] = out


if __name__ == "__main__":
    check_il_smp()
    check_missing()
    flow()
    check_reference_definition()
    check_cellulose()
    check_il_primary()
    check_running()
    json.dump(res, open(OUT / "review_checks.json", "w"), indent=1, default=str)
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))
