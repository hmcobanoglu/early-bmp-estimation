"""Checks requested in the second round of pre-submission reviews (descriptive).

a. 2016-2018 reference vs organiser values on conditions complete in both calculations; bottles in
   the setup without gas records.
b. 2023-2024 primary set: organisers' net SMP vs recalculation for the bottles of the 16 primary
   conditions at times up to day 10.
c. Between-laboratory deviation of the 16 primary references against all accepted round-1 peers.
d. Number of tests in the cellulose 2x2 table.
e. Heat-map data: day-10 signed error by laboratory and substrate (2016-2018, complete conditions).
f. 2023-2024 primary set: signed errors by laboratory and material for the corrected model and ratio.

Run: python -I 02_code_v2/scripts/16_final_checks.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bmp_early import cases as C      # noqa: E402

R = C.ROOT / "03_outputs_v2"
OUT = R / "results"
SI = C.ROOT / "05_literature" / "supplements" / "Hafner2020_SI" / "extracted" / "data"
RAWL = C.RAW / "IL-BMP_Hafner2026"
res = {}


def a_reference():
    refs = pd.read_csv(R / "iis" / "iis_references.csv")
    d = pd.read_csv(SI / "D.csv")[["test_id", "substrate", "ISR_nom", "inoc_type", "BMP_1p3dnet", "rate_crit_met_1p3dnet"]]
    d["material"] = d.substrate + "|" + d.ISR_nom.round().astype(int).astype(str)
    d = d[d.inoc_type.isin(["Own", "Own2"]) | ~d.duplicated(["test_id", "material"], keep=False)]
    m = refs.merge(d, left_on=["test", "material"], right_on=["test_id", "material"])
    both = m[m.complete & (m.rate_crit_met_1p3dnet == 1)]
    dd = (100 * (both.ystar / both.BMP_1p3dnet - 1)).abs()
    res["a_IIS_both_complete"] = dict(n=len(both), median=round(float(dd.median()), 4),
                                      p90=round(float(dd.quantile(.9)), 3), p95=round(float(dd.quantile(.95)), 3),
                                      gt1=int((dd > 1).sum()), matched=len(m))
    b, t, g = C._iis_raw()
    miss = b[~b["bottle.id"].isin(g["bottle.id"].unique())]
    res["a_IIS_setup_bottles"] = int(len(b))
    res["a_IIS_bottles_with_gas"] = int(g["bottle.id"].nunique())
    res["a_IIS_bottles_without_gas"] = miss[["test.id", "bottle.id", "substrate"]].to_dict(orient="records")
    st = refs.set_index("test")
    res["a_IIS_tests_affected_status"] = {int(k): refs[refs.test == k].status.value_counts().to_dict()
                                          for k in miss["test.id"].unique() if k in st.index}


def b_il_primary():
    s = pd.read_csv(RAWL / "setup.csv")
    gas = pd.read_csv(RAWL / "biogas.csv")
    smp = pd.read_csv(RAWL / "SMP.csv")
    p = pd.read_csv(R / "il" / "il_predictions.csv", dtype={"lab": str})
    pr = p[(p.day == 10) & (p.status == "validated") & (p.base != C.CEL_IL) & (p["round"] == 1)
           & p.lab.str.startswith("E") & p.sync]
    s["material"] = [C.cond_label(m_, i) for m_, i in zip(s.substrate, C.nominal_isr(s.ISR))]
    keys = set(zip(pr.lab, pr.test, pr.material))
    sub = s[[(l, f, m_) in keys for l, f, m_ in zip(s.lab_id, s.file_id, s.material)]]
    rows = []
    for f, st in s[s.file_id.isin(sub.file_id.unique())].groupby("file_id"):
        bk = st[st.substrate == "Blank"]
        gf = gas[(gas.file_id == f) & (gas.time_d <= 10)]
        bt, by = [], []
        for _, r in bk.iterrows():
            gb = gf[gf.bottle_id == r.bottle_id].sort_values("time_d")
            bt.append(gb.time_d.to_numpy())
            by.append(gb.cvCH4.to_numpy() / r.mass_inoc)
        grid = np.unique(np.concatenate(bt))
        vals = [np.where((grid >= tt.min()) & (grid <= tt.max()), np.interp(grid, tt, yy), np.nan) for tt, yy in zip(bt, by)]
        bm = np.nanmean(np.vstack(vals), axis=0)
        for _, r in sub[sub.file_id == f].iterrows():
            gb = gf[gf.bottle_id == r.bottle_id].sort_values("time_d")
            net = (gb.cvCH4.to_numpy() - np.interp(gb.time_d, grid, bm) * r.mass_inoc) / (r.mass_sub * r.c_vs_sub_f)
            rows.append(pd.DataFrame({"bottle_id": r.bottle_id, "time_d": gb.time_d.to_numpy(), "net": net}))
    rc = pd.concat(rows).merge(smp[["bottle_id", "time_d", "smp"]], on=["bottle_id", "time_d"])
    rc = rc[rc.time_d > 0]
    rel = 100 * (rc.net / rc.smp - 1)
    absd = (rc.net - rc.smp).abs()
    d10 = rc.sort_values("time_d").groupby("bottle_id").last()
    res["b_IL_primary_recalc_upto_day10"] = dict(bottles=int(rc.bottle_id.nunique()), points=len(rc),
                                                 median_abs_pct=round(float(rel.abs().median()), 4), p90_abs_pct=round(float(rel.abs().quantile(.9)), 3), p95_abs_pct=round(float(rel.abs().quantile(.95)), 3),
                                                 max_abs_pct_smp_gt20=round(float(rel[rc.smp > 20].abs().max()), 3),
                                                 max_abs_NmL=round(float(absd.max()), 3),
                                                 last_point_max_abs_pct=round(float((100 * (d10.net / d10.smp - 1)).abs().max()), 4))


def c_dl_primary():
    rl = pd.read_csv(R / "il" / "il_references.csv", dtype={"lab": str})
    acc = rl[(rl["round"] == 1) & (rl.status == "validated")]
    p = pd.read_csv(R / "il" / "il_predictions.csv", dtype={"lab": str})
    pr = p[(p.day == 10) & (p.status == "validated") & (p.base != C.CEL_IL) & (p["round"] == 1)
           & p.lab.str.startswith("E") & p.sync]
    out = []
    for _, r in pr.iterrows():
        o = acc[(acc.material == r.material) & (acc.lab != r.lab)]
        if len(o) >= 3:
            out.append(dict(lab=r.lab, d=abs(np.log(r.ystar / o.ystar.median())), peers=len(o)))
    d = pd.DataFrame(out)
    res["c_dL_primary16"] = dict(dL=round(float(d.groupby("lab").d.mean().mean()), 4), n=len(d),
                                 min_peers=int(d.peers.min()))


def d_tests():
    d = pd.read_csv(OUT / "cellulose_ratio_day10.csv", dtype={"lab": str})
    ri = pd.read_csv(R / "iis" / "iis_references.csv", dtype={"lab": str})
    t_ratio = d.dropna(subset=["cel_ratio"]).test.nunique()
    t_noncel = d.test.nunique()
    res["d_tests"] = dict(with_ratio=int(t_ratio), with_noncel_condition=int(t_noncel),
                          with_cellulose=int(ri[ri.material.str.startswith("CEL|")].test.nunique()),
                          all=int(ri.test.nunique()))


def e_heatmap():
    d = pd.read_csv(R / "iis" / "iis_lolo_predictions.csv", dtype={"lab": str})
    x = d[(d.day == 10) & d.status.isin(["validated", "outside_qc"]) & d.locked.notna()].copy()
    x["e"] = np.log(x.locked / x.ystar)
    h = x.groupby(["lab", "base"]).e.mean().unstack()
    cr = pd.read_csv(OUT / "cellulose_ratio_day10.csv", dtype={"lab": str}).groupby("lab").cel_ratio.median()
    acc = x.groupby("lab").status.apply(lambda s: (s == "validated").mean())
    h = h.join(cr.rename("cel_ratio")).join(acc.rename("share_accepted"))
    h.to_csv(OUT / "heatmap_lab_substrate_day10.csv")
    res["e_heatmap_labs"] = int(len(h))


def f_il_labs():
    p = pd.read_csv(R / "il" / "il_predictions.csv", dtype={"lab": str})
    pr = p[(p.day == 10) & (p.status == "validated") & (p.base != C.CEL_IL) & (p["round"] == 1)
           & p.lab.str.startswith("E") & p.sync].copy()
    pr["e_P3"] = np.log(pr.locked / pr.ystar)
    pr["e_P1"] = np.log(pr.P1 / pr.ystar)
    pr[["lab", "base", "ystar", "locked", "P1", "e_P3", "e_P1", "pi_lo", "pi_up"]].to_csv(OUT / "il_primary_day10.csv", index=False)
    w1 = (np.abs(pr.P1 / pr.ystar - 1) <= 0.1)
    res["f_IL_P1_within10_raw"] = f"{int(w1.sum())}/{len(pr)}"


if __name__ == "__main__":
    a_reference()
    b_il_primary()
    c_dl_primary()
    d_tests()
    e_heatmap()
    f_il_labs()
    json.dump(res, open(OUT / "final_checks.json", "w"), indent=1, default=str)
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))
