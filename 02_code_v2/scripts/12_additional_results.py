"""Additional results requested in the internal review (computed after the lock;
listed in protocol/deviations.md):
1. between-laboratory mean absolute log deviation of the reference (same metric as the
   prediction error): |ln(B_ref / median B_ref of the same material and ISR in other labs)|,
   laboratory-weighted;
2. paired laboratory-level comparisons prespecified in the protocol (section 5);
3. day-t cellulose progress as a warning signal: errors by same-test cellulose ratio on day 10;
4. source of the difference between the recalculated IL reference and organiser values.

Run: python -I 02_code_v2/scripts/12_additional_results.py
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
rng = np.random.default_rng(11)


def dev_between_labs(refs):
    v = refs[refs.status == "validated"].copy()
    v["base"] = v.material.str.split("|").str[0]
    out = []
    for _, r in v.iterrows():
        others = v[(v.material == r.material) & (v.lab != r.lab)]
        if len(others) >= 3:
            out.append(dict(lab=r.lab, base=r.base, d=abs(np.log(r.ystar / others.ystar.median()))))
    d = pd.DataFrame(out)
    cls = {"CEL": "readily", "SA": "feed", "SB": "feed", "SC": "feed", "SD": "straw"}
    d["cls"] = d.base.map(cls)
    res = {"all": d.groupby("lab").d.mean().mean()}
    for c, g in d.groupby("cls"):
        res[c] = g.groupby("lab").d.mean().mean()
    for b, g in d.groupby("base"):
        res[b] = g.groupby("lab").d.mean().mean()
    return {k: round(float(v), 4) for k, v in res.items()}


def paired(df, a, b, day=10, B=2000):
    d = df[(df.day == day) & np.isfinite(df[a]) & np.isfinite(df[b]) & (df[a] > 0) & (df[b] > 0)]
    ea = np.abs(np.log(d[a] / d.ystar)).groupby(d.lab).mean()
    eb = np.abs(np.log(d[b] / d.ystar)).groupby(d.lab).mean()
    diff = (ea - eb).dropna()
    bs = [rng.choice(diff.values, len(diff)).mean() for _ in range(B)]
    return dict(a=a, b=b, day=day, mean_diff=round(float(diff.mean()), 4),
                ci=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                labs=int(len(diff)), labs_a_better=int((diff < 0).sum()), conditions=int(len(d)))


def main():
    res = {}
    refs = pd.read_csv(R / "iis" / "iis_references.csv", dtype={"lab": str})
    res["between_lab_abs_log_dev_IIS"] = dev_between_labs(refs)
    iis = pd.read_csv(R / "iis" / "iis_lolo_predictions.csv", dtype={"lab": str})
    v = iis[iis.status == "validated"]
    res["paired_IIS"] = [paired(v, a, b, d) for d in (7, 10, 14) for a, b in
                         (("locked", "P1"), ("P3", "P2"), ("locked", "P0"), ("C1", "P1"))]
    o = iis[iis.status == "outside_qc"]
    res["paired_IIS_outside"] = [paired(o, "C1", "P1", 10), paired(o, "locked", "P1", 10)]
    il = pd.read_csv(R / "il" / "il_predictions.csv", dtype={"lab": str})
    il["eval_lab"] = il.lab.str.startswith("E")
    prim = il[(il.status == "validated") & (il.base != C.CEL_IL) & (il["round"] == 1) & il.eval_lab & il.sync]
    res["paired_IL_primary"] = [paired(prim, "locked", "P1", 10), paired(prim, "locked", "P0", 10)]
    # cellulose progress on day 10 as a warning signal (all observed non-cellulose IIS conditions)
    obs = iis[(iis.day == 10) & iis.status.isin(["validated", "outside_qc"]) & (iis.base != "CEL")].copy()
    ref_cel = iis[(iis.day == 10) & (iis.base == "CEL") & (iis.status == "validated")].y_t.mean()
    obs["cel_ratio"] = obs.y_cel_t / ref_cel
    obs["e"] = np.log(obs.locked / obs.ystar)
    obs["grp"] = pd.cut(obs.cel_ratio, [0, 0.9, 1.1, 9], labels=["<0.9", "0.9-1.1", ">1.1"])
    g = obs.dropna(subset=["e", "grp"]).groupby("grp", observed=True)
    res["cellulose_ratio_day10"] = {str(k): dict(n=int(len(x)), labs=int(x.lab.nunique()),
                                                 male=round(float(np.abs(x.e).groupby(x.lab).mean().mean()), 4),
                                                 bias=round(float(x.e.groupby(x.lab).mean().mean()), 4),
                                                 share_outside_qc=round(float((x.status == "outside_qc").mean()), 3))
                                    for k, x in g}
    # IL reference vs organiser bottle values: compare with SMP at the final observation
    raw = C.RAW / "IL-BMP_Hafner2026"
    smp = pd.read_csv(raw / "SMP.csv")
    calc = pd.read_csv(raw / "BMP_calc.csv")
    last = smp.sort_values("time_d").groupby("bottle_id").smp.last().rename("smp_last")
    cc = calc.merge(last, on="bottle_id")
    dl = (100 * (cc.bmp_calc / cc.smp_last - 1)).abs()
    res["IL_bmpcalc_vs_final_SMP_pct"] = dict(median=round(float(dl.median()), 3),
                                               p90=round(float(dl.quantile(.9)), 3), n=int(len(dl)))
    json.dump(res, open(OUT / "additional_results.json", "w"), indent=1)
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
