"""All numbers reported in the manuscript, written to 03_outputs_v2/results/.

Run after 06_il_evaluation.py:  python -I 02_code_v2/scripts/07_results_tables.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bmp_early import cases as C       # noqa: E402
from bmp_early import pipeline as P    # noqa: E402

R = C.ROOT / "03_outputs_v2"
OUT = R / "results"
OUT.mkdir(parents=True, exist_ok=True)
METHODS = ["locked", "P0", "P1", "P2", "P3", "C1"]
rng = np.random.default_rng(7)


def metric_rows(df, label, by_class=True):
    rows = []
    groups = [("all", df)] + ([(c, d) for c, d in df.groupby("cls")] if by_class else [])
    for cls, g in groups:
        for day in P.DAYS:
            dd = g[g.day == day]
            for m in METHODS:
                mean, G, n, per = P.lab_weighted(dd, m)
                lo, hi = P.bootstrap_ci(per)
                sgn = P.lab_weighted(dd, m, fn=lambda e: e)[0]
                ok = np.isfinite(dd[m]) & (dd[m] > 0)
                w10 = (np.abs(dd[m] / dd.ystar - 1) <= 0.10) & ok
                rows.append(dict(set=label, cls=cls, day=day, method=m, mae_log=mean, ci_lo=lo, ci_hi=hi,
                                 bias_log=sgn, G=G, n_est=n, n_cases=len(dd),
                                 share_est=ok.mean() if len(dd) else np.nan,
                                 share_w10_all=w10.groupby(dd.lab).mean().mean() if len(dd) else np.nan))
    return rows


def first_day(df, method="locked", tol=0.10, target=0.90, B=500):
    """First evaluated day at which the laboratory-weighted share of cases within
    tolerance reaches the target; stability by laboratory bootstrap."""
    out = []
    for cls, g in df.groupby("cls"):
        def fd(sub):
            for day in P.DAYS:
                dd = sub[sub.day == day]
                if dd.empty:
                    continue
                w = ((np.abs(dd[method] / dd.ystar - 1) <= tol) & np.isfinite(dd[method])).groupby(dd.lab).mean().mean()
                if w >= target:
                    return day
            return None
        labs = g.lab.unique()
        point = fd(g)
        bs = []
        for _ in range(B):
            pick = rng.choice(labs, len(labs), replace=True)
            sub = pd.concat([g[g.lab == l].assign(lab=f"{l}_{i}") for i, l in enumerate(pick)])
            bs.append(fd(sub))
        vc = pd.Series([b if b is not None else "none" for b in bs]).value_counts(normalize=True)
        ts = g.drop_duplicates(["lab", "test", "material"]).tstar
        out.append(dict(cls=cls, tol=tol, first_day=point, boot_share_point=float(vc.get(point if point else "none", 0)),
                        boot_share_none=float(vc.get("none", 0)), median_tstar=float(ts.median()),
                        q25_tstar=float(ts.quantile(.25)), q75_tstar=float(ts.quantile(.75)),
                        n_cases=int(len(ts)), G=int(g.lab.nunique())))
    return out


def main():
    iis = pd.read_csv(R / "iis" / "iis_lolo_predictions.csv", dtype={"lab": str})
    il = pd.read_csv(R / "il" / "il_predictions.csv", dtype={"lab": str})
    iis_v = iis[iis.status == "validated"]
    il["eval_lab"] = il.lab.str.startswith("E")
    il_p = il[(il.status == "validated") & (il.base != C.CEL_IL) & (il["round"] == 1) & il.eval_lab & il.sync]

    rows = metric_rows(iis_v, "IIS_LOLO_validated")
    ong = iis_v[iis_v.tstar > iis_v.day]
    rows += metric_rows(ong, "IIS_LOLO_ongoing", by_class=False)
    rows += metric_rows(iis[iis.status == "outside_qc"], "IIS_LOLO_outsideQC")
    rows += metric_rows(il_p, "IL_primary")
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)

    fdays = []
    for tol in (0.05, 0.10, 0.15):
        fdays += [dict(archive="IIS", **r) for r in first_day(iis_v, tol=tol)]
    pd.DataFrame(fdays).to_csv(OUT / "first_day.csv", index=False)

    # data flow
    refs_iis = pd.read_csv(R / "iis" / "iis_references.csv", dtype={"lab": str})
    refs_il = pd.read_csv(R / "il" / "il_references.csv", dtype={"lab": str})
    refs_il["eval_lab"] = refs_il.lab.str.startswith("E")
    flow = {
        "IIS": dict(labs=refs_iis.lab.nunique(), tests=refs_iis.test.nunique(), cases=len(refs_iis),
                    status=refs_iis.status.value_counts().to_dict(),
                    validated_labs=refs_iis[refs_iis.status == "validated"].lab.nunique(),
                    cel_tests=int(refs_iis.material.str.startswith("CEL|").groupby(refs_iis.test).any().sum()),
                    cel_qc_pass_tests=int(refs_iis.groupby("test").qc_strict.first().sum()),
                    day10_est=int(np.isfinite(iis_v[iis_v.day == 10].locked).sum()),
                    day10_cases=int((iis_v.day == 10).sum()),
                    steps_day10=iis_v[iis_v.day == 10].step.value_counts().to_dict()),
        "IL": dict(labs=refs_il.lab.nunique(), files=refs_il.test.nunique(), cases=len(refs_il),
                   status=refs_il.status.value_counts().to_dict(),
                   T1_eval_cases=int(((refs_il["round"] == 1) & refs_il.eval_lab).sum()),
                   T1_eval_status=refs_il[(refs_il["round"] == 1) & refs_il.eval_lab].status.value_counts().to_dict(),
                   primary_day10_cases=int((il_p.day == 10).sum()),
                   primary_labs=int(il_p.lab.nunique()),
                   steps_day10=il_p[il_p.day == 10].step.value_counts().to_dict(),
                   ystar_vs_bmpcalc_abs_pct=dict(zip(["median", "p90"], (100 * (refs_il.ystar / refs_il.bmp_calc - 1)).abs()
                                                     .quantile([.5, .9]).round(2).tolist()))),
    }
    # reference timing and Y* by material (validated)
    v = refs_iis[refs_iis.status == "validated"].assign(base=lambda d: d.material.str.split("|").str[0])
    flow["IIS_by_material"] = v.groupby("base").agg(n=("ystar", "size"), labs=("lab", "nunique"),
                                                     ystar_mean=("ystar", "mean"), ystar_sd=("ystar", "std"),
                                                     tstar_median=("tstar", "median")).round(1).to_dict(orient="index")
    disp = pd.read_csv(R / "iis" / "dispersion_log_scale.csv", index_col=0)
    flow["dispersion"] = disp.round(4).to_dict(orient="index")
    cov = pd.read_csv(R / "il" / "il_interval_coverage.csv")
    flow["intervals_IL_primary"] = cov.round(3).to_dict(orient="records")
    json.dump(flow, open(OUT / "flow_and_context.json", "w"), indent=1, default=str)
    print(json.dumps(flow, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
