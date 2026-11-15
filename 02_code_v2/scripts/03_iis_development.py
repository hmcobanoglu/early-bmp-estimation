"""IIS-BMP development analysis: reference BMP, case table, leave-one-laboratory-out
estimates for days 5-21, and the all-IIS models that are locked before evaluation.

Run after 02_ystar_biogas_iis.R:  python -I 02_code_v2/scripts/03_iis_development.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bmp_early import cases as C          # noqa: E402
from bmp_early import pipeline as P       # noqa: E402

OUT = C.ROOT / "03_outputs_v2" / "iis"
OUT.mkdir(parents=True, exist_ok=True)
QC = {"strict": (340, 395, 6.0), "relaxed": (330, 415, 15.0)}


def references(bottles):
    o = pd.read_csv(C.ROOT / "03_outputs_v2" / "inventory" / "iis_ystar_biogas_bottles.csv")
    labs = bottles.drop_duplicates("test").set_index("test")["lab"]
    isr = bottles.groupby(["test", "material"])["isr"].mean()
    g = o.groupby(["test.id", "cond"]).agg(ystar=("cvCH4", "mean"), sd=("cvCH4", "std"),
                                               tstar=("time.d", "max"), rr=("rrvCH4", "max"),
                                               nb=("cvCH4", "size")).reset_index()
    g = g.rename(columns={"test.id": "test", "cond": "material"})
    g["lab"] = g["test"].map(labs)
    g["complete"] = g["rr"] < 1.0
    g["rsd"] = 100 * g["sd"] / g["ystar"]
    g["isr"] = [isr.get((t, m), np.nan) for t, m in zip(g.test, g.material)]
    cel = (g[g.material.str.startswith("CEL|")]
           .assign(pref=lambda x: (x.material != "CEL|2").astype(int))
           .sort_values(["test", "pref"]).drop_duplicates("test").set_index("test"))
    for name, (lo, hi, rmax) in QC.items():
        ok = cel.complete & cel.ystar.between(lo, hi) & (cel.rsd <= rmax)
        g[f"qc_{name}"] = g["test"].map(ok).fillna(False).astype(bool)
    g["status"] = np.where(~g.complete, "incomplete",
                           np.where(g.qc_strict, "validated", "outside_qc"))
    g["status_relaxed"] = np.where(~g.complete, "incomplete",
                                   np.where(g.qc_relaxed, "validated", "outside_qc"))
    return g.dropna(subset=["lab"]).set_index(["lab", "test", "material"])


def lolo(df, curves, status_col="status", keep_models=False):
    rows, models = [], {}
    d0 = df.copy()
    d0["status"] = d0[status_col]
    for day in P.DAYS:
        dd = d0[d0.day == day]
        for lab in sorted(dd.lab.unique()):
            tr, te = dd[dd.lab != lab], dd[dd.lab == lab]
            m = P.fit(tr, curves, day)
            if keep_models:
                models[(day, lab)] = m
            rows.append(P.predict(m, te))
    return pd.concat(rows, ignore_index=True), models


def main():
    bottles = C.iis_bottle_curves()
    bottles.to_csv(C.PROC / "iis_bottle_curves.csv", index=False)
    refs = references(bottles)
    refs.reset_index().to_csv(OUT / "iis_references.csv", index=False)
    cut_cache = {t: C.iis_bottle_curves(t_cut=t) for t in P.DAYS}
    df, curves = P.assemble(bottles, refs, C.CEL_IIS, C.CLASS_IIS, bottles_at=cut_cache.get)
    pickle.dump((df, curves), open(OUT / "iis_cases.pkl", "wb"))
    pred, fold_models = lolo(df, curves, keep_models=True)
    pred.drop(columns=["tt", "yy"]).to_csv(OUT / "iis_lolo_predictions.csv", index=False)
    pickle.dump(fold_models, open(OUT / "iis_lolo_fold_models.pkl", "wb"))
    pred_r, _ = lolo(df, curves, "status_relaxed")
    pred_r.drop(columns=["tt", "yy"]).to_csv(OUT / "iis_lolo_predictions_relaxedqc.csv", index=False)
    # all-IIS models (one per day) for the locked external evaluation
    models = {day: P.fit(df, curves, day) for day in P.DAYS}
    pickle.dump(models, open(OUT / "iis_models_locked.pkl", "wb"))
    print(refs.status.value_counts().to_dict())
    v = pred[pred.status == "validated"]
    for day in P.DAYS:
        d = v[v.day == day]
        for col in ["locked", "P0", "P1", "P2", "P3", "C1"]:
            m, G, n, per = P.lab_weighted(d, col)
            print(day, col, round(m, 4), G, n)


if __name__ == "__main__":
    main()
