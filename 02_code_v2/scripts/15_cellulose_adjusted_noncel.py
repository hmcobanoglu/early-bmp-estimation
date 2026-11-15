"""Check of the cellulose-adjusted ratio (deviation 7): gamma is re-estimated in every
leave-one-laboratory-out fold (and for the methods fixed on all 2016-2018 data) explicitly from
non-cellulose training conditions, and cellulose conditions receive the unadjusted class-completion
estimate. The fold models are reused unchanged. The result is identical to the locked estimates,
because the locked code leaves the control yield missing for cellulose conditions.

Writes the estimates to results/cellulose_adjusted_noncel_{iis,il}.csv; the locked prediction files are not changed.
Run: python -I 02_code_v2/scripts/15_cellulose_adjusted_noncel.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bmp_early import cases as C       # noqa: E402
from bmp_early import pipeline as P    # noqa: E402

R = C.ROOT / "03_outputs_v2"
CEL = {"CEL", "Cellulose"}


def gamma_noncel(model, train):
    tr = train[(train.y_t > 0) & ~train.base.isin(CEL)]
    p1 = np.array([P._p1(model, r) for _, r in tr.iterrows()])
    rel = tr.y_cel_t.to_numpy() / model["cel_mean_yt"]
    ok = np.isfinite(p1) & np.isfinite(rel) & (rel > 0) & (p1 > 0)
    x, y = np.log(rel[ok]), np.log(tr.ystar.to_numpy()[ok] / p1[ok])
    return float((x @ y) / (x @ x)) if ok.sum() >= 5 else 0.0


def apply(df, model, g):
    rel = df.y_cel_t / model["cel_mean_yt"]
    adj = df.P1 * np.power(rel.where(rel > 0), g)
    out = np.where(df.base.isin(CEL) | ~np.isfinite(adj), df.P1, adj)
    return out


def main():
    cases, _ = pickle.load(open(R / "iis" / "iis_cases.pkl", "rb"))
    cases["lab"] = cases.lab.astype(str)
    acc = cases[cases.status == "validated"]
    folds = pickle.load(open(R / "iis" / "iis_lolo_fold_models.pkl", "rb"))
    locked = pickle.load(open(R / "iis" / "iis_models_locked.pkl", "rb"))

    iis = pd.read_csv(R / "iis" / "iis_lolo_predictions.csv", dtype={"lab": str})
    iis["C1n"] = np.nan
    gam = []
    for (day, lab), model in folds.items():
        tr = acc[(acc.day == day) & (acc.lab != lab)]
        g = gamma_noncel(model, tr)
        gam.append(dict(day=day, lab=lab, gamma_noncel=g, gamma_locked=model["gamma"]))
        m = (iis.day == day) & (iis.lab == lab)
        iis.loc[m, "C1n"] = apply(iis[m], model, g)
    iis[["lab", "test", "material", "day", "status", "C1", "C1n"]].to_csv(R / "results" / "cellulose_adjusted_noncel_iis.csv", index=False)

    il = pd.read_csv(R / "il" / "il_predictions.csv", dtype={"lab": str})
    il["C1n"] = np.nan
    for day, model in locked.items():
        g = gamma_noncel(model, acc[acc.day == day])
        gam.append(dict(day=day, lab="all", gamma_noncel=g, gamma_locked=model["gamma"]))
        m = il.day == day
        il.loc[m, "C1n"] = apply(il[m], model, g)
    il[["lab", "test", "material", "day", "status", "C1", "C1n"]].to_csv(R / "results" / "cellulose_adjusted_noncel_il.csv", index=False)
    gm = pd.DataFrame(gam)
    gm.to_csv(R / "results" / "gamma_noncel.csv", index=False)
    print(gm.groupby("day")[["gamma_noncel", "gamma_locked"]].median().round(3))
    v = iis[(iis.day == 10) & (iis.status == "validated")]
    for c in ("C1", "C1n", "P1"):
        print(c, round(P.lab_weighted(v, c)[0], 4), round(P.lab_weighted(v[v.base != "CEL"], c)[0], 4))


if __name__ == "__main__":
    main()
