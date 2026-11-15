"""Locked evaluation on IL-BMP 2023-2024 (run only after protocol/LOCK.txt exists).

Reference: t* from 05_ystar_il.R; Y* = mean published net SMP of the bottles at t*.
Primary set: Test 1, evaluation laboratories (E*), non-cellulose materials, validated
references, synchronous blank/substrate measurements up to the prediction day.

Run: python -I 02_code_v2/scripts/06_il_evaluation.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bmp_early import cases as C       # noqa: E402
from bmp_early import pipeline as P    # noqa: E402

IIS = C.ROOT / "03_outputs_v2" / "iis"
OUT = C.ROOT / "03_outputs_v2" / "il"
OUT.mkdir(parents=True, exist_ok=True)
RAW = C.RAW / "IL-BMP_Hafner2026"
ALPHA = 0.10


def references():
    setup = pd.read_csv(RAW / "setup.csv")
    smp = pd.read_csv(RAW / "SMP.csv")
    ts = pd.read_csv(OUT / "il_tstar_biogas_bottles.csv")
    calc = pd.read_csv(RAW / "BMP_calc.csv")
    ts = ts.rename(columns={"time_d": "tstar_b"})[["bottle_id", "tstar_b", "rrvCH4"]]
    b = setup[setup.substrate != "Blank"].merge(ts, on="bottle_id", how="left")
    b["cond"] = [C.cond_label(m, i) for m, i in zip(b.substrate, C.nominal_isr(b.ISR))]
    vals = []
    for _, r in b.iterrows():
        s = smp[smp.bottle_id == r.bottle_id].sort_values("time_d")
        v = np.interp(r.tstar_b, s.time_d, s.smp) if (np.isfinite(r.tstar_b) and len(s)) else np.nan
        vals.append(v)
    b["y_b"] = vals
    g = b.groupby(["lab_id", "file_id", "cond"]).agg(round=("test", "first"),
        ystar=("y_b", "mean"), sd=("y_b", "std"), tstar=("tstar_b", "max"), rr=("rrvCH4", "max"),
        isr=("ISR", "mean"), nb=("y_b", "size")).reset_index()
    g["complete"] = g.rr < 1.0
    g["rsd"] = 100 * g.sd / g.ystar
    cel = (g[g.cond.str.startswith(C.CEL_IL + "|")]
           .assign(pref=lambda x: (x.cond != C.CEL_IL + "|2").astype(int))
           .sort_values(["file_id", "pref"]).drop_duplicates("file_id").set_index(["lab_id", "file_id"]))
    for name, (lo, hi, rmax) in {"strict": (340, 395, 6.0), "relaxed": (330, 415, 15.0)}.items():
        ok = cel.complete & cel.ystar.between(lo, hi) & (cel.rsd <= rmax)
        g[f"qc_{name}"] = [bool(ok.get((l, f), False)) for l, f in zip(g.lab_id, g.file_id)]
    g["status"] = np.where(~g.complete, "incomplete", np.where(g.qc_strict, "validated", "outside_qc"))
    g["status_relaxed"] = np.where(~g.complete, "incomplete", np.where(g.qc_relaxed, "validated", "outside_qc"))
    # comparison with organiser calculation and submitted values (reported, not used)
    calc = calc.merge(b[["bottle_id", "cond"]], on="bottle_id")
    cm = calc.groupby(["lab_id", "file_id", "cond"]).bmp_calc.mean().rename("bmp_calc")
    g = g.join(cm, on=["lab_id", "file_id", "cond"])
    g = g.rename(columns={"lab_id": "lab", "file_id": "test", "cond": "material"})
    return g.set_index(["lab", "test", "material"])


def cv_plus(fold_models, lolo, df, day):
    """CV+ intervals with leave-one-laboratory-out residuals of the locked procedure,
    scores weighted 1/n_g; returns lower, upper (ln scale) per row."""
    res = lolo[(lolo.day == day) & (lolo.status == "validated") & np.isfinite(lolo.locked)].copy()
    res["score"] = np.abs(np.log(res.locked / res.ystar))
    lo, up = np.full(len(df), np.nan), np.full(len(df), np.nan)
    preds = {g: P.predict(m, df).locked.to_numpy() for (d, g), m in fold_models.items() if d == day}
    for i, (_, r) in enumerate(df.iterrows()):
        rr = res[res.cls == r.cls]
        labs = [g for g in rr.lab.unique() if g in preds]
        if len(labs) < 9:
            continue
        v_lo, v_up, w = [], [], []
        for g in labs:
            sc = rr[rr.lab == g].score.to_numpy()
            mu = preds[g][i]
            if not (np.isfinite(mu) and mu > 0):
                continue
            v_lo += list(np.log(mu) - sc)
            v_up += list(np.log(mu) + sc)
            w += [1.0 / (len(sc) * (len(labs) + 1))] * len(sc)
        if not w:
            continue
        w = np.array(w)
        up[i] = _wq_upper(np.array(v_up), w, 1 - ALPHA)
        lo[i] = -_wq_upper(-np.array(v_lo), w, 1 - ALPHA)
    return lo, up


def _wq_upper(v, w, level):
    """Smallest v with cumulative weight >= level; the remaining mass (1 - sum w)
    sits at +inf (finite-sample correction of the pooled weighted distribution)."""
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    k = np.searchsorted(cw, level - 1e-12)
    return v[o][k] if k < len(v) else np.inf


def summarise(pred, label, mask):
    rows = []
    d = pred[mask]
    for day in P.DAYS:
        dd = d[d.day == day]
        for col in ["locked", "P0", "P1", "P2", "P3", "C1"]:
            m, G, n, per = P.lab_weighted(dd, col)
            lo, hi = P.bootstrap_ci(per)
            within = np.abs(dd[col] / dd.ystar - 1) <= 0.10
            rows.append(dict(set=label, day=day, method=col, mean_abs_log_err=m, ci_lo=lo, ci_hi=hi,
                             G=G, n_est=n, n_cases=len(dd), share_within10=float(within.mean()) if len(dd) else np.nan))
    return rows


def main():
    if not (C.ROOT / "protocol" / "LOCK.txt").exists():
        raise SystemExit("protocol/LOCK.txt missing: lock the IIS models first")
    refs = references()
    refs.reset_index().to_csv(OUT / "il_references.csv", index=False)
    bottles = C.il_bottle_curves()
    df, curves = P.assemble(bottles, refs, C.CEL_IL, C.CLASS_IL)
    df["eval_lab"] = df.lab.str.startswith("E")
    # synchronous blank and substrate measurements up to the prediction day
    keyed = {k: d for k, d in bottles.groupby(["lab", "test", "material"])}
    def _sync(l, t, m, day):
        d = keyed.get((l, t, m))
        if d is None:
            return False
        d = d[d.time <= day]
        return bool(len(d)) and bool(np.all(d.blank_time == d.time.round(4)))
    df["sync"] = [_sync(l, t, m, day) for l, t, m, day in zip(df.lab, df.test, df.material, df.day)]
    models = pickle.load(open(IIS / "iis_models_locked.pkl", "rb"))
    fold_models = pickle.load(open(IIS / "iis_lolo_fold_models.pkl", "rb"))
    lolo = pd.read_csv(IIS / "iis_lolo_predictions.csv", dtype={"lab": str})
    preds = []
    for day in P.DAYS:
        dd = df[df.day == day]
        pr = P.predict(models[day], dd)
        lo, up = cv_plus(fold_models, lolo, dd.reset_index(drop=True), day)
        pr["pi_lo"], pr["pi_up"] = np.exp(lo), np.exp(up)
        preds.append(pr)
    pred = pd.concat(preds, ignore_index=True)
    pred.drop(columns=["tt", "yy"]).to_csv(OUT / "il_predictions.csv", index=False)
    noncel = pred.base != C.CEL_IL
    val = pred.status == "validated"
    sets = {
        "primary_T1_eval_noncel_sync": val & noncel & (pred['round'] == 1) & pred.eval_lab & pred.sync,
        "T2_eval_noncel_sync": val & noncel & (pred['round'] == 2) & pred.eval_lab & pred.sync,
        "T1T2_eval_noncel_sync": val & noncel & pred.eval_lab & pred.sync,
        "T1_all_labs_noncel_sync": val & noncel & (pred['round'] == 1) & pred.sync,
        "T1_eval_noncel_incl_async": val & noncel & (pred['round'] == 1) & pred.eval_lab,
        "T1_eval_cellulose_sync": val & ~noncel & (pred['round'] == 1) & pred.eval_lab & pred.sync,
        "T1_eval_noncel_relaxedqc_sync": (pred.status_relaxed == "validated") & noncel & (pred['round'] == 1) & pred.eval_lab & pred.sync,
    }
    rows = []
    for k, msk in sets.items():
        rows += summarise(pred, k, msk)
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "il_summary.csv", index=False)
    cov = pred[sets["primary_T1_eval_noncel_sync"] & np.isfinite(pred.pi_lo)]
    cov = cov.assign(hit=(cov.ystar >= cov.pi_lo) & (cov.ystar <= cov.pi_up),
                     width=np.log(cov.pi_up / cov.pi_lo))
    cov.groupby(["day", "cls"]).agg(n=("hit", "size"), coverage=("hit", "mean"), mean_ln_width=("width", "mean"),
                                   labs=("lab", "nunique")).to_csv(OUT / "il_interval_coverage.csv")
    print(s[(s.set == "primary_T1_eval_noncel_sync") & (s.method == "locked")].to_string())


if __name__ == "__main__":
    main()
