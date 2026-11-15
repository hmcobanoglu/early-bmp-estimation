"""Fully nested sensitivity for the residual-corrected kinetic model, 2016-2018 study, day 10.

In the locked procedure the class-completion curves, the kinetic bound kappa_c, the cellulose mean,
gamma, imputation values and standardisation were estimated on the whole outer training set and
reused in the inner grouped cross-validation that selected the ridge penalty. Here every one of
these quantities is re-estimated inside each inner training fold before the penalty is scored.
The outer test laboratory is never used. The locked 2023-2024 evaluation is not affected.

Run: python -I 02_code_v2/scripts/17_nested_sensitivity.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bmp_early import cases as C        # noqa: E402
from bmp_early import estimators as E   # noqa: E402
from bmp_early import pipeline as P     # noqa: E402

R = C.ROOT / "03_outputs_v2"
DAY = 10


def base_model(train, curves, day):
    """Everything in pipeline.fit except the ridge step; returns model and design."""
    tr = train[(train.status == "validated") & (train.day == day)].copy()
    model = {"day": day, "fbar": P._fbar(tr, curves)}
    tr = tr[tr.y_t > 0]
    model["kappa"] = {c: 1.1 * float(np.nanmax(d.ystar / d.y_t)) for c, d in tr.groupby("cls")}
    cel = tr.loc[tr.base.isin(["CEL", "Cellulose"]), "y_t"]
    model["cel_mean_yt"] = float(cel.mean()) if len(cel) else np.nan
    p1v = np.array([P._p1(model, r) for _, r in tr.iterrows()])
    rel = tr.y_cel_t.to_numpy() / model["cel_mean_yt"]
    ok = np.isfinite(p1v) & np.isfinite(rel) & (rel > 0) & (p1v > 0)
    x, y = np.log(rel[ok]), np.log(tr.ystar.to_numpy()[ok] / p1v[ok])
    model["gamma"] = float((x @ y) / (x @ x)) if ok.sum() >= 5 else 0.0
    X, Y, G = [], [], []
    for _, r in tr.iterrows():
        o = P._p2(model, r)
        if np.isfinite(o["pred"]):
            X.append(E.p3_design(r, o, r.y_cel_t / model["cel_mean_yt"], r.isr, r.cls))
            Y.append(np.log(r.ystar / o["pred"]))
            G.append(r.lab)
    X = np.array(X, float)
    model["impute"] = np.nanmean(X, axis=0)
    X = np.where(np.isfinite(X), X, model["impute"])
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    model.update(mu=mu, sd=sd)
    return model, (X - mu) / sd, np.array(Y), np.array(G)


def design_test(model, df):
    """Standardised design and first-order estimate for rows of df (np.nan where no fit)."""
    Z, p2 = [], []
    for _, r in df.iterrows():
        o = P._p2(model, r)
        if np.isfinite(o["pred"]) and o["pred"] > 0:
            x = np.array(E.p3_design(r, o, r.y_cel_t / model["cel_mean_yt"], r.isr, r.cls), float)
            x = np.where(np.isfinite(x), x, model["impute"])
            Z.append((x - model["mu"]) / model["sd"])
            p2.append(o["pred"])
        else:
            Z.append(np.full(len(model["mu"]), np.nan))
            p2.append(np.nan)
    return np.array(Z), np.array(p2)


def main():
    df, curves = pickle.load(open(R / "iis" / "iis_cases.pkl", "rb"))
    df["lab"] = df.lab.astype(str)
    d = df[df.day == DAY]
    folds = pickle.load(open(R / "iis" / "iis_lolo_fold_models.pkl", "rb"))
    lolo = pd.read_csv(R / "iis" / "iis_lolo_predictions.csv", dtype={"lab": str})
    labs = sorted(d[d.status == "validated"].lab.unique())
    out, sel = [], []
    for g in labs:
        if (DAY, g) not in folds:
            continue
        outer = d[d.lab != g]
        acc = outer[outer.status == "validated"]
        ilabs = acc.lab.to_numpy()
        gkf = GroupKFold(n_splits=min(5, len(np.unique(ilabs))))
        err = np.zeros((len(P.ALPHAS), 0)).tolist()
        scores = {a: [] for a in P.ALPHAS}
        for itr, ite in gkf.split(acc, groups=ilabs):
            tr_labs = set(acc.iloc[itr].lab)
            te = acc.iloc[ite]
            m, Z, Y, _ = base_model(outer[outer.lab.isin(tr_labs)], curves, DAY)
            Zt, p2t = design_test(m, te)
            ok = np.isfinite(p2t)
            yt = np.log(te.ystar.to_numpy()[ok] / p2t[ok])
            for a in P.ALPHAS:
                rg = Ridge(alpha=a).fit(Z, Y)
                scores[a].append(np.mean(np.abs(rg.predict(Zt[ok]) - yt)))
        a_best = min(P.ALPHAS, key=lambda a: np.mean(scores[a]))
        m, Z, Y, _ = base_model(outer, curves, DAY)
        rg = Ridge(alpha=a_best).fit(Z, Y)
        held = d[(d.lab == g)]
        Zh, p2h = design_test(m, held)
        p3 = np.where(np.isfinite(p2h), p2h * np.exp(rg.predict(np.nan_to_num(Zh))), np.nan)
        old = folds[(DAY, g)]
        sel.append(dict(lab=g, alpha_nested=float(a_best), alpha_locked=float(old.get("alpha", np.nan))))
        lo = lolo[(lolo.day == DAY) & (lolo.lab == g)].set_index(["test", "material"])
        for (t_, mat), r, v in zip(held[["test", "material"]].itertuples(index=False), held.itertuples(), p3):
            base = lo.loc[(t_, mat)]
            nested = v if np.isfinite(v) else (base.P1 if np.isfinite(base.P1) and base.P1 > 0 else base.P0)
            out.append(dict(lab=g, test=t_, material=mat, status=r.status, cls=r.cls, ystar=r.ystar,
                            nested=nested, nested_P3=v, locked=base.locked, P1=base.P1))
        print(g, a_best, old.get("alpha"), flush=True)
    o = pd.DataFrame(out)
    o.to_csv(R / "results" / "nested_day10.csv", index=False)
    s = pd.DataFrame(sel)
    v = o[o.status == "validated"]
    res = dict(labs=len(s), alpha_same=int((np.isclose(s.alpha_nested, s.alpha_locked)).sum()),
               male_locked=P.lab_weighted(v, "locked")[0], male_nested=P.lab_weighted(v, "nested")[0],
               male_P1=P.lab_weighted(v, "P1")[0],
               n_nested_P3=int(np.isfinite(v.nested_P3).sum()), n_cond=len(v))
    for c in ("readily", "feed", "straw"):
        res[f"male_nested_{c}"] = P.lab_weighted(v[v.cls == c], "nested")[0]
        res[f"male_locked_{c}"] = P.lab_weighted(v[v.cls == c], "locked")[0]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ea = np.abs(np.log(v.nested / v.ystar)).groupby(v.lab).mean()
    eb = np.abs(np.log(v.P1 / v.ystar)).groupby(v.lab).mean()
    el = np.abs(np.log(v.locked / v.ystar)).groupby(v.lab).mean()
    rng = np.random.default_rng(11)
    for name, diff in (("nested_minus_P1", (ea - eb).dropna()), ("nested_minus_locked", (ea - el).dropna())):
        bs = [rng.choice(diff.values, len(diff)).mean() for _ in range(2000)]
        res[name] = [float(diff.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)),
                     int((diff < 0).sum()), int(len(diff))]
    json.dump(res, open(R / "results" / "nested_day10.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
