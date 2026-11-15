"""Reference BMP checks (IIS): agreement with Hafner et al. (2020) supplementary
subset D, and dispersion of reference BMP (s_r bottle level, s_L between laboratories).

Run: python -I 02_code_v2/scripts/04_reference_checks.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bmp_early import cases as C   # noqa: E402

OUT = C.ROOT / "03_outputs_v2" / "iis"
SI = C.ROOT / "05_literature" / "supplements" / "Hafner2020_SI" / "extracted" / "data"


def main():
    refs = pd.read_csv(OUT / "iis_references.csv")
    d = pd.read_csv(SI / "D.csv")[["test_id", "substrate", "ISR_nom", "inoc_type", "BMP_1p3dnet",
                                    "time_d_1p3dnet", "rate_crit_met_1p3dnet"]]
    d["material"] = d.substrate + "|" + d.ISR_nom.round().astype(int).astype(str)
    d = d[d.inoc_type.isin(["Own", "Own2"]) | ~d.duplicated(["test_id", "material"], keep=False)]
    m = refs.merge(d, left_on=["test", "material"], right_on=["test_id", "material"], how="outer",
                   indicator=True)
    both = m[m._merge == "both"].copy()
    both["d_bmp"] = 100 * (both.ystar / both.BMP_1p3dnet - 1)
    both["d_t"] = both.tstar - both.time_d_1p3dnet
    q = lambda s: s.abs().quantile([0.5, 0.9, 0.95]).round(3).tolist()
    lines = [f"cases: ours {len(refs)}, Hafner D {len(d)}, matched {len(both)}, "
             f"only ours {int((m._merge == 'left_only').sum())}, only D {int((m._merge == 'right_only').sum())}",
             f"|dBMP| % median/p90/p95: {q(both.d_bmp)}",
             f"|dt| d median/p90/p95: {q(both.d_t)}",
             f"cases |dBMP| > 1%: {int((both.d_bmp.abs() > 1).sum())}",
             f"criterion met (ours vs D) crosstab:\n{pd.crosstab(both.complete, both.rate_crit_met_1p3dnet)}"]
    both[both.d_bmp.abs() > 1].to_csv(OUT / "reference_mismatches.csv", index=False)

    # dispersion on the natural-log scale, validated cases
    bot = pd.read_csv(C.ROOT / "03_outputs_v2" / "inventory" / "iis_ystar_biogas_bottles.csv")
    bot = bot.rename(columns={"test.id": "test"}).drop(columns=["substrate"]).rename(columns={"cond": "material"})
    val = refs[refs.status == "validated"][["lab", "test", "material", "ystar"]]
    bv = bot.merge(val, on=["test", "material"])
    bv = bv[bv.cvCH4 > 0]
    bv["lnb"] = np.log(bv.cvCH4)
    w = bv.groupby(["test", "material"]).lnb.agg(["var", "size"]).dropna()
    s_r = {}
    w["base"] = [m.split("|")[0] for m in w.index.get_level_values("material")]
    for mat, g in w.groupby("base"):
        s_r[mat] = float(np.sqrt((g["var"] * (g["size"] - 1)).sum() / (g["size"] - 1).sum()))
    s_r["all"] = float(np.sqrt((w["var"] * (w["size"] - 1)).sum() / (w["size"] - 1).sum()))
    val = val.assign(lny=np.log(val.ystar), lab=val.lab.astype(str))
    s_L = {}
    val["base"] = val.material.str.split("|").str[0]
    for mat, g in val.groupby("base"):
        if g.lab.nunique() >= 5:
            fit = smf.mixedlm("lny ~ 1", g, groups=g["lab"]).fit(reml=True)
            s_L[mat] = float(np.sqrt(fit.cov_re.iloc[0, 0]))
    fit = smf.mixedlm("lny ~ C(base)", val, groups=val["lab"]).fit(reml=True)
    s_L["all"] = float(np.sqrt(fit.cov_re.iloc[0, 0]))
    disp = pd.DataFrame({"s_r": pd.Series(s_r), "s_L": pd.Series(s_L)})
    disp.to_csv(OUT / "dispersion_log_scale.csv")
    lines.append("dispersion (ln scale):\n" + disp.round(4).to_string())
    txt = "\n".join(lines)
    (OUT / "reference_checks.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
