"""Main-text Tables 1 and 2 (LaTeX) and the numbers they contain, generated from saved outputs.

Table 1: condition flow in both archives and reference BMP by substrate (incl. 2023-2024 dispersion).
Table 2: day-10 errors of all rules in the three evaluation sets, with paired differences from the
class-completion ratio.

Run: python -I 02_code_v2/scripts/14_manuscript_tables.py
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
TAB = C.ROOT / "04_manuscript_v2" / "tables"
TAB.mkdir(exist_ok=True)
RAW = C.RAW / "IL-BMP_Hafner2026"
B = chr(92)
NUM = {}
RULES = [("P0", "Last observed net yield"), ("P1", "Class-completion ratio"),
         ("C1", "Cellulose-adjusted ratio"), ("P2", "Lag-adjusted first-order fit"),
         ("locked", "Residual-corrected first-order model")]
UNIT = "NmL~CH$_4$~g$^{-1}$~VS"


def f3(x):
    return f"{x:.3f}".replace("-", "$-$")


def paired(df, a, b, B_=2000, seed=11):
    d = df[np.isfinite(df[a]) & np.isfinite(df[b]) & (df[a] > 0) & (df[b] > 0)]
    ea = np.abs(np.log(d[a] / d.ystar)).groupby(d.lab).mean()
    eb = np.abs(np.log(d[b] / d.ystar)).groupby(d.lab).mean()
    diff = (ea - eb).dropna()
    rng = np.random.default_rng(seed)
    bs = [rng.choice(diff.values, len(diff)).mean() for _ in range(B_)]
    return float(diff.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int((diff < 0).sum()), len(diff)


# ---------------- 2023-2024 bottle values for dispersion ----------------
def il_bottles():
    setup = pd.read_csv(RAW / "setup.csv")
    smp = pd.read_csv(RAW / "SMP.csv")
    ts = pd.read_csv(R / "il" / "il_tstar_biogas_bottles.csv").rename(columns={"time_d": "tstar_b"})
    b = setup[setup.substrate != "Blank"].merge(ts[["bottle_id", "tstar_b"]], on="bottle_id", how="left")
    b["material"] = [C.cond_label(m, i) for m, i in zip(b.substrate, C.nominal_isr(b.ISR))]
    vals = []
    for _, r in b.iterrows():
        s = smp[smp.bottle_id == r.bottle_id].sort_values("time_d")
        vals.append(np.interp(r.tstar_b, s.time_d, s.smp) if (np.isfinite(r.tstar_b) and len(s)) else np.nan)
    b["y_b"] = vals
    return b.drop(columns=["test"]).rename(columns={"lab_id": "lab", "file_id": "test"})


def s_r(bot):
    bot = bot[bot.y_b > 0]
    w = bot.groupby(["test", "material"]).y_b.apply(lambda v: np.log(v)).groupby(level=[0, 1]).agg(["var", "size"]).dropna()
    return float(np.sqrt((w["var"] * (w["size"] - 1)).sum() / (w["size"] - 1).sum()))


def d_l(refs):
    out = []
    for _, r in refs.iterrows():
        o = refs[(refs.material == r.material) & (refs.lab != r.lab)]
        if len(o) >= 3:
            out.append(dict(lab=r.lab, d=abs(np.log(r.ystar / o.ystar.median()))))
    d = pd.DataFrame(out)
    return float(d.groupby("lab").d.mean().mean()) if len(d) else np.nan


def table1():
    ri = pd.read_csv(R / "iis" / "iis_references.csv", dtype={"lab": str})
    rl = pd.read_csv(R / "il" / "il_references.csv", dtype={"lab": str})
    rl["ref_lab"] = ~rl.lab.str.startswith("E")
    pi = pd.read_csv(R / "iis" / "iis_lolo_predictions.csv", dtype={"lab": str})
    pl = pd.read_csv(R / "il" / "il_predictions.csv", dtype={"lab": str})
    disp = pd.read_csv(R / "iis" / "dispersion_log_scale.csv", index_col=0)
    add = json.load(open(R / "results" / "additional_results.json"))["between_lab_abs_log_dev_IIS"]

    def cnt(x):
        return [x.lab.nunique(), x.test.nunique(), len(x)]

    acc_i = ri[ri.status == "validated"]
    est_i = pi[(pi.day == 10) & (pi.status == "validated") & pi.locked.notna()]
    flow = [["2016--2018", "Conditions in the dataset"] + cnt(ri),
            ["", "Criterion met in all bottles"] + cnt(ri[ri.complete]),
            ["", "Test met the cellulose criteria (accepted)"] + cnt(acc_i),
            ["", "Estimate available on day 10"] + cnt(est_i)]
    for rd in (1, 2):
        for ref in (False, True):
            x = rl[(rl["round"] == rd) & (rl.ref_lab == ref)]
            who = "reference" if ref else "other"
            lab = f"2023--2024, round {rd}, {who} labs"
            flow += [[lab, "Conditions in the dataset"] + cnt(x),
                     ["", "Accepted (all substrates)"] + cnt(x[x.status == "validated"])]
            if rd == 1 and not ref:
                a = x[(x.status == "validated") & ~x.material.str.startswith(C.CEL_IL)]
                pr = pl[(pl.day == 10) & (pl["round"] == 1) & pl.lab.str.startswith("E") & (pl.status == "validated")
                        & (pl.base != C.CEL_IL) & pl.sync]
                flow += [["", "Accepted, non-cellulose"] + cnt(a),
                         ["", "Synchronous blank measurements (primary set)"] + cnt(pr)]
    NUM["flow"] = flow
    # reference by substrate
    names = {"CEL": "Microcrystalline cellulose", "SA": "Feed SA (pig feed)", "SB": "Feed SB",
             "SC": "Feed SC (lipid-rich)", "SD": "Wheat straw"}
    acc_i = acc_i.assign(base=acc_i.material.str.split("|").str[0])
    rows = []
    for k, nm in names.items():
        g = acc_i[acc_i.base == k]
        rows.append(["2016--2018" if k == "CEL" else "", nm, len(g), g.lab.nunique(),
                     f"{g.ystar.mean():.0f} $\\pm$ {g.ystar.std():.0f}", f"{100 * g.ystar.std() / g.ystar.mean():.1f}",
                     f"{g.tstar.median():.1f} ({g.tstar.quantile(.25):.1f}--{g.tstar.quantile(.75):.1f})",
                     f"{disp.loc[k, 's_r']:.3f}", f"{add[k]:.3f}"])
    rows.append(["", "All substrates", len(acc_i), acc_i.lab.nunique(), "", "", "", f"{disp.loc['all', 's_r']:.3f}", f"{add['all']:.3f}"])
    acc_l = rl[(rl["round"] == 1) & (rl.status == "validated")].copy()
    acc_l["base"] = acc_l.material.str.split("|").str[0]
    bot = il_bottles()
    bot = bot.merge(acc_l[["lab", "test", "material"]], on=["lab", "test", "material"])
    il_disp = {}
    for i, k in enumerate(["Cellulose", "Rabbit feed", "Sunflower seeds", "Wheat straw"]):
        g = acc_l[acc_l.base == k]
        sr, dl = s_r(bot[bot.material.str.startswith(k + "|")]), d_l(g)
        il_disp[k] = dict(s_r=sr, d_L=dl, n=len(g), labs=int(g.lab.nunique()))
        rows.append(["2023--2024, round 1" if i == 0 else "", "Microcrystalline cellulose" if k == "Cellulose" else k,
                     len(g), g.lab.nunique(), f"{g.ystar.mean():.0f} $\\pm$ {g.ystar.std():.0f}", f"{100 * g.ystar.std() / g.ystar.mean():.1f}",
                     f"{g.tstar.median():.1f} ({g.tstar.quantile(.25):.1f}--{g.tstar.quantile(.75):.1f})",
                     f"{sr:.3f}", f"{dl:.3f}"])
    sr_all, dl_all = s_r(bot), d_l(acc_l)
    il_disp["all"] = dict(s_r=sr_all, d_L=dl_all)
    rows.append(["", "All substrates", len(acc_l), acc_l.lab.nunique(), "", "", "", f"{sr_all:.3f}", f"{dl_all:.3f}"])
    NUM["IL_dispersion_round1_accepted"] = il_disp

    L = [B + "begin{table}[!htbp]",
         B + "caption{Test conditions in the two inter-laboratory studies and reference BMP of accepted conditions. (a) Number of laboratories, tests (data files in 2023--2024) and conditions at each step. (b) Reference BMP (mean $\\pm$ SD and relative standard deviation between conditions), day at which the 1\\% net 3 d criterion was met (median and interquartile range), within-condition standard deviation of ln bottle BMP ($s_r$) and mean absolute log deviation of a laboratory's reference from the median of the other laboratories ($d_L$).}",
         B + "label{tab:data}", B + "centering" + B + "footnotesize",
         B + "resizebox{0.85" + B + "textwidth}{!}{%", B + "begin{tabular}{llrrr}", B + "toprule",
         B + "multicolumn{5}{l}{" + B + "textit{(a) Condition flow}}" + B + B,
         "Study and group & Step & Laboratories & Tests & Conditions" + B + B, B + "midrule"]
    L += [" & ".join(str(v) for v in r) + B + B for r in flow]
    L += [B + "bottomrule", B + "end{tabular}}", "", B + "vspace{1em}", "",
          B + "resizebox{" + B + "textwidth}{!}{%", B + "begin{tabular}{llrrccccc}", B + "toprule",
          B + "multicolumn{9}{l}{" + B + "textit{(b) Reference BMP of accepted conditions}}" + B + B,
          "Study & Substrate & $n$ & Labs & BMP (" + UNIT + ") & RSD (\\%) & Criterion day (d) & $s_r$ & $d_L$" + B + B, B + "midrule"]
    for r in rows:
        if r[0].startswith("2023"):
            L.append(B + "midrule")
        L.append(" & ".join(str(v) for v in r) + B + B)
    L += [B + "bottomrule", B + "end{tabular}}", B + "end{table}"]
    (TAB / "table1.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def table2():
    pi = pd.read_csv(R / "iis" / "iis_lolo_predictions.csv", dtype={"lab": str})
    pl = pd.read_csv(R / "il" / "il_predictions.csv", dtype={"lab": str})
    sets = [("2016--2018, accepted conditions (leave-one-laboratory-out)", pi[(pi.day == 10) & (pi.status == "validated")]),
            ("2016--2018, failing the cellulose criteria (leave-one-laboratory-out)", pi[(pi.day == 10) & (pi.status == "outside_qc")]),
            ("2023--2024, primary set (methods fixed on 2016--2018)",
             pl[(pl.day == 10) & (pl.status == "validated") & (pl.base != C.CEL_IL) & (pl["round"] == 1)
                & pl.lab.str.startswith("E") & pl.sync])]
    L = [B + "begin{table}[!htbp]",
         B + "caption{Day-10 errors of early BMP estimates. MALE: laboratory-weighted mean absolute log error (95\\% CI, laboratory bootstrap); the relative deviation $" + B + "exp(" + B + "mathrm{MALE})-1$ is given in the text. Signed: laboratory-weighted mean of $" + B + "ln(" + B + "hat{B}_t/B_{" + B + "mathrm{ref}})$. Within 10\\%: conditions within $" + B + "pm$10\\% of the reference out of all eligible conditions (laboratory-weighted share); conditions without an estimate count as outside. Last column: paired laboratory-level difference in mean absolute log error from the class-completion ratio (95\\% CI; laboratories in which the method had the smaller error); negative values indicate a smaller error than the ratio.}",
         B + "label{tab:error}", B + "centering" + B + "footnotesize", B + "resizebox{" + B + "textwidth}{!}{%",
         B + "begin{tabular}{lcccccc}", B + "toprule",
         "Estimation method & Labs & Conditions with estimate & MALE (95\\% CI) & Signed & Within 10\\% & Difference from class-completion ratio" + B + B,
         B + "midrule"]
    out = {}
    for title, d in sets:
        L.append(B + "multicolumn{7}{l}{" + B + "textit{" + title + "}}" + B + B)
        out[title] = {}
        for code, name in RULES:
            m, G, n, per = P.lab_weighted(d, code)
            lo, hi = P.bootstrap_ci(per)
            sg = P.lab_weighted(d, code, fn=lambda e: e)[0]
            ok = np.isfinite(d[code]) & (d[code] > 0)
            within = (np.abs(d[code] / d.ystar - 1) <= 0.10) & ok
            w10 = within.groupby(d.lab).mean().mean()
            kn = f"{int(within.sum())}/{len(d)}"
            if code == "P1":
                pd_txt = "--"
                pdv = None
            else:
                pdv = paired(d, code, "P1")
                pd_txt = f"{f3(pdv[0])} ({f3(pdv[1])} to {f3(pdv[2])}; {pdv[3]}/{pdv[4]})"
            out[title][code] = dict(male=m, ci=[lo, hi], G=G, n=n, signed=sg, w10=w10, kn=kn, paired=pdv)
            L.append(f"{name} & {G} & {n} & {f3(m)} ({f3(lo)}--{f3(hi)}) & {f3(sg)} & {kn} ({w10:.2f}) & {pd_txt}" + B + B)
        L.append(B + "midrule")
    L[-1] = B + "bottomrule"
    L += [B + "end{tabular}}", B + "end{table}"]
    (TAB / "table2.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    NUM["table2"] = out


def paired_all():
    """Single source of all paired comparisons (main text and supplement)."""
    pi = pd.read_csv(R / "iis" / "iis_lolo_predictions.csv", dtype={"lab": str})
    pl = pd.read_csv(R / "il" / "il_predictions.csv", dtype={"lab": str})
    prim = pl[(pl.status == "validated") & (pl.base != C.CEL_IL) & (pl["round"] == 1) & pl.lab.str.startswith("E") & pl.sync]
    sets = [("2016--2018 accepted", pi[pi.status == "validated"], (7, 10, 14)),
            ("2016--2018 failing criteria", pi[pi.status == "outside_qc"], (10,)),
            ("2023--2024 primary", prim, (10, 14))]
    pairs = [("locked", "P1"), ("P3", "P2"), ("locked", "P0"), ("C1", "P1")]
    out = []
    for nm, d, days in sets:
        for day in days:
            dd = d[d.day == day]
            for a, b in pairs:
                v = paired(dd, a, b)
                n = int((np.isfinite(dd[a]) & np.isfinite(dd[b]) & (dd[a] > 0) & (dd[b] > 0)).sum())
                out.append(dict(set=nm, day=day, a=a, b=b, diff=v[0], lo=v[1], hi=v[2], a_better=v[3], labs=v[4],
                                conditions=n))
    NUM["paired"] = out


if __name__ == "__main__":
    table1()
    table2()
    paired_all()
    json.dump(NUM, open(R / "results" / "manuscript_numbers.json", "w"), indent=1, default=str)
    sys.stdout.reconfigure(encoding="utf-8")
    print((TAB / "table1.tex").read_text(encoding="utf-8"))
    print((TAB / "table2.tex").read_text(encoding="utf-8"))
