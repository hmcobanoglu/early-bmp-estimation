"""Supplementary tables (LaTeX) generated from analysis outputs; written to
04_manuscript_v2/supplementary_tables.tex and included by supplementary.tex.

Run after 07, 12, 13 and 14:  python -I 02_code_v2/scripts/11_supplementary.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bmp_early import cases as C        # noqa: E402
from bmp_early import pipeline as P     # noqa: E402

R = C.ROOT / "03_outputs_v2"
OUT = C.ROOT / "04_manuscript_v2"
B = chr(92)
NAME = {"locked": "Residual-corrected first-order model", "P0": "Last observed net yield",
        "P1": "Class-completion ratio", "P2": "Lag-adjusted first-order fit",
        "C1": "Cellulose-adjusted ratio", "P3": "Corrected first-order fit without fallback"}
CLS = {"all": "All substrates", "readily": "Microcrystalline cellulose", "feed": "Animal feeds", "straw": "Wheat straw"}
ADD = json.load(open(R / "results" / "additional_results.json"))
CHK = json.load(open(R / "results" / "review_checks.json"))


def m(x):
    return f"{x:.3f}".replace("-", "$-$")


def ci(v, lo, hi):
    return f"{m(v)} ({m(lo)}--{m(hi)})"


def table(caption, label, head, rows, spec):
    lines = [B + "begin{table}[p]", B + "caption{" + caption + "}", B + "label{" + label + "}",
             B + "centering" + B + "footnotesize", B + "resizebox{" + B + "textwidth}{!}{%",
             B + "begin{tabular}{" + spec + "}", B + "toprule", " & ".join(head) + B + B, B + "midrule"]
    for r in rows:
        lines.append(B + "midrule" if r == "MID" else " & ".join(str(x) for x in r) + B + B)
    lines += [B + "bottomrule", B + "end{tabular}}", B + "end{table}", B + "clearpage"]
    return "\n".join(lines)


def s1():
    FC = json.load(open(R / "results" / "final_checks.json"))
    il = pd.read_csv(R / "il" / "il_references.csv")
    il = il[il.complete]
    d = (100 * (il.ystar / il.bmp_calc - 1)).abs().dropna()
    raw = C.RAW / "IL-BMP_Hafner2026"
    smp = pd.read_csv(raw / "SMP.csv")
    calc = pd.read_csv(raw / "BMP_calc.csv")
    last = smp.sort_values("time_d").groupby("bottle_id").smp.last().rename("smp_last")
    fl = (100 * (calc.merge(last, on="bottle_id").pipe(lambda x: x.bmp_calc / x.smp_last) - 1)).abs()
    A, D = CHK["A_IL_smp_recomputed"], CHK["D_IIS_common_day_vs_own_day_pct"]
    Ia, Ib = FC["a_IIS_both_complete"], FC["b_IL_primary_recalc_upto_day10"]

    def q(v):
        return [f"{v.median():.3f}", f"{v.quantile(.9):.3f}", f"{v.quantile(.95):.3f}"]

    rows = [["2016--2018", "Recalculated reference vs organiser values (Hafner et al., 2020), conditions complete in both",
             "Conditions", Ia["n"], f"{Ia['median']:.3f}", f"{Ia['p90']:.3f}", f"{Ia['p95']:.3f}"],
            ["2016--2018", "Reference at each bottle's criterion day vs bottle mean at the condition completion day",
             "Accepted conditions", D["n"], f"{abs(D['median']):.3f}", f"{D['p90_abs']:.3f}", f"{D['p95_abs']:.3f}"],
            ["2023--2024", "Organisers' net SMP vs net SMP recalculated from cumulative methane and blank bottles",
             "Observations", A["points"], f"{A['median_abs_pct']:.3f}", f"{A['p90_abs_pct']:.3f}", f"{A['p95_abs_pct']:.3f}"],
            ["2023--2024", "As above, bottles of the 16 primary conditions, records up to day 10 only",
             "Observations", Ib["points"], f"{Ib['median_abs_pct']:.3f}", f"{Ib['p90_abs_pct']:.3f}", f"{Ib['p95_abs_pct']:.3f}"],
            ["2023--2024", "Organisers' bottle BMP vs net SMP at the last measurement", "Bottles", len(fl)] + q(fl),
            ["2023--2024", "Recalculated reference vs organisers' bottle BMP, complete conditions", "Conditions", len(d)] + q(d)]
    miss = FC["a_IIS_bottles_without_gas"]
    t = table("Checks of the reference BMP and of the net methane curves (absolute relative difference, \\%). "
              f"Criterion status in the 2016--2018 study: of {Ia['matched']} conditions matched to the organiser values, 366 met the criterion in both calculations, 4 in neither, 17 only in the recalculation and 10 only in the organiser values. "
              f"Changing the reference day changed one accepted condition by more than 1\\% (maximum {D['max']:.1f}\\%). "
              f"{len(miss)} of {FC['a_IIS_setup_bottles']} bottles listed in the 2016--2018 setup had no gas records and were not used "
              f"({sum(b['substrate'] != 'BK' for b in miss)} substrate and {sum(b['substrate'] == 'BK' for b in miss)} inoculum-only bottles in "
              f"{len({b['test.id'] for b in miss})} tests, {sum('validated' in v for v in FC['a_IIS_tests_affected_status'].values())} of them accepted).",
              "tab:s1", ["Study", "Comparison", "Unit", "$n$", "Median", "90th pct", "95th pct"], rows, "p{1.6cm}p{7.4cm}lrrrr")
    return t


def s2():
    mt = pd.read_csv(R / "results" / "metrics.csv")
    days = [5, 7, 10, 14, 21]
    rows = []

    def add(setname, cls, methods, label):
        d = mt[(mt.set == setname) & (mt.cls == cls)]
        g = d[d.method == "locked"].set_index("day")
        rows.append([B + "textit{" + label + "}", ""] + [f"{int(g.loc[x, 'n_est'])}/{int(g.loc[x, 'n_cases'])} ({int(g.loc[x, 'G'])})" for x in days])
        for me in methods:
            x = d[d.method == me].set_index("day")
            rows.append(["", NAME[me]] + [m(x.loc[k, "mae_log"]) for k in days])

    for c in ("all", "readily", "feed", "straw"):
        add("IIS_LOLO_validated", c, ["locked", "P1", "C1", "P2", "P0"] if c == "all" else ["locked", "P1", "P0"],
            "Accepted, " + CLS[c].lower())
    add("IIS_LOLO_outsideQC", "all", ["locked", "P1", "C1", "P0"], "Failing the cellulose criteria")
    add("IIS_LOLO_ongoing", "all", ["locked", "P1", "P0"], "Accepted, criterion not yet met on the test day")
    X = json.load(open(R / "results" / "supplement_extras.json"))
    fo = X["first_order_available"]
    rows.append([B + "textit{First-order fit available}", ""] + [f"{fo[str(k)]}/206" for k in days])
    return table("Laboratory-weighted mean absolute log error by test day in the 2016--2018 study (leave-one-laboratory-out). Group rows give conditions with an estimate / eligible conditions (laboratories with an estimate) for the residual-corrected first-order model. "
                 f"On day 10, {X['day10_abs_err_gt_0.2']} accepted conditions had an absolute log error above 0.2 with this model, and the largest laboratory mean was {X['day10_max_lab_mean']:.2f}.",
                 "tab:s2", ["Set", "Method", "Day 5", "Day 7", "Day 10", "Day 14", "Day 21"], rows, "llccccc")


def s3():
    rows = []
    base = pd.read_csv(R / "iis" / "iis_lolo_predictions.csv", dtype={"lab": str})
    rel = pd.read_csv(R / "iis" / "iis_lolo_predictions_relaxedqc.csv", dtype={"lab": str})
    sens = pd.read_csv(R / "iis" / "iis_lolo_sensitivity.csv", dtype={"lab": str})
    sets = [("Primary analysis", base[base.status == "validated"]),
            ("Relaxed cellulose screen (330--415, RSD $\\le$15\\%)", rel[rel.status_relaxed == "validated"] if "status_relaxed" in rel else rel[rel.status == "validated"]),
            ("Modified Gompertz model in place of first-order", sens[(sens.variant == "gompertz") & (sens.status == "validated")]),
            ("Upper bound from theoretical BMP", sens[(sens.variant == "first_order_tbmp_cap") & (sens.status == "validated")])]
    for name, d in sets:
        vals = []
        for day in (5, 10, 21):
            mm, G, n, per = P.lab_weighted(d[d.day == day], "locked")
            lo, hi = P.bootstrap_ci(per)
            vals.append(ci(mm, lo, hi))
        G = P.lab_weighted(d[d.day == 10], "locked")[1]
        rows.append([name, G] + vals)
    rl = json.load(open(R / "results" / "supplement_extras.json"))["reference_last_measurement"]
    nested = json.load(open(R / "results" / "nested_day10.json"))
    rows.append(["Fully nested penalty selection", 26, "", m(nested["male_nested"]), ""])
    rows.append(["Reference at the last measurement", 26] + [m(rl[f"day{d}_locked"][1]) for d in (5, 10, 21)])
    rows.append(["Reference at the last measurement, class-completion ratio", 26] + [m(rl[f"day{d}_P1"][1]) for d in (5, 10, 21)])
    return table("Sensitivity analyses in the 2016--2018 study (residual-corrected first-order model unless stated; mean absolute log error with 95\\% CI where computed). For the last two rows the methods were unchanged and only the reference was replaced by the bottle mean at the last measurement, which was on median "
                 f"{rl['ref_last_vs_ref_pct_median']:.1f}\\% higher than the BMP at the 1\\% net 3 d criterion; the class-completion ratio gave {m(rl['day10_P1'][0])} on day 10 with the criterion reference.",
                 "tab:s3", ["Analysis", "Laboratories", "Day 5", "Day 10", "Day 21"], rows, "lrccc")


def s4():
    s = pd.read_csv(R / "il" / "il_summary.csv")
    lab = {"primary_T1_eval_noncel_sync": "Round 1, other laboratories (primary set)",
           "T2_eval_noncel_sync": "Round 2, other laboratories",
           "T1T2_eval_noncel_sync": "Rounds 1 and 2, other laboratories",
           "T1_all_labs_noncel_sync": "Round 1, including reference laboratories",
           "T1_eval_noncel_incl_async": "Round 1, other laboratories, including asynchronous blanks",
           "T1_eval_noncel_relaxedqc_sync": "Round 1, other laboratories, relaxed cellulose screen",
           "T1_eval_cellulose_sync": "Round 1, other laboratories, cellulose"}
    rows = []
    for k, name in lab.items():
        r10 = s[(s.set == k) & (s.day == 10) & (s.method == "locked")].iloc[0]
        r14 = s[(s.set == k) & (s.day == 14) & (s.method == "locked")].iloc[0]
        p10 = s[(s.set == k) & (s.day == 10) & (s.method == "P1")].iloc[0]
        rows.append([name, int(r10.G), int(r10.n_cases), ci(r10.mean_abs_log_err, r10.ci_lo, r10.ci_hi),
                     m(p10.mean_abs_log_err), ci(r14.mean_abs_log_err, r14.ci_lo, r14.ci_hi)])
    rows.append("MID")
    F = CHK["F_IL_primary"]
    for mat, v in F["10"]["by_material"].items():
        v14 = F["14"]["by_material"][mat]
        rows.append([f"Primary set, {mat.lower()}", v["labs"], v["n"], m(v["male"]), m(v["male_P1"]), m(v14["male"])])
    lo, hi = F["10"]["drop_one_lab_range"]
    rows.append(["Primary set, one laboratory left out (range)", 5, "", f"{m(lo)}--{m(hi)}", "", ""])
    return table("Evaluation in the 2023--2024 study with methods fixed on the 2016--2018 study: mean absolute log error of the residual-corrected first-order model (95\\% CI) and of the class-completion ratio. The reference laboratories contributed few conditions because most of them measured blanks at times different from the substrate bottles: of their 15 accepted non-cellulose conditions in round 1, 6 conditions from 2 laboratories had synchronous blank measurements.",
                 "tab:s4", ["Evaluation set", "Laboratories", "Conditions", "Day 10, corrected model", "Day 10, ratio", "Day 14, corrected model"],
                 rows, "lrrccc")


def s5():
    c = pd.read_csv(R / "il" / "il_interval_coverage.csv")
    lolo = pd.read_csv(R / "iis" / "iis_lolo_predictions.csv", dtype={"lab": str})
    cal = lolo[(lolo.status == "validated") & np.isfinite(lolo.locked)]
    cls = {"feed": "Feeds and seeds", "straw": "Wheat straw"}
    rows = []
    for r in c.itertuples():
        k = cal[(cal.day == r.day) & (cal.cls == r.cls)]
        rows.append([int(r.day), cls[r.cls], k.lab.nunique(), len(k), int(r.n), int(r.labs),
                     f"{int(round(r.coverage * r.n))}/{int(r.n)}", f"{r.mean_ln_width:.2f}", f"{np.exp(r.mean_ln_width):.2f}"])
    return table("Observed coverage and width of the nominal 90\\% prediction intervals of the residual-corrected first-order model in the primary 2023--2024 set. Calibration: laboratories and accepted conditions of the 2016--2018 study contributing residuals for the class and day. Width is the mean of ln(upper/lower); its exponential is the geometric mean ratio of upper to lower limit. No interval was computed for the class-completion ratio.",
                 "tab:s5", ["Day", "Class", "Calibration labs", "Calibration conditions", "Evaluated conditions", "Evaluated labs", "Reference inside", "Mean ln width", "Upper/lower"],
                 rows, "rlrrrrrrr")


def s6():
    N = json.load(open(R / "results" / "manuscript_numbers.json"))["paired"]
    rows = []
    prev = None
    for p in N:
        if prev is not None and p["set"] != prev:
            rows.append("MID")
        prev = p["set"]
        rows.append([p["set"], int(p["day"]), NAME[p["a"]], NAME[p["b"]],
                     f"{m(p['diff'])} ({m(p['lo'])} to {m(p['hi'])})", f"{p['a_better']}/{p['labs']}", int(p["conditions"])])
    return table("Paired laboratory-level comparisons: difference in mean absolute log error (method A minus method B) on the conditions with both estimates, 95\\% bootstrap CI over laboratories, and laboratories in which method A had the smaller error. Table 2 and the text use the same values.",
                 "tab:s6", ["Set", "Day", "Method A", "Method B", "Difference (95\\% CI)", "A smaller", "Conditions"],
                 rows, "lrllcrr")


def s7():
    E = CHK["E_cellulose_groups_heldout_mean"]
    rows = []
    for col in ("locked", "P1"):
        for k, v in E[col].items():
            rows.append([NAME[col] if k == "<0.9" else "", k.replace("<", "$<$").replace(">", "$>$").replace("-", "--"),
                         v["n"], v["labs"], m(v["male"]), m(v["bias"]), f"{v['gmr']:.2f}", f"{v['share_e_pos']:.2f}", f"{v['share_outside']:.2f}"])
    T = CHK["E_test_level"]
    rows.append("MID")
    tot = sum(T.values())
    rows.append([B + "multicolumn{9}{l}{Tests (n = " + str(tot) + ") with a day-10 cellulose ratio below 0.9: " +
                 f"{T['low_other']} failing the cellulose criteria, {T['low_validated']} accepted; at or above 0.9: "
                 f"{T['not_low_other']} failing, {T['not_low_validated']} accepted." + "}"])
    return table("Day-10 errors of non-cellulose conditions of the 2016--2018 study (accepted and failing the cellulose criteria) by the day-10 cellulose yield of the same test, relative to the mean of accepted cellulose conditions in the other laboratories. The 0.9 and 1.1 limits were set after the evaluation. GMR: geometric mean ratio of estimate to reference. Of the 124 tests, two had no cellulose control and one had no cellulose value by day 10.",
                 "tab:s7", ["Method", "Cellulose ratio", "Conditions", "Laboratories", "MALE", "Signed", "GMR", "Share above reference", "Share failing criteria"],
                 rows, "llrrrrrrr")


def s9():
    t = pd.read_csv(R / "results" / "completion_fractions_locked.csv")
    t = t[t.day == 10]
    rows = [[CLS[c], *[f"{t[t.cls == c][f'f_{d}'].iloc[0]:.3f}" for d in (5, 7, 10, 14, 21)]] for c in ("readily", "feed", "straw")]
    return table("Class-completion fractions $" + B + "bar{f}_c(t)$ of the methods fixed on all accepted conditions of the 2016--2018 study (fraction of the reference BMP reached on day $t$). Values between the listed days are interpolated linearly.",
                 "tab:s9", ["Class", "Day 5", "Day 7", "Day 10", "Day 14", "Day 21"], rows, "lccccc")


def s8():
    f = pd.read_csv(R / "results" / "first_day.csv")
    rows = []
    for c in ("readily", "feed", "straw"):
        for _, r in f[f.cls == c].sort_values("tol").iterrows():
            fd = "none" if pd.isna(r.first_day) else int(r.first_day)
            rows.append([CLS[c] if r.tol == 0.05 else "", f"$" + B + f"pm${int(r.tol * 100)}\\%", fd,
                         f"{r.boot_share_point:.2f}" if fd != "none" else "--", f"{r.boot_share_none:.2f}",
                         f"{r.median_tstar:.1f}" if r.tol == 0.05 else ""])
    return table("First evaluated day (5, 7, 10, 14 or 21 d) at which the laboratory-weighted share of accepted conditions within the tolerance of the reference BMP reached 90\\% (2016--2018, residual-corrected first-order model), with the share of 500 laboratory bootstrap resamples selecting the same day and selecting no day.",
                 "tab:s8", ["Class", "Tolerance", "First day", "Same day", "No day", "Median criterion day"],
                 rows, "llcccc")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    body = "\n\n".join([s1(), s2(), s3(), s4(), s5(), s6(), s7(), s8(), s9()])
    (OUT / "supplementary_tables.tex").write_text(body + "\n", encoding="utf-8")
    print("written")


if __name__ == "__main__":
    main()
