"""IIS-BMP inventory: durations, measurement density, recalculated Y*, QC status,
day-10 eligibility and reconciliation with the published SMP.csv.

Run:  python -I 02_code_v2/scripts/01_inventory_iis.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bmp_early import iis  # noqa: E402

OUT = iis.ROOT / "03_outputs_v2" / "inventory"
OUT.mkdir(parents=True, exist_ok=True)
T_EVAL = (5, 7, 10, 14, 21)


def main():
    tests, bottles, gas, smp = iis.load()
    smp = smp.merge(bottles[["bottle.id", "test.id", "lab.id", "study", "substrate"]],
                    on="bottle.id", how="left")

    # --- bottle-level measurement summary
    g = gas.merge(bottles[["bottle.id", "substrate", "test.id"]], on="bottle.id")
    per_bottle = g.groupby("bottle.id").agg(t_last=("time.d", "max"),
                                           n_obs=("time.d", "size"))
    per_bottle["n_obs_le10"] = g[(g["time.d"] > 0) & (g["time.d"] <= 10)] \
        .groupby("bottle.id").size().reindex(per_bottle.index, fill_value=0)
    per_bottle = per_bottle.join(bottles.set_index("bottle.id")[["substrate", "test.id"]])

    # --- recalculated net SMP and reference per unit
    net = iis.net_smp_bottle(bottles, gas)
    units = []
    for (test_id, sub), b in net.groupby(["test.id", "substrate"]):
        grid, y, nb = iis.unit_curve(b, "smp_net")
        t_term, ystar = iis.termination_1pct_3d(grid, y)
        hb = smp[(smp["test.id"] == test_id) & (smp["substrate"] == sub)]
        hb = hb.rename(columns={"SMP": "smp_h"})
        hgrid, hy, _ = iis.unit_curve(hb, "smp_h") if not hb.empty else (grid, y * np.nan, 0)
        _, ystar_h = iis.termination_1pct_3d(hgrid, hy) if not hb.empty else (np.nan, np.nan)
        pb = per_bottle[(per_bottle["test.id"] == test_id) & (per_bottle["substrate"] == sub)]
        row = dict(test_id=test_id, lab_id=b["lab.id"].iloc[0], study=b["study"].iloc[0],
                   substrate=sub, material_class=iis.MATERIAL_CLASS.get(sub, "other"),
                   n_bottles=nb, t_last_common=grid[-1], t_term=t_term, ystar=ystar,
                   ystar_hafner_smp=ystar_h,
                   min_obs_le10=pb["n_obs_le10"].min() if len(pb) else 0,
                   tbmp=iis.TBMP.get(sub, np.nan))
        for t in T_EVAL:
            row[f"y_at_{t}"] = y[t] if t < len(y) else np.nan
        units.append(row)
    units = pd.DataFrame(units)

    # bottle-level cellulose BMP for test QC (each bottle at the unit's t_term)
    qc = []
    for test_id, b in net[net["substrate"] == iis.CELLULOSE].groupby("test.id"):
        u = units[(units["test_id"] == test_id) & (units["substrate"] == iis.CELLULOSE)]
        t_term = u["t_term"].iloc[0] if len(u) else np.nan
        if not np.isfinite(t_term):
            qc.append(dict(test_id=test_id, cel_mean=np.nan, cel_rsd=np.nan, qc="cel_not_terminated"))
            continue
        vals = []
        for _, bb in b.groupby("bottle.id"):
            bb = bb.dropna().sort_values("time.d")
            if bb["time.d"].max() >= t_term:
                vals.append(np.interp(t_term, bb["time.d"], bb["smp_net"]))
        vals = np.array(vals)
        m, s = vals.mean(), vals.std(ddof=1) if len(vals) > 1 else np.nan
        rsd = 100 * s / m if m > 0 else np.nan
        ok = iis.CEL_RANGE[0] <= m <= iis.CEL_RANGE[1] and (np.isfinite(rsd) and rsd <= iis.CEL_RSD_MAX)
        qc.append(dict(test_id=test_id, cel_mean=m, cel_rsd=rsd, qc="pass" if ok else "fail"))
    qc = pd.DataFrame(qc)
    units = units.merge(qc, on="test_id", how="left")
    units["qc"] = units["qc"].fillna("no_cellulose")
    units["status"] = np.where(~np.isfinite(units["ystar"]), "c_not_observed",
                               np.where(units["qc"] == "pass", "a_valid", "b_qc_not_passed"))
    units["eligible_day10"] = (units["min_obs_le10"] >= 3) & np.isfinite(units["y_at_10"])
    units["ystar_gt_tbmp"] = units["ystar"] > units["tbmp"]
    units["rel_diff_vs_hafner"] = 100 * (units["ystar"] / units["ystar_hafner_smp"] - 1)
    units.to_csv(OUT / "iis_units.csv", index=False)
    per_bottle.to_csv(OUT / "iis_bottles_measurement_summary.csv")

    # --- summaries
    lines = []
    p = lines.append
    p(f"labs: {bottles['lab.id'].nunique()}  tests: {bottles['test.id'].nunique()}  "
      f"substrate units: {len(units)}")
    p("\nunits by substrate x status:\n" + units.pivot_table(
        index="substrate", columns="status", values="test_id", aggfunc="count",
        fill_value=0).to_string())
    p("\nqc of tests (cellulose criterion):\n" + qc["qc"].value_counts().to_string())
    p("\ntermination day (t_term) by substrate, status a:\n" + units[units.status == "a_valid"]
      .groupby("substrate")["t_term"].describe()[["count", "min", "25%", "50%", "75%", "max"]].to_string())
    p("\ncommon last day by substrate:\n" + units.groupby("substrate")["t_last_common"]
      .describe()[["min", "25%", "50%", "75%", "max"]].to_string())
    a10 = units[(units.status == "a_valid") & units.eligible_day10]
    p(f"\nday-10 eligible & status a: units {len(a10)}, labs G = {a10['lab_id'].nunique()}")
    p("by class:\n" + a10.groupby("material_class").agg(
        units=("test_id", "size"), labs=("lab_id", "nunique")).to_string())
    for t in T_EVAL:
        done = (units.status == "a_valid") & (units["t_term"] <= t)
        p(f"status a units already terminated by day {t}: {int(done.sum())}")
    p("\nY* vs Y* from published SMP.csv (same termination rule), % difference:\n"
      + units["rel_diff_vs_hafner"].describe().to_string())
    p(f"\nunits with Y* > TBMP: {int(units['ystar_gt_tbmp'].sum())}")
    p("\nmeasurement methods (tests):\n" + tests["method.biogas"].value_counts().to_string())
    txt = "\n".join(lines)
    (OUT / "iis_inventory_summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
