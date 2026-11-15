"""Supplementary extras (descriptive):

a. Class-completion fractions of the methods fixed on all 2016-2018 data, for a worked example.
b. Sensitivity of the day-10 error to the reference definition: bottle mean at the last
   measurement instead of the BMP at the 1% net 3 d criterion (2016-2018, accepted conditions).
c. Counts quoted in the text: first-order fits available by day, conditions with day-10 error
   above 0.2, largest laboratory mean error.

Run: python -I 02_code_v2/scripts/19_supplement_extras.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bmp_early import cases as C      # noqa: E402
from bmp_early import pipeline as P   # noqa: E402

R = C.ROOT / "03_outputs_v2"
OUT = R / "results"
res = {}


def completion_table():
    models = pickle.load(open(R / "iis" / "iis_models_locked.pkl", "rb"))
    rows = []
    for day, m in models.items():
        for cls, (g, f) in m["fbar"].items():
            rows.append(dict(day=day, cls=cls, **{f"f_{t}": float(np.interp(t, g, f)) for t in (5, 7, 10, 14, 21)}))
    t = pd.DataFrame(rows)
    # the curve fitted on day-10 training data is the one used for day-10 estimates
    t.to_csv(OUT / "completion_fractions_locked.csv", index=False)
    d10 = t[t.day == 10].set_index("cls")
    res["f_day10_at_day10"] = d10["f_10"].round(4).to_dict()
    res["f_day10_at_day9"] = {c: float(np.interp(9, *models[10]["fbar"][c])) for c in models[10]["fbar"]}


def last_measurement_reference():
    bc = C.iis_bottle_curves()
    last = bc.sort_values("time").groupby(["lab", "test", "material", "bottle"]).smp.last().reset_index()
    ref_last = last.groupby(["lab", "test", "material"]).smp.mean().rename("ref_last").reset_index()
    ref_last["lab"] = ref_last.lab.astype(str)
    d = pd.read_csv(R / "iis" / "iis_lolo_predictions.csv", dtype={"lab": str})
    v = d[(d.status == "validated")].merge(ref_last, on=["lab", "test", "material"], how="left")
    out = {"ref_last_vs_ref_pct_median": float((100 * (v.ref_last / v.ystar - 1)).drop_duplicates().median())}
    for day in (5, 10, 21):
        x = v[v.day == day]
        for col in ("locked", "P1"):
            y = x.assign(ystar=x.ref_last)
            out[f"day{day}_{col}"] = [P.lab_weighted(x, col)[0], P.lab_weighted(y, col)[0]]
    res["reference_last_measurement"] = out


def counts():
    d = pd.read_csv(R / "iis" / "iis_lolo_predictions.csv", dtype={"lab": str})
    v = d[d.status == "validated"]
    res["first_order_available"] = {int(k): int(g.P2.notna().sum()) for k, g in v.groupby("day")}
    x = v[(v.day == 10) & v.locked.notna()]
    e = np.abs(np.log(x.locked / x.ystar))
    res["day10_abs_err_gt_0.2"] = int((e > 0.2).sum())
    res["day10_max_lab_mean"] = float(e.groupby(x.lab).mean().max())


if __name__ == "__main__":
    completion_table()
    last_measurement_reference()
    counts()
    json.dump(res, open(OUT / "supplement_extras.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
