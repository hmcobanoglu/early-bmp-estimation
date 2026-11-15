"""IIS sensitivity analyses (leave-one-laboratory-out, validated cases):
modified Gompertz base and composition-informed asymptote cap (1.10 x TBMP).

Run after 03_iis_development.py:  python -I 02_code_v2/scripts/03b_iis_sensitivity.py
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bmp_early import cases as C          # noqa: E402
from bmp_early import estimators as E     # noqa: E402
from bmp_early import pipeline as P       # noqa: E402

OUT = C.ROOT / "03_outputs_v2" / "iis"
TBMP = {"SA": 440.0, "SB": 448.0, "SC": 606.0, "SD": 481.0, "CEL": 414.0}  # Hafner et al. 2020, Table 1


def run(df, curves, label):
    rows = []
    for day in P.DAYS:
        dd = df[df.day == day]
        for lab in sorted(dd.lab.unique()):
            m = P.fit(dd[dd.lab != lab], curves, day)
            rows.append(P.predict(m, dd[dd.lab == lab]))
    pr = pd.concat(rows, ignore_index=True).drop(columns=["tt", "yy"])
    pr["variant"] = label
    return pr


def main():
    df, curves = pickle.load(open(OUT / "iis_cases.pkl", "rb"))
    df["tbmp"] = df.material.map(TBMP)
    out = []
    E.SETTINGS.update(form="gompertz", tbmp_cap=None)
    out.append(run(df, curves, "gompertz"))
    E.SETTINGS.update(form="first_order", tbmp_cap=1.10)
    out.append(run(df, curves, "first_order_tbmp_cap"))
    E.SETTINGS.update(form="first_order", tbmp_cap=None)
    pd.concat(out).to_csv(OUT / "iis_lolo_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
