# Early BMP estimates from partial methane production curves

Code, analysis record and results for the article
"Inter-laboratory evaluation of early biochemical methane potential estimates from partial methane
production curves".

The analysis uses the open bottle-level data of two international inter-laboratory BMP studies:

| Dataset | Source | Version | Licence |
|---|---|---|---|
| 2016-2018 study | https://github.com/sashahafner/BMP-kinetics-paper-2022 | commit 55715f3 | MIT |
| 2023-2024 study | https://github.com/AU-BCE-EE/Hafner-2026-IL-BMP | commit a4e3dc1 | CC BY 4.0 |

Copies of the files used are in `01_data_v2/raw/` together with their licences and SHA-256 checksums
(`MANIFEST_sha256.txt`). Please cite the original data publications (Hafner et al., 2020, 2022, 2026)
when reusing these data.

## Repository structure

```
01_data_v2/raw/          source data at the pinned versions, licences, checksums
02_code_v2/bmp_early/    condition construction, estimation methods, evaluation functions
02_code_v2/scripts/      analysis scripts, run in numeric order
03_outputs_v2/           reference values, condition x day estimates, fitted models, summaries
protocol/                analysis protocol, LOCK record (checksums), deviations log
```

## Scripts

| Script | Purpose |
|---|---|
| 01 | Inventory of the 2016-2018 data |
| 02, 05 (R) | Day at which the 1% net 3 d criterion is met, per bottle (biogas::summBg) |
| 03, 03b | Reference values, leave-one-laboratory-out evaluation and sensitivity runs (2016-2018) |
| 04 | Agreement with organiser values, repeatability and between-laboratory dispersion |
| 06 | Evaluation of the fixed methods on the 2023-2024 data, prediction intervals |
| 07 | Summary metrics and first test day meeting the error tolerance |
| 08 | Figures |
| 11, 14 | Supplementary and main tables |
| 12, 13, 15, 16, 17, 19 | Descriptive checks added after the evaluation (see `protocol/deviations.md`) |

The 2023-2024 evaluation (script 06 onwards) was run only after the files listed in
`protocol/LOCK.txt` had been recorded. Later corrections and additional analyses are listed in
`protocol/deviations.md`.

## Reproduction

Python 3.14 with numpy 2.4.6, scipy 1.16.3, pandas 2.3.3, scikit-learn 1.8.0, statsmodels 0.14.6 and
matplotlib 3.10.9 (`requirements.txt`); R 4.6.2 with biogas 1.64.0.

```
pip install -r requirements.txt
Rscript 02_code_v2/scripts/02_ystar_biogas_iis.R
python -I 02_code_v2/scripts/03_iis_development.py
...
```

The Python scripts locate the repository root from their own path. For the R scripts set the
environment variable `BMP_ROOT` to the repository root or run them from that folder.
Scripts 08, 11 and 14 write figures and tables to `04_manuscript_v2/`, which is created if missing.
Scripts 04 and 16 also read subset D of the supplementary data of Hafner et al. (2020, Water 12, 1752),
expected at `05_literature/supplements/Hafner2020_SI/extracted/data/D.csv`.

## Licence

Code: MIT (see `LICENSE`). Data: licences of the original repositories (see `01_data_v2/raw/`).
