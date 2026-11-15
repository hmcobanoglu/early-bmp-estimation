# Analysis protocol — cross-laboratory evaluation of early BMP estimates

Version 1.0 (to be locked by commit + SHA-256 before any computation on IL-BMP gas, SMP or BMP files).
Author: H. Muratçobanoğlu. Code: `02_code_v2/`. Data: `01_data_v2/raw/MANIFEST_sha256.txt`.

## 0. What was seen before locking
- Full text and supplements of Hafner et al. 2020, 2022, 2026; Koch et al. 2026; Holliger et al. 2016, 2021.
- IIS-BMP (development archive): all files; exploratory analyses in `03_outputs_v2/inventory/`.
- IL-BMP (evaluation archive): README, `setup.csv`, and the **time stamps only** of `biogas.csv` (to check blank/substrate measurement synchrony: 91% of substrate bottles synchronous with blanks up to day 10). No gas volumes, SMP, BMP_calc, BMP_rep or SMP_eval values were analysed.

## 1. Units and archives
- Case = laboratory × test × material; prediction for the triplicate mean.
- Development: IIS-BMP 2016–2018 (Hafner et al. 2020, 2022), 37 laboratories, 124 tests, 399 cases.
- Evaluation: IL-BMP 2023–2024 (Hafner et al. 2026).
  - **Primary evaluation set:** Test 1, evaluation laboratories (IDs E\*), non-cellulose materials, cases whose substrate and blank bottles were measured at the same times up to the prediction day.
  - Sensitivity sets: Test 2; Tests 1+2 (laboratory-weighted); including reference laboratories R01–R06; including asynchronous cases; cellulose reported separately.

## 2. Reference BMP (Y*)
- IIS: `biogas::summBg(when = "1p3d")` (biogas 1.64.0, R 4.6.2) on normalised cumulative CH4 (`cvCH4`), blank correction by inoculum mass, normalised by substrate VS. Y* = mean of bottle values; t* = time at which the criterion was met.
- IL: t* from `summBg(when = "1p3d")` on cumulative CH4 of `biogas.csv` (the 1% ratio rule is invariant to a constant normalisation factor); Y* = mean of the published net SMP (`SMP.csv`) of the bottles at t*. Agreement with `BMP_calc` reported (after locking).
- Reconciliation (IIS): case-level comparison with Hafner et al. 2020 Supplementary subset D (`BMP_1p3dnet`, `time_d_1p3dnet`): median absolute difference, 90th/95th percentiles, unmatched cases listed.
- Status: **validated** (Y* observed and test cellulose Y* within 340–395 NmL g⁻¹ VS with bottle RSD ≤ 6%), **outside cellulose QC** (Y* observed, test fails or has no cellulose), **incomplete** (criterion not met; no error computed). Relaxed QC (330–415, RSD ≤ 15%) = sensitivity. Primary = validated.

## 3. Information available at day t (t = 5, 7, 10, 14, 21 d; primary t = 10)
- IIS inputs: net SMP recomputed from `cvCH4` of substrate and blank bottles **with blank values interpolated only between blank observations ≤ t**; last observation ≤ t used; no interpolation across t.
- IL inputs: published net SMP truncated at t, primary set restricted to synchronous cases (blank correction at each point uses the blank measured at the same time).
- Allowed covariates: ISR, last observation age, criterion-met-within-0–t indicator (1% net 3 d visible from 0–t data), same-test cellulose net SMP at t (scaled by training-fold mean cellulose Y*), material class (a priori: cellulose/whey; feeds/seeds; wheat straw).
- Forbidden: any data after t, full-curve kinetics, BMP_calc/BMP_rep/SMP_eval, termination day, QC flags (used only to define the evaluated set).
- Test: truncating the raw files at t and rebuilding features gives identical values.

## 4. Estimators (code names in brackets)
1. Last observed net methane yield [P0]: ŷ = Y_t. Requires Y_t > 0.
2. Class-completion ratio [P1]: ŷ = Y_t / f̄_c(t_last), f̄_c = mean Y_t/Y* of training validated cases of class c at t_last (linear interpolation between integer days). Requires Y_t > 0 and f̄_c > 0.05.
3. Lag-adjusted first-order fit [P2, a priori kinetic base]: lag end τ = end of the interval preceding the maximum interval rate computed from 0–t data (Hafner et al. 2022); y(t) = y(τ) + A[1 − exp(−k(t − τ))], fitted to points t ≥ τ; ŷ = y(τ) + A. Bounds k ∈ [0.01, 1.5] d⁻¹, A ∈ [0, κ_c·Y_t] with κ_c = 1.1 × max training (Y*/Y_t) in class c at t (portable, composition-free). Fails if fewer than 4 observations in (0, t], fewer than 3 points at or after τ, maximum interval rate at the last observation, non-convergence, or ŷ ≤ 0. Staleness rule (t − t_last > 4 d) applies only when the criterion is not yet met within 0–t.
4. Residual-corrected kinetic model [P3, main]: r = ln(Y*/ŷ_P2) predicted by Ridge regression (α chosen by grouped inner 5-fold CV over laboratories) on: ln(Y_t/ŷ_P2), ratio of last-interval rate to maximum rate, k̂, t − τ, last observation age, ISR, criterion-met indicator, cellulose ratio (with missing indicator; zero/negative → missing), class indicators. ŷ = ŷ_P2·exp(r̂). Fit on cases where P2 succeeded.
5. Cellulose-adjusted ratio [C1, secondary]: ŷ = P1·(f_cel,i(t)/f̄_cel(t))^γ; γ fitted on non-cellulose training cases; falls back to P1 when the control is missing, ≤ 0, or measured later than t.
- Fallback chain (identical in both archives): P3 → P2 → P1 → P0 → no estimate. The step used is recorded; "no estimate" counts are reported.
- Sensitivity: modified Gompertz base (Zwietering et al. 1990) with its own bounds; composition-informed bound A ≤ 1.10·TBMP − y(τ).

## 5. Validation and estimands
- IIS development: leave-one-laboratory-out (LOLO); all scaling, f̄_c, κ_c, α, γ re-estimated inside each outer fold.
- **Primary estimate:** laboratory-weighted mean |ln(ŷ/Y*)| at day 10 for the locked procedure (P3 with fallback) in the primary IL evaluation set (first average within laboratory, then across laboratories), with laboratory-cluster bootstrap 95% CI (2000 resamples) and G, n.
- Also reported: the same quantity for IIS LOLO (development); signed log error; share of validated cases receiving an estimate; share within ±10% (|ŷ/Y* − 1| ≤ 0.10) among all validated cases and among those with an estimate; per day, per class; fixed cohort (all validated cases) and ongoing cases (t* > t).
- Comparisons (secondary, same cases): P3 vs P2, P3 vs P0, C1 vs P1: laboratory-level differences, mean difference with bootstrap CI, number of laboratories improved.

## 6. Prediction intervals (secondary)
- Jackknife+/CV+ (Barber et al. 2021) on LOLO residuals of the all-IIS P3; score = |ln(ŷ/Y*)|, each case weighted 1/n_g (pooled-CDF weighting, Dunn et al. 2023); computed per day and class; nominal 90%; interval unavailable if fewer than 9 laboratories contribute. Evaluated on IL: empirical coverage, mean width, miss counts per laboratory.

## 7. Dispersion of reference BMP
- s_r: pooled within-case SD of ln(bottle BMP) (validated cases).
- s_L: between-laboratory SD of ln Y* from a mixed model ln Y* ~ material + (1|lab) (REML; lab×material term added if estimable).
- Reported on the natural-log scale next to the mean absolute log error; no subtraction or equivalence claim.

## 8. Decision summary (descriptive)
- First evaluated day at which ≥ 90% of validated cases of a class (laboratory-weighted) received an estimate within ±10%; stability by laboratory bootstrap (share of resamples giving each day; "no eligible day" frequency); tolerances 5% and 15% as scenarios; median t* of the same cases given as a descriptive timing difference.

## 9. Locking
- Before IL computation: commit code, this protocol, IIS outputs and the fitted model objects; record SHA-256 in `protocol/LOCK.txt`.
- Deviations after locking are listed in `protocol/deviations.md`.

## 10. Refinements made before locking (v1.1)
- Case definition extended to laboratory × test (IL: submitted data file) × material × nominal ISR (nearest of 1, 2, 4 on the log scale). Several IIS tests ran the same material at two or three ISRs (Hafner et al. 2020); pooling them as one triplicate was an error found during development. The 1% net 3 d evaluation in `summBg` is run per condition.
- IIS truncation implemented by deleting all gas records after day t (substrate and blank bottles) and recomputing net SMP; this replaces interpolation with a stored blank time.
- When a test has more than one cellulose condition, the ISR 2 condition is used for quality control and for the cellulose covariate.
- Development results (IIS, validated cases, day 10, laboratory-weighted mean absolute log error): locked procedure 0.083 (G = 26, n = 203); class-completion ratio 0.093; lag-adjusted first-order fit 0.148; residual-corrected model on cases with a successful fit 0.067. Gompertz base 0.080 and TBMP cap 0.083 (sensitivity, not adopted). These values were seen before locking.
