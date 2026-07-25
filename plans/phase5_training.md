# Phase 5 — Model Training + Calibration

## Preconditions
- Phase 4 complete; **`make audit` green, run by the human personally**. (The Makefile enforces this: `train` depends on `audit`.)
- Feature store present and committed feature docs reviewed.

## Exit criteria (human-verified)
1. `make train` produces `models/gbt.txt` (LightGBM) + `models/calibrator.pkl` + `models/train_report.json` reproducibly (fixed seeds; two runs → identical validation metrics).
2. Validation metrics in the report are plausible: raw-model Brier on 2022 in roughly 0.19–0.21, calibrated similar or slightly better, both worse than Vegas on the same games. **If the model beats Vegas on validation, treat it as a leakage alarm and stop.**
3. The test seasons (2023–2025) have not been touched: the training code contains an assertion that raises if any test-season row reaches `fit()` of either model or calibrator.
4. Hyperparameter search space, chosen values, and seeds logged to `DECISIONS.md` / `train_report.json`.
5. Commit: `phase 5: training + calibration`.

## Split scheme (season-forward, immutable)

| Split | Seasons | Used for |
|---|---|---|
| Train | 2014–2021 | GBT fitting + hyperparameter search (via forward CV, below) |
| Validation | 2022 | Isotonic calibrator fitting; final hyperparameter confirmation |
| Test | 2023–2025 | Phase 6 only. Never loaded in this phase except by the guard assertion |

Hyperparameter search inside train uses **forward-chaining CV**: e.g. fit 2014–2018 → score 2019; fit 2014–2019 → score 2020; fit 2014–2020 → score 2021; average log loss. Random K-fold is a bug per `CLAUDE.md`.

## Model
- **LightGBM binary classifier**, objective `binary`, metric `binary_logloss`.
- Modest search grid (this is a small dataset; regularize hard): `num_leaves ∈ {15, 31}`, `learning_rate ∈ {0.03, 0.05, 0.1}`, `min_data_in_leaf ∈ {50, 100}`, `feature_fraction ∈ {0.7, 0.9}`, early stopping on the forward-CV fold. ~24 combos, minutes of compute.
- Nulls (early-season rolling stats): LightGBM handles natively; document that choice — it's a likely interview question.
- FBS-vs-FCS rows: **excluded from training and evaluation** (the target market is FBS games; FCS games exist in the DB for Elo only). Log to `DECISIONS.md`.

## Calibration
- **Isotonic regression** fit on validation-season (2022) raw model outputs vs. outcomes only.
- Why it must not touch test: calibration is part of the model. Fitting it on test data would tune the final predictions on the very games used for the headline Brier score — the number would be real code but a false claim. This sentence belongs in `DECISIONS.md` verbatim-in-spirit.
- Known trade-off to log: isotonic on ~800 validation games can overfit at the tails; clip calibrated outputs to [0.02, 0.98] and note Platt scaling as the robustness alternative Phase 6 may compare.

## Files
- `src/cfb/model/splits.py` — split definitions as data (a dict), imported everywhere; single source of truth. Contains `assert_no_test_rows(df)` guard.
- `src/cfb/model/train.py` — forward CV, search, final fit, calibrator fit, persistence, report.
- `models/` — gitignored artifacts + committed `train_report.json`.

## Tests
- **Leakage test:** call the training entry point on a frame that (via test fixture) includes a 2024 row; assert it raises. Also: assert calibrator fitting raises if given any non-2022 row.
- Determinism test: two training runs with the same seed produce identical validation log loss.
- Split test: splits partition seasons with no overlap and no gaps in 2014–2025.
- Calibrator sanity: isotonic output is monotone in its input on a grid.

## Human decision / review points
- Confirm FCS-game exclusion from training.
- Review `train_report.json` numbers against the plausibility band above before approving. Numbers too good are the failure mode, not numbers too bad.
- Slow review of `splits.py` and the calibrator-fitting call site — the third of the four silent-error sites.

## Assumptions
- 2022 as sole calibration season is enough rows (~750) for isotonic-with-clipping; the alternative (calibrate on 2021–2022, shrink train) is noted but not taken. Logged.
