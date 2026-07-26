# Model card — cfb-model v0.1.0

## Intended use

**Educational and portfolio use only.** This model exists to demonstrate a leakage-resistant modelling pipeline on a domain where a hard external benchmark exists. It is **not a betting tool**, it does not beat the market, and it has never been evaluated for that purpose — no closing-line value, no bet sizing, no transaction costs.

The claim, stated in full: **calibration approaching the de-vigged Vegas closing line**. On held-out seasons the model is short of that line by 0.0149 Brier.

## Data

- Source: [CollegeFootballData.com](https://collegefootballdata.com/), seasons 2014–2025.
- One row per completed game. FBS-vs-FCS games are excluded from modelling and scoring; they still feed Elo and the rolling windows.
- Nothing is imputed. Missing lines, missing box scores and the one cancelled game are excluded explicitly and recorded in `RISKS.md`.

## Features

26 inputs, all as-of-kickoff: `home_elo_pre`, `away_elo_pre`, `elo_diff`, `week`, `season_type`, `neutral_site`, `conference_game`, `rest_days_home`, `rest_days_away`, `rest_diff`, `off_ppg_roll_home`, `def_ppg_roll_home`, `off_ypp_roll_home`, `def_ypp_roll_home`, `pace_roll_home`, `off_ppg_roll_away`, `def_ppg_roll_away`, `off_ypp_roll_away`, `def_ypp_roll_away`, `pace_roll_away`, `prev_season_win_pct_home`, `prev_season_win_pct_away`, `prior_games_home`, `prior_games_away`, `fcs_games_in_window_home`, `fcs_games_in_window_away`.

**No market information is a feature.** The betting line is the benchmark, never an input, and the feature builder is scanned mechanically for market terms as part of the Phase 4 audit. Every column is documented in `FEATURES.md`, generated from the code.

## Split scheme

Season-forward and immutable: **train 2014–2021, validation 2022, test 2023–2025**. Hyperparameters were chosen by forward-chaining cross-validation; the isotonic calibrator was fitted on the validation season alone. Random splits are a bug in this project, not a choice.

Test seasons were untouched until this evaluation: the guards run inside `fit()` rather than beside it, and the feature store passes a leakage audit that recomputes a sample of rows from a database truncated at each game's kickoff.

## Evaluation

**2398 held-out games**: test seasons, FBS vs FBS, and a closing line present — the intersection. 0 games were dropped for having no line from any provider. Every system below is scored on that one frame, in one order, so the comparison is between systems and not between game sets.

| System | Brier | Log loss |
|---|---|---|
| model (calibrated) | 0.1913 | 0.5631 |
| de-vigged closing line | 0.1763 | 0.5223 |
| naive home baseline | 0.2427 | 0.6784 |
| Elo only | 0.1902 | 0.5589 |

Per season:

| Season | n | Model Brier | Line Brier | Gap |
|---|---|---|---|---|
| 2023 | 792 | 0.1854 | 0.1715 | +0.0139 |
| 2024 | 798 | 0.1999 | 0.1819 | +0.0180 |
| 2025 | 808 | 0.1885 | 0.1756 | +0.0129 |

Full table and reliability curves: [`results_table.md`](results_table.md).

## Limitations

- Not a betting tool. The model scores worse than the de-vigged closing line, and the line already has vig on top of it. Nothing here has been evaluated against closing-line value, bet sizing, or transaction costs, and no such claim is made.
- The benchmark is spread-derived, not a traded price. `p_home_devig` converts a closing spread through a normal margin model with a fixed sigma fitted on 2014-2021 (RISKS #15, #17). It is a good yardstick, not the market's own probability.
- One model, one split, one run. There is no repeated-seed study and no confidence interval on any gap reported here. Differences of a thousandth of a Brier point between systems are not resolvable at this sample size.
- Hyperparameters are barely tuned in any meaningful sense. The 24-point grid spans 0.0053 in mean forward-CV log loss best to worst (RISKS #23); the chosen point is defensible, not discovered.
- The isotonic calibrator did not generalise. Fitted on a single season of 776 games (RISKS #4, #25), it improved 2022 in-sample and made every held-out season worse than leaving the model raw; the measured cost and the mechanism are in `results_table.md`. The headline is still the calibrated model, because re-picking the raw one on the strength of its test-season score would be selecting a model on the test set.
- A one-parameter logistic of the Elo difference scores better than the shipped calibrated model on these seasons. The 26-feature booster earns its keep only before calibration, and only narrowly, which says the rolling box-score features carry little signal that Elo does not already carry.
- Elo is cold-started in 2014 and every FCS opponent shares one fixed 1200 rating (RISKS #18, #19, #22), so the rating features are noisiest exactly where the training data begins and are inflated for teams fresh off an FCS game.
- Plays are reconstructed as rushing attempts plus pass attempts, not observed (RISKS #21), so every yards-per-play and pace feature rests on a derived denominator that will not match every published play count.
- Team strength is represented by Elo, rest, and rolling box-score rates only. There are no injuries, no weather, no personnel, no travel distance, and no market information of any kind — that last one by rule, since the line is the benchmark.

## The statement that matters

**This model does not beat the market.** It lands between a naive home-field baseline and the de-vigged closing line, closer to the line, and short of it by 0.0149 Brier over 2398 games. That is the intended result. `make evaluate` fails with a non-zero exit code if the model ever scores better than the line, because on public box-score data the explanation for that would be a leak.
