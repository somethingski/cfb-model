# Phase 6 — Evaluation

## Preconditions
- Phase 5 complete; model + calibrator persisted; test seasons untouched.
- This phase is the **only** code allowed to read test-season labels.

## Exit criteria (human-verified)
1. `make evaluate` produces `results/results_table.md`, `results/reliability.png` (overall + per-season), and `results/model_card.md`.
2. All four systems scored on the **identical game set** (test seasons, FBS-vs-FBS, games having a Vegas benchmark — the intersection; report how many games were dropped for missing lines).
3. The headline in every artifact is the **model-vs-Vegas gap**, stated as a gap. Human greps outputs for over-claiming language ("beats", "outperforms the market", "edge") and finds none.
4. Numbers pass an exact-arithmetic spot check: human recomputes Brier for one small slice (e.g. one test week) by hand/spreadsheet and matches the pipeline.
5. Commit: `phase 6: evaluation`.

## Systems compared (all on the same games)

| System | Definition |
|---|---|
| (a) Model | calibrated GBT from Phase 5 |
| (b) Vegas | de-vigged closing-line `p_home` from Phase 2 |
| (c) Naive home baseline | constant `p_home` = home-team win rate on **train seasons** (~0.56; computed from train only — even baselines respect the split) |
| (d) Elo-only | logistic of (Elo diff + HFA) with the logistic scale fit on train seasons |

## Metrics and outputs
- **Brier score** and **log loss** per system: overall test period and per season (2023, 2024, 2025) — per-season columns expose calibration drift (a named risk in `RISKS.md`).
- **Reliability curves**: 10 equal-count bins, predicted vs. empirical frequency, with counts per bin; model and Vegas overlaid on one plot. Expected honest picture: Vegas hugs the diagonal tightest; model close behind; Elo-only visibly looser.
- **Skill framing for the README:** report the gap both raw (e.g. "model Brier 0.201 vs. Vegas 0.192") and as % of the naive-to-Vegas distance closed: `(Brier_naive − Brier_model) / (Brier_naive − Brier_vegas)`. This is the single most interview-effective honest number — "the model closes X% of the gap between a naive baseline and the closing line."
- **Model card** (`results/model_card.md`): intended use (educational/portfolio), data range, feature list, split scheme, metrics table, limitations (imported from Phase 7 list), and the explicit statement that the model does not beat the market and is not a betting tool.
- Optional robustness row (only if time permits; not a gate): metrics restricted to moneyline-sourced benchmark games vs. spread-fallback games, testing sensitivity to the Phase 2 fallback.

## Files
- `src/cfb/eval/evaluate.py` — loads model+calibrator, predicts test seasons, scores all four systems, writes artifacts.
- `src/cfb/eval/plots.py` — reliability curves.
- `results/` — committed (small text + one PNG); these are the numbers the README cites, so they must be in-repo and reproducible.

## Tests
- Unit: Brier and log-loss implementations against hand-computed values on a 4-game toy set (or verify sklearn's versions against the same by test).
- Game-set test: the four systems' evaluation frames are identical (same game_ids, same order).
- **Leakage tripwire:** assert the model's test Brier is ≥ Vegas's minus a small tolerance (0.002). If the model "beats" Vegas, the run fails with an instruction to investigate leakage rather than celebrate. (Genuinely beating the closing line on 2,000+ games with public data is extraordinary-claims territory; the default explanation is a bug.)
- Assert no train/val season appears in the evaluation frame.

## Human decision / review points
- Approve the final results table before it flows into Phase 7 — these numbers go on a resume line; verify them the way you verified resume claims.
- Review the reliability plot: a model curve that looks *better* than Vegas's is, again, a red flag first.

## Assumptions
- 2025 season data is complete in CFBD by now (it should be); if late-season games are missing, report actual coverage rather than silently shrinking the test set.
