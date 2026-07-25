# Phase 7 — Writeup and Reproducibility

## Preconditions
- Phase 6 complete; `results/` artifacts approved by the human.
- All prior phases' tests green.

## Exit criteria (human-verified)
1. **`make reproduce` runs clean from empty state**: fresh clone + `.env` → ingest (cache-assisted) → benchmark → elo → features → audit → train → evaluate → results identical to committed `results/`. Human performs this in a fresh directory personally.
2. README reads as a 3–5 page report and contains **real numbers** from `results/`, no placeholders.
3. Zero over-claiming: human (or a grep script) checks README, model card, and docstrings for "beat", "edge", "profitable", "outperform the market" — all absent or explicitly negated.
4. Commit + tag: `v1.0`.

## README structure (3–5 pages, report-style prose — no bullet spam)

1. **Abstract** (5 sentences): what was built, the honest headline ("a calibrated model that closes X% of the Brier-score gap between a naive home baseline and the de-vigged Vegas closing line on 2023–2025 FBS games"), and why approaching-not-beating is the right claim.
2. **Problem framing**: why the closing line is the efficient-market benchmark; why calibration (not accuracy) is the target metric.
3. **Data**: CFBD, 2014–2025, schema sketch, line coverage honesty.
4. **Methods**: Elo spec (parameters, values, why), features, the anti-leakage framework — give the recompute-under-truncation audit its own subsection; it is the differentiator.
5. **Evaluation**: the four-system table, reliability figure, per-season stability.
6. **Limitations** (honest, specific): cannot beat the market and doesn't claim to; the "closing" line may predate true close (Phase 2 caveat); FCS opponents handled by fixed-rating heuristic; mid-season coaching changes, injuries, and weather unmodeled; isotonic tails on ~750 calibration games; 2020 distribution shift; results are three test seasons, not a stationarity proof.
7. **Reproduction**: `.env` setup, `make reproduce`, expected runtime.

## `make reproduce`
Chains: `ingest → vegas → elo → features → audit → train → evaluate → test`. Properties:
- Fails loudly at the first broken stage; the audit dependency means training is impossible without a passing gate even in reproduction.
- Deterministic given the cache (network variance removed); document that a truly-from-API run may differ only if CFBD retroactively edits data, and that the cache directory can be zipped for archival.
- Prints a final summary comparing produced metrics to committed `results/` values.

## Files
- `README.md` — the report.
- `Makefile` — all stubs replaced; `reproduce` target complete.
- `scripts/check_claims.sh` — the over-claiming grep (run in CI/test so it can't regress).

## Tests
- `test_no_overclaiming.py`: scans README/model card/UI strings for the banned-claims list; fails on hits. (Yes, a test for honesty — it makes the constraint durable across future edits.)
- `test_makefile_targets.py`: every documented target exists.

## Human decision / review points
- Read the README aloud once. Every sentence you couldn't defend to an interviewer gets rewritten or deleted. Cross-check each number against `results/` exactly — no rounding drift between the results table and prose.
- Decide the resume line now, while the numbers are fresh, and log it in `DECISIONS.md` so future-you doesn't reconstruct (and inflate) it from memory.

## Assumptions
- The README doubles as the "operating explainer" content-wise; Phase 8's app links to it rather than duplicating it.
