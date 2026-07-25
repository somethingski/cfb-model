# CLAUDE.md — Project Conventions

Read this file, `DECISIONS.md`, and `RISKS.md` at the start of every session, before
writing any code.

## What this project claims

This project builds a college football game-outcome model. **The claim is calibration
approaching the de-vigged Vegas closing line.** Never write code, comments,
documentation, commit messages, or UI text that states or implies the model beats the
market, has an "edge," "outperforms" the line, or is profitable against the spread.

If a result appears to beat Vegas, the prior is **bug**, not breakthrough. Stop and
investigate before writing it down.

## The anti-leakage invariant

**Every feature for a game uses only information available strictly before that game's
kickoff.** This is the core technical claim of the project and it is load-bearing.

- All train/validation/test splits are **season-forward**. Random splits are a bug.
- `train_test_split`, `KFold`, and `shuffle=True` on game rows are bugs. Hyperparameter
  search uses forward-chaining cross-validation.
- **Vegas lines are never model features.** A line-derived column in the feature list is
  a stop-everything error. Lines exist only as the Phase 2 benchmark.
- Season-level aggregates (final records, end-of-year rankings, season totals) are
  leakage when applied to a mid-season game. Use as-of-kickoff values only.
- The Phase 4 leakage audit is a hard gate. If it fails, **the fix is in the features,
  never in the audit.** Do not loosen a tolerance, shrink a sample, or skip a check to
  get green.

## Governance

- **Log every non-obvious choice to `DECISIONS.md`** as it happens, not retroactively.
  Format: `YYYY-MM-DD | decision | rationale`. Append-only.
- **Never fabricate, interpolate, or impute missing source data silently.** Missing
  lines, missing stats, anomalous seasons: exclude explicitly and document the gap in
  `RISKS.md`. Imputation "to fix a gap" that nobody asked for must be reverted.
- Record data-quality gaps in `RISKS.md` with likelihood, impact, and mitigation.

## Style

- Python 3.11+ with type hints on public functions; Google-style docstrings.
- Anything touching time ordering is a **small pure function**, tested in isolation.
  Chronology bugs are silent; testability is the defense.
- `pytest` for all logic. A test that cannot fail is not a test — leakage and
  chronology tests must include a poisoned-input case proving the check fires.
- `src/cfb/config.py` is the single source of truth for paths, season range, and
  constants. No other module hardcodes a path or a season range.
- Handle errors explicitly. Never swallow exceptions.
- Never hardcode secrets. `CFBD_API_KEY` comes from the environment via `.env`.

## Session ritual

One phase per session, fresh context each time. The phase plans live in `plans/`.
At session start: read `CLAUDE.md`, `DECISIONS.md`, `RISKS.md`, then restate the
phase's preconditions, exit criteria, and files before writing anything.
