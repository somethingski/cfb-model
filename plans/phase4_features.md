# Phase 4 — Feature Engineering + Anti-Leakage Framework  ⛔ HARD GATE

This phase **is** the resume story. The anti-leakage framework matters more than the model. The project does not proceed to Phase 5 until the leakage audit passes and the human has reviewed it line by line. The audit must never be weakened to make it pass.

## Preconditions
- Phases 0–3 complete; `elo_pregame` populated; chronology test green.
- Human has read this plan's audit design and approved it **before** feature code is written (audit-first development: the gate exists before the thing it gates).

## Exit criteria (human-verified)
1. Parquet feature store written: `data/features/features.parquet`, one row per FBS-vs-FBS completed game 2014–2025, plus FBS-home-vs-FCS games flagged by an `fcs_opponent` column.
2. **Leakage audit passes on a fresh run** (see below), and the human has run it personally.
3. Feature documentation table generated: every feature's name, definition, and the *latest timestamp of information it depends on*, asserted < kickoff.
4. Null-handling policy visible per feature (early-season rows have legitimate nulls for rolling stats — nulls are kept and documented, never back-filled from future games).
5. Commit: `phase 4: features + leakage audit`.

## Feature list (v1 — deliberately boring and defensible)

| Feature | Definition | Leakage note |
|---|---|---|
| `elo_diff` | `home_elo_pre − away_elo_pre` | Phase 3 guarantees pre-game |
| `home_elo_pre`, `away_elo_pre` | raw levels | ditto |
| `neutral_site`, `conference_game`, `week`, `season_type` | game metadata | known at scheduling time |
| `rest_days_home`, `rest_days_away`, `rest_diff` | days since each team's previous game | prior games only |
| Rolling per-team, prior games this season, shifted by one game: `off_ppg_roll`, `def_ppg_roll`, `off_ypp_roll`, `def_ypp_roll`, `pace_roll` (plays/game) | mean over that team's **previous** games in the season (min 1 prior game, else null) | the shift-by-one is the whole trick; test it |
| `prev_season_win_pct_home/away` | last season's record | strictly historical |
| `fcs_opponent` | flag | metadata |

Explicitly **excluded** from v1 (log to `DECISIONS.md`): recruiting/returning production (coverage inconsistency — revisit only if Phase 6 motivates it), weather (unavailable pre-kickoff reliably in CFBD), and anything derived from the game's own box score.

**Vegas lines are not features.** The line is the benchmark, not an input. Feeding the line to the model would make "approaching the line" circular. State this in `CLAUDE.md` if not already there.

## Parquet feature-store layout
Single file, one row per game: `game_id, season, week, start_date, home_team_id, away_team_id, [features...], label_home_win`. Partitioning by season is unnecessary at this scale (~9k rows); note in `DECISIONS.md`. The label column lives here for convenience but is used only by Phase 5's training code — never by feature computation.

## The leakage audit script (`src/cfb/features/audit.py`) — the gate

**Design: recompute-under-truncation.** For a random sample of N=200 games (seeded, reproducible):
1. Read game *g*'s kickoff time *t*.
2. Construct a truncated view of the database containing **only games with `start_date < t`** (SQL views or temp tables; no copying the full DB per game — implement efficiently).
3. Recompute every feature for *g* from the truncated view (reusing the same feature functions — the audit must call production code, not a reimplementation).
4. Assert equality with the stored feature row, exact for ints/flags, `1e−9` for floats.

Any mismatch = leakage or nondeterminism; either is a hard failure with a diff printed. The audit also:
- Asserts no feature column is perfectly correlated with the label within any season (a canary for label leakage).
- Runs the Phase 3 chronology test as a sub-check (Elo feeds features).

`make audit` runs it; Phase 5's Makefile target **depends on** `make audit` so the gate is mechanical, not honor-system.

## Files
- `src/cfb/features/build.py` — feature computation; every rolling function takes an explicit `as_of` cutoff internally (design features around the cutoff, don't bolt it on).
- `src/cfb/features/audit.py` — as above.
- `src/cfb/features/docs.py` — generates the feature documentation table into `FEATURES.md`.
- `gold/features_fixture.json` — 3 games with hand-computed rolling stats (pick week-4-ish games so rolling windows are small enough to compute by hand).

## Tests
- `test_shift_by_one.py`: construct a synthetic 3-game season; assert game 3's rolling stats equal the mean of games 1–2 and game 1's are null. **This test fails if the shift is dropped** — the canonical leakage bug.
- `test_gold_features.py`: fixture regression.
- `test_audit_catches_leakage.py`: deliberately build a poisoned feature (rolling stat without the shift) in a test fixture and assert the audit **fails** on it. An audit that has never been seen to fail proves nothing.
- `test_rest_days.py`: hand-checked rest-day arithmetic including season openers (null or capped — pick, document).

## Human decision / review points
- Approve the audit design before feature code exists.
- After the audit passes: **you run it yourself**, then read `FEATURES.md` end to end and challenge any feature whose pre-kickoff availability you can't explain aloud. That explanation is literally the interview.
- Decide season-opener rest-day policy (recommend: cap at 30 days rather than null; log it).

## Assumptions
- ~9k rows and ~15 features is small; the win here is correctness discipline, not feature volume. Resist scope creep.
