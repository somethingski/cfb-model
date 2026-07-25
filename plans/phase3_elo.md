# Phase 3 — Custom Elo System

## Preconditions
- Phases 0–2 complete; DB and benchmark table built.
- Human confirms the FCS policy and parameter starting values below before Claude Code writes code.

## Exit criteria (human-verified)
1. `elo_pregame(game_id, home_elo_pre, away_elo_pre)` table populated for every game 2014–2025, computed in strict chronological order.
2. The **chronology test** passes (see Tests — this is the phase's leakage boundary).
3. Sanity checks printed and eyeballed: perennial powers (Alabama, Georgia, Ohio State) sit in the top ranks across most seasons; Elo-only predictions (logistic of Elo diff + HFA) achieve Brier ≈ 0.19–0.21 on 2014–2021 — plausible for Elo, clearly worse than Vegas. If Elo-only *matches* Vegas, suspect leakage, not brilliance.
4. Parameters and their tuning ranges logged to `DECISIONS.md`.
5. Commit: `phase 3: elo`.

## Full specification

- **Initialization:** all FBS teams start at 1500 in 2014. Teams entering FBS later start at 1300 (below-average newcomer prior).
- **Update rule:** standard Elo, `R' = R + K * MOV_mult * (S − E)`, where `E = 1 / (1 + 10^(−(R_self − R_opp ± HFA)/400))`, S ∈ {0, 0.5, 1}.
- **K-factor:** start at **K = 35**; tunable in [20, 50] by grid search on **train seasons only (2014–2021)**, objective = Elo-only log loss.
- **MOV multiplier (FiveThirtyEight form):** `ln(|margin| + 1) × 2.2 / (0.001 × |elo_diff_winner_perspective| + 2.2)` — dampens blowouts by favorites, prevents autocorrelation.
- **Home-field advantage:** start at **HFA = 65** Elo points added to the home side inside E; 0 for neutral-site games; tunable in [40, 90] on train seasons only.
- **Preseason regression:** at each new season, `R ← (2/3)·R + (1/3)·1500`. Coefficient tunable in {1/4, 1/3, 1/2} on train seasons only. (Regression toward global mean, not conference mean — simpler, defensible; conference realignment makes conference means noisy. Log to `DECISIONS.md`.)
- **FCS opponents:** assign every FCS opponent a fixed rating of **1200** that never updates. FBS teams' ratings do update from these games (a loss to FCS should hurt). Rationale: excluding the games discards real information; tracking FCS ratings adds hundreds of tiny-sample teams. Decision point below.

## The ordering constraint (leakage boundary — state in code comments)
Elo is computed by iterating games sorted by `(start_date, game_id)`. The pre-game ratings written to `elo_pregame` for game *g* are snapshotted **before** applying *g*'s update. Same-day games use pre-day ratings for all of them only if truly simultaneous — since `start_date` includes time, plain datetime ordering suffices; `game_id` breaks exact ties deterministically.

Parameter tuning is a second, subtler leakage boundary: K/HFA/regression tuned on 2014–2021 only, then **frozen**. Tuning them on data the model is later evaluated on would leak through the feature.

## Files
- `src/cfb/elo/engine.py` — pure Elo engine: `expected()`, `mov_multiplier()`, `update()`, `run_season_regression()`. No DB access.
- `src/cfb/elo/pipeline.py` — reads games in order, snapshots pre-game ratings, writes `elo_pregame`, applies updates.
- `src/cfb/elo/tune.py` — grid search on train seasons; writes chosen params to a committed `elo_params.json`.
- `gold/elo_fixture.json` — a tiny synthetic league (4 teams, 6 hand-computed games) with expected ratings after each game, computed by hand.

## Tests
- Unit: `expected()` symmetric (E_home + E_away = 1); MOV multiplier matches hand-computed values; season regression exact arithmetic.
- `test_gold_elo.py`: synthetic-league regression test, exact to 6 decimals.
- **Chronology/leakage test (required):** build Elo through week *w* of a season using the full pipeline; separately build it on a copy of the DB with all games after week *w* **deleted**; assert the pre-game ratings for week-*w* games are byte-identical. If any future game influences a rating, this fails.
- Neutral-site test: HFA term is zero when `neutral_site = 1`.

## Human decision / review points
- **FCS = fixed 1200:** confirm or override (alternative: exclude FCS games from Elo updates entirely; weaker, but simpler to defend). Log either way.
- Review the tuned parameter values for plausibility (K wildly at a grid edge suggests the grid or objective is wrong).
- Slow review of the snapshot-before-update ordering in `pipeline.py` — this is one of the four silent-error sites named in the operating guide.

## Assumptions
- 2020's irregular schedule stays in Elo computation (real games, real information) but is flagged in `RISKS.md` as a distribution-shift risk for any season-level analysis.
