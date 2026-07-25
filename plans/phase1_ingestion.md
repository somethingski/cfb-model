# Phase 1 — Data Ingestion (CFBD → SQLite)

## Preconditions
- Phase 0 complete and committed; `pytest` green.
- `.env` present locally with a working `CFBD_API_KEY` (human verifies with one manual curl or the client's `--check` mode before backfill).
- Claude Code has read `CLAUDE.md`, `DECISIONS.md`, `RISKS.md`.

## Exit criteria (human-verified)
1. `make ingest` backfills 2014–2025 into `data/cfb.sqlite` idempotently — running it twice produces identical row counts and hits the API zero times on the second run (cache serves everything).
2. Data-quality assertions pass and print a per-season summary table (games per season, % with lines, % with team stats).
3. Gold fixture tests pass against hand-checked values.
4. No raw API key appears anywhere in code, cache filenames, or logs.
5. Commit: `phase 1: ingestion`.

## CFBD endpoints to call

| Endpoint | Params | Purpose |
|---|---|---|
| `/games` | `year`, `division=fbs` | Game spine: ids, teams, kickoff datetime, scores, neutral flag, week, conference flags |
| `/games/teams` | `year`, `week` (or per-game) | Per-team box stats per game |
| `/lines` | `year` | Spreads, over/unders, moneylines per game per book |
| `/stats/season/advanced` | `year` | Season advanced stats (context; per-game rolling stats are built in Phase 4 from game-level data) |
| `/teams/fbs` | `year` | Team master list, conference membership per season |
| `/calendar` | `year` | Week boundaries (for rest-day and week features) |

**Note:** `/games` includes both teams' FBS status; keep games where the *home* team is FBS but the opponent is FCS — Phase 3 needs an explicit FCS policy, so the games must exist in the DB.

## SQLite schema

```sql
teams(team_id INTEGER PK, school TEXT UNIQUE, ...);
team_seasons(team_id, season, conference, division, PK(team_id, season));
games(game_id INTEGER PK, season, week, season_type,           -- 'regular'/'postseason'
      start_date TEXT,                                          -- ISO8601 UTC; the leakage clock
      neutral_site INTEGER, conference_game INTEGER,
      home_team_id, away_team_id, home_points, away_points,
      completed INTEGER,
      FOREIGN KEY refs to teams);
game_team_stats(game_id, team_id, is_home INTEGER,
      stat_name TEXT, stat_value REAL,                          -- long/tidy format
      PK(game_id, team_id, stat_name));
lines(game_id, provider TEXT, spread REAL, over_under REAL,
      home_moneyline INTEGER, away_moneyline INTEGER,
      PK(game_id, provider));
```

Indexes: `games(season, week)`, `games(start_date)`, `game_team_stats(team_id, game_id)`, `lines(game_id)`.

`start_date` is the **canonical leakage clock** for the whole project — every later phase orders by it. Assert non-null for all completed games.

## Rate limiting and caching
- Client wraps `requests` with: API key from env, ~1 req/sec throttle, exponential backoff on 429/5xx (max 5 retries), and a **disk cache**: `cache/{endpoint}/{param-hash}.json` written on success, read before any network call. Cache is the idempotency mechanism.
- Backfill script iterates seasons → endpoints, upserting (`INSERT OR REPLACE`) into SQLite. Safe to interrupt and resume.

## Data-quality assertions (fail loudly, never auto-fix)
- Games per season within sane bounds (roughly 700–900 FBS-involved games; 2020 lower — assert a documented lower bound for 2020 only, log to `RISKS.md`).
- No duplicate `game_id`; no completed game with null scores; no game with null `start_date`.
- Line coverage per season reported (expect high but not 100%; **do not** fill gaps — Phase 2 handles missing lines by exclusion from the benchmark set, documented).

## Gold fixture (`gold/games_fixture.json`)
Hand-check and commit 6–10 games spanning eras and edge cases, e.g. a 2014 game, a 2020 game, a neutral-site game, an FBS-vs-FCS game, a postseason game. For each: game_id, teams, final score, spread and moneylines from at least one book. **The human verifies these against the CFBD website by eye before committing.** Test asserts DB rows match the fixture exactly.

## Tests
- `test_client_cache.py`: with a mocked network layer, second identical request never calls the network.
- `test_schema_quality.py`: runs the data-quality assertions against the built DB (marked as integration; skipped if DB absent).
- `test_gold_games.py`: fixture regression test.
- **Leakage-relevant:** `test_start_date_present.py` — every completed game has a parseable UTC kickoff; the leakage clock cannot have holes.

## Human decision / review points
- Confirm cache-on-disk approach (raw JSON, committed to `.gitignore`) — cheap insurance against API changes.
- Eyeball the per-season summary table before approving exit. Weird counts here poison everything downstream.

## Assumptions
- CFBD free tier remains sufficient for a full backfill with caching (it historically is, throttled). If quota blocks the backfill, stop and report — do not scrape alternate sources.
- Tidy (long) format for team stats: stat names vary across eras; long format absorbs that without schema churn. Log to `DECISIONS.md`.
