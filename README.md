# cfb-model

A college football game-outcome model, built in phases with an explicit anti-leakage
framework.

**Status: Phase 1 (ingestion).** CFBD data lands in SQLite; no features and no model yet.

The project's claim is **calibration approaching the de-vigged Vegas closing line** —
not beating it. Results, baselines, and limitations are written up in Phase 7; this
file is a stub until then.

## Layout

| Path | Purpose |
|---|---|
| `src/cfb/` | Package: `config`, then `ingest`, `vegas`, `elo`, `features`, `model`, `eval` by phase |
| `plans/` | The phase-by-phase build plans |
| `gold/` | Committed hand-verified fixtures |
| `tests/` | pytest suite, including the leakage audit from Phase 4 |
| `data/`, `cache/` | Gitignored: SQLite DB, feature parquet, raw API responses |

## Development

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Copy `.env.example` to `.env` and add a
[CollegeFootballData.com](https://collegefootballdata.com/) API key before Phase 1.

## Ingestion

```bash
make ingest-check   # one live request, confirms the API key works
make ingest         # backfill 2014-2025 into data/cfb.sqlite
```

Every API response is cached under `cache/`, so the backfill is idempotent and resumable:
a second run reproduces the first with zero network calls, and the run reports its network
and cache counts so that claim is checkable. `games.start_date` is the canonical kickoff
clock every later phase orders by.

Coverage gaps (missing lines, missing box scores) are reported per season and left alone —
later phases exclude them explicitly and `RISKS.md` records them. Nothing is interpolated.

See `CLAUDE.md` for project conventions, `DECISIONS.md` for the choice log, and
`RISKS.md` for known gaps.
