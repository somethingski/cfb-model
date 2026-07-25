# cfb-model

A college football game-outcome model, built in phases with an explicit anti-leakage
framework.

**Status: Phase 0 (scaffold).** No data, no features, no model yet.

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

See `CLAUDE.md` for project conventions, `DECISIONS.md` for the choice log, and
`RISKS.md` for known gaps.
