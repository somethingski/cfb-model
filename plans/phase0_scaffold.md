# Phase 0 — Repo Scaffold and Project Governance

## Preconditions
- Empty git repository initialized (`git init` done by human, or do it as step 1).
- Python 3.11+ available (verify with `python --version` and record the exact version in `DECISIONS.md`).
- Human has a CollegeFootballData.com API key (do **not** ask for its value; only `.env.example` references it).

## Exit criteria (all human-verified)
1. `pytest` runs and passes (a trivial smoke test is fine at this stage).
2. `pip install -e .` (or `pip install -r requirements.txt`) succeeds in a fresh venv.
3. `CLAUDE.md`, `DECISIONS.md`, `RISKS.md`, `.env.example`, `README.md` stub, and the full directory tree exist exactly as specified below.
4. `.gitignore` excludes `.env`, `data/`, `cache/`, model artifacts, and `__pycache__`.
5. First commit made: `phase 0: scaffold and governance`.

## Directory structure to create

```
cfb-model/
├── CLAUDE.md
├── DECISIONS.md
├── RISKS.md
├── README.md                # stub only; real content is Phase 7
├── Makefile                 # targets stubbed: ingest, features, train, evaluate, reproduce, test
├── pyproject.toml           # or requirements.txt — see Assumptions
├── .env.example             # CFBD_API_KEY=your-key-here
├── .gitignore
├── src/cfb/
│   ├── __init__.py
│   ├── config.py            # paths, season range (2014–2025), constants; reads .env
│   ├── ingest/              # Phase 1
│   ├── vegas/               # Phase 2
│   ├── elo/                 # Phase 3
│   ├── features/            # Phase 4
│   ├── model/               # Phase 5
│   └── eval/                # Phase 6
├── data/                    # gitignored: cfb.sqlite, features/*.parquet
├── cache/                   # gitignored: raw API JSON responses
├── gold/                    # committed hand-verified fixtures (populated Phase 1+)
├── tests/
│   └── test_smoke.py
└── app/                     # Phase 8
```

## Files and responsibilities

- **`CLAUDE.md`** — project conventions Claude Code must follow every session. Must contain, verbatim in spirit:
  - The honesty-over-inflation rule: "The project's claim is calibration approaching the de-vigged Vegas closing line. Never write code, comments, docs, or UI text implying the model beats the market."
  - The anti-leakage invariant: "Every feature for a game uses only information available strictly before that game's kickoff. All train/val/test splits are season-forward. Random splits are a bug."
  - Governance: "Log every non-obvious choice to DECISIONS.md with a one-line rationale. Never fabricate, interpolate, or impute missing source data silently — document the gap in RISKS.md and handle it explicitly."
  - Session ritual: "At session start, read CLAUDE.md, DECISIONS.md, RISKS.md before writing code."
  - Style: Python 3.11+, type hints, small pure functions for anything touching time ordering, pytest for all logic.
- **`DECISIONS.md`** — append-only. Format: `YYYY-MM-DD | decision | rationale`. Seed with Phase 0 decisions (Python version, package layout, dependency pinning approach).
- **`RISKS.md`** — table of risks, each with likelihood, impact, mitigation. Seed with at least: (1) feature leakage → mitigation: Phase 4 audit gate + season-forward splits; (2) CFBD API gaps/missing lines → mitigation: coverage assertions, document gaps, never interpolate; (3) small-sample opponents (FCS) → mitigation: explicit Elo policy in Phase 3; (4) calibration drift across seasons → mitigation: reliability curves per season in Phase 6; (5) rate limits → mitigation: response cache + throttled client.
- **`pyproject.toml`** — pinned deps: `requests`, `pandas`, `pyarrow`, `xgboost` *or* `lightgbm` (pick one now — see decision point), `scikit-learn`, `pytest`, `python-dotenv`, `matplotlib`. No Streamlit yet (Phase 8).
- **`Makefile`** — stub each target with `@echo "not implemented until phase N"` so `make reproduce` fails loudly, not silently.
- **`src/cfb/config.py`** — single source of truth for: `SEASONS = range(2014, 2026)`, DB path, cache dir, feature-store dir. No other module hardcodes paths or season ranges.

## Tests
- `tests/test_smoke.py`: imports `cfb.config`, asserts season range and that paths are constructible. (Leakage-sensitive tests begin in Phase 1; there is no data yet.)

## Human decision / review points
- **GBT library:** recommend **LightGBM** (faster iteration on tabular data, native categorical support); XGBoost equally defensible. Human confirms; log to `DECISIONS.md`.
- **Review `CLAUDE.md` wording yourself** — this file steers every future session; it is the highest-leverage 30 lines in the repo.

## Assumptions (stated, not buried)
- `pyproject.toml` over `requirements.txt` (modern default, editable install, single file). If the human prefers requirements.txt, say so before Phase 1.
- Season range ends at 2025 (last completed season as of mid-2026). Update `config.py` when 2026 completes — one line, by design.
