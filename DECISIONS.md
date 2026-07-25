# DECISIONS

Append-only log of non-obvious choices. Format: `YYYY-MM-DD | decision | rationale`.

| Date | Decision | Rationale |
|---|---|---|
| 2026-07-24 | Python **3.12.13** for the project venv (plan requires 3.11+) | 3.14.6 is also installed on this machine, but the scientific stack (pyarrow, LightGBM) does not reliably ship 3.14 wheels yet; 3.12 installs from wheels with no source builds. |
| 2026-07-24 | LightGBM as the GBT library (over XGBoost) | Faster iteration on small tabular data and native categorical handling. XGBoost was equally defensible; choice logged so Phase 5 does not relitigate it. |
| 2026-07-24 | `pyproject.toml` with an editable install, not `requirements.txt` | Single file for metadata and deps; `pip install -e .` makes `src/cfb` importable without path hacks. |
| 2026-07-24 | `src/` layout (`src/cfb/`) rather than a top-level `cfb/` package | Prevents accidentally importing the source tree instead of the installed package, which is how "works on my machine" test passes happen. |
| 2026-07-24 | Dependencies pinned with lower bounds (`>=`), not exact pins, at Phase 0 | Phase 0 has no numerical results to protect yet. Exact pins are added in Phase 7 alongside `make reproduce`, when reproducibility becomes a stated deliverable. |
| 2026-07-24 | Season range `2014–2025` lives only in `config.py` | Advancing to 2026 is a one-line change by design; no other module hardcodes it. |
| 2026-07-24 | Makefile targets stubbed with a failing `exit 1`, not a bare `echo` | `make reproduce` must fail loudly before the pipeline exists, so a green run can never be mistaken for a real one. |
