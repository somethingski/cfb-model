# Phase 8 — Running Predictions: Streamlit Demo (Option B)

## The choice, stated
**Option B (simple Streamlit frontend), with a one-page "how predictions work" doc folded in as a sidebar/expander.** Rationale: in an interview, thirty seconds of a live, honest side-by-side beats five minutes of you narrating a document — the demo *is* the explainer when the benchmark comparison is on screen. Option A's content isn't lost: the README (Phase 7) already teaches the pipeline end to end, and the app's "How to read this" expander carries the interpretation guidance (what calibrated vs. suspicious output looks like, how to sanity-check). What Option A adds beyond that doesn't justify a separate deliverable; what Option B adds — a demoable artifact — does. Log to `DECISIONS.md`.

## Preconditions
- Phase 7 complete; `v1.0` tagged; `make reproduce` verified clean.
- Trained model + calibrator present; feature pipeline importable as a library.

## Exit criteria (human-verified)
1. `make app` (→ `streamlit run app/app.py`) starts with no setup beyond the existing env + `pip install streamlit`.
2. For an upcoming (unplayed) week: pick week → the app lists scheduled FBS games, or pick two teams; it shows **model probability, de-vigged current Vegas probability, and the gap, side by side, unconditionally visible** — the benchmark is never behind a click.
3. For a historical game: same view plus the actual result, clearly labeled as hindsight.
4. The calibration reliability plot from Phase 6 renders on the main page with one line of interpretation.
5. A persistent footer/banner: *"Portfolio calibration project. This model approaches, and does not beat, the market. Not a betting tool."* — and `test_no_overclaiming.py` extended to cover all UI strings.
6. Upcoming-week features pass a **live leakage self-check** before display (below).
7. Commit: `phase 8: streamlit demo`.

## Design

- **Single file `app/app.py`** + `app/predict.py` (the only new logic). Streamlit over FastAPI+static: one dependency, one command, zero frontend build — accessibility requirement wins.
- **`predict.py` — the upcoming-game path** (the only genuinely new code):
  1. Fetch the target week's scheduled games via the existing cached CFBD client (`/games` for current season returns future games).
  2. Compute features for unplayed games with the **same Phase 4 functions**, with `as_of = now`. No reimplementation — reimplementing the feature logic for serving is how serving/training skew is born.
  3. Fetch current lines via `/lines`, de-vig with Phase 2 functions; if no line exists yet, display "no line posted" — never a placeholder number.
  4. Predict with the persisted model + calibrator.
- **Live leakage self-check:** before rendering, assert every input game's `start_date > now` (for the upcoming view) and that no feature depended on data timestamped ≥ now (re-use the audit's dependency-timestamp machinery from Phase 4 in a lightweight mode). Refuse to render on failure — same gate philosophy, extended to serving.
- **"How to read this" expander** (Option A's soul, ~300 words): calibrated probability means "of games I call 70%, about 70% resolve yes"; a healthy model sits *near* the line with small, unsystematic gaps; a suspicious output is one far from the line or systematically on one side — investigate the model before doubting the market; sanity-check by asking what the model can't know (injuries this week, suspensions, weather).
- **Current-season caveat displayed in-app:** early-season predictions rest on regressed Elo and thin rolling stats; the app labels weeks 1–3 as "low-information."

## Files
- `app/app.py` — UI only; no business logic.
- `app/predict.py` — upcoming-game feature assembly + prediction, importing Phases 1/2/4/5 code.
- `tests/test_predict_serving.py` — see Tests.
- `Makefile` — `app` target; `pyproject.toml` gains a `[project.optional-dependencies] app = ["streamlit"]` extra (core pipeline stays streamlit-free).

## Tests
- **Serving/training-skew test (the leakage-class test for this phase):** for 20 historical games, run them through the *serving* path (`predict.py` with `as_of` = their kickoff) and assert features and probabilities are identical to the Phase 4 store / Phase 5 batch predictions. If serving ever diverges from training, this fails.
- UI-string honesty: extend the banned-claims scan to `app/`.
- No-line handling: a game without a posted line renders the model probability with an explicit "no benchmark available" state, not a crash and not a fake number.

## Human decision / review points
- Confirm Streamlit over FastAPI (recommended above).
- Demo rehearsal: run the app, walk the interview script once — pick a week, explain the two numbers, open the expander, point at the calibration plot. If any screen makes you reach for a hedge you haven't written down, write it into the UI.

## Assumptions
- During the offseason (now, July), "upcoming week" means Week 1 of 2026 once CFBD posts the schedule and early lines; until lines post, the historical view is the demo path. The app handles both without special-casing.
