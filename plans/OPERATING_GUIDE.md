# Operating Guide — Driving Claude Code Through This Build

For you, not for Claude Code. Keep it open in a tab.

## The core loop (every phase)
1. **Fresh session.** One phase per session, always. Paste the phase plan.
2. **Restate before code.** First instruction: "Read CLAUDE.md, DECISIONS.md, RISKS.md. Then restate this phase's preconditions, exit criteria, and files before writing anything." If the restatement is wrong, fix it now — it's the cheapest correction point you'll get.
3. **Let it build.** Free rein within the plan.
4. **Run the tests yourself.** `pytest` on your machine. "Tests pass" in its output is a claim, not evidence.
5. **Verify each exit criterion against real output** — actual row counts, actual numbers, actual files. Check them off literally.
6. **Commit once per phase**, message = phase name.

## The rules that matter most
- One phase per session; fresh context between phases.
- CLAUDE.md / DECISIONS.md / RISKS.md read at every session start; non-obvious choices logged to DECISIONS.md as they happen, not retroactively.
- **The Phase 4 leakage audit is a hard gate.** If it fails, the fix is in the features, never in the audit. Any session that proposes loosening a tolerance, shrinking the sample, or skipping a check to get green is a session you stop and redirect.
- Vegas lines are never model features. If a line-derived column ever appears in the feature list, that's a stop-everything moment.

## Steering corrections you will likely need
- **Over-claiming in prose.** It will drift toward "outperforms" and "edge" in READMEs and docstrings. The claim is *approaches the closing line*. The `test_no_overclaiming.py` scan exists because this correction is needed more than once.
- **Skipping boring baselines.** Insist on the naive home baseline and Elo-only rows in every results table. The story is the gap structure, and the gap needs both ends.
- **Random splits.** Any `train_test_split`, `KFold`, or `shuffle=True` on game rows is a bug. Season-forward only, including inside hyperparameter search (forward-chaining CV).
- **Fabricated or interpolated data.** Missing lines, missing stats, weird 2020 counts: the answer is exclusion + a RISKS.md entry, never imputation from thin air. If it "fixes" a gap without you asking, revert.
- **Too-good numbers.** If the model matches or beats Vegas anywhere, your prior is *bug*, not breakthrough. The Phase 6 tripwire encodes this, but hold the prior yourself too.

## Let it run vs. review closely
**Free rein:** repo scaffolding, SQLite schema, ingestion plumbing, caching, Makefile, plot code, Streamlit layout, test boilerplate.
**Slow, line-by-line review — the four silent-error sites** (each can invalidate results without ever throwing):
1. **De-vig + spread sign convention** (Phase 2) — verify against a game you check by hand.
2. **Elo snapshot-before-update chronology** (Phase 3) — read `pipeline.py`'s loop yourself.
3. **The leakage audit** (Phase 4) — confirm it calls production feature code and that the poisoned-feature test proves it can fail.
4. **The calibration split** (Phase 5) — isotonic fit on 2022 only; the guard assertion exists and fires.

## Realistic sequencing
Phases 0–3 are fast — a session each, maybe two for ingestion if the API misbehaves. **Phase 4 deserves the most calendar time and the most of your attention: the anti-leakage framework is the resume story more than the model is.** Budget it like a phase and a half. Phases 5–6 are quick once 4 is solid — the model is small and the evaluation is mechanical. 7 is mostly your writing judgment; 8 is a fun session.

## What "done" looks like
- `make reproduce` runs clean from empty state on your machine.
- All tests pass on your machine, including the audit, the poisoned-feature test, and the over-claiming scan.
- README states an honest calibration result with real numbers, headline framed as gap-to-Vegas and %-of-gap-closed.
- DECISIONS.md explains every choice you'd be asked about in an interview — de-vig method, K-factor, FCS policy, split scheme, isotonic-not-Platt, FCS-game exclusion from training, Streamlit-not-FastAPI.
- You can demo the app and narrate the two numbers on screen without hedging beyond what's already written into the UI.
