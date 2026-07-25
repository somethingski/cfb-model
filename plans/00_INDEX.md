# CFB Game-Outcome Model — Plan Series Index

Hand these to Claude Code **one file per session**, in order. Do not skip the Phase 4 gate.

| File | Phase | One-line scope |
|---|---|---|
| `phase0_scaffold.md` | 0 | Repo scaffold, governance docs, env, test harness |
| `phase1_ingestion.md` | 1 | CFBD ingestion → SQLite, caching, backfill, data-quality gates |
| `phase2_vegas.md` | 2 | Closing lines → de-vigged home-win probability benchmark |
| `phase3_elo.md` | 3 | Chronological Elo with MOV, HFA, preseason regression |
| `phase4_features.md` | 4 | Feature store + leakage audit — **hard gate** |
| `phase5_training.md` | 5 | Season-forward splits, GBT training, isotonic calibration |
| `phase6_evaluation.md` | 6 | Brier/log-loss vs. Vegas + baselines, reliability curves, model card |
| `phase7_writeup.md` | 7 | README-as-report, `make reproduce`, limitations |
| `phase8_frontend.md` | 8 | Streamlit demo app with honest benchmark display (Option B — rationale inside) |
| `OPERATING_GUIDE.md` | — | Your field manual for driving Claude Code (not for Claude Code) |

## Global conventions (repeated in every phase, enforced in CLAUDE.md)

- **Honesty over inflation.** The claim is *calibration approaching the de-vigged closing line*. No document, docstring, UI string, or commit message may state or imply the model beats the market.
- **Anti-leakage invariant.** Every feature for game *g* is a function of information available strictly before *g*'s kickoff. Splits are season-forward, never random.
- **Append-only governance.** Non-obvious choices → `DECISIONS.md` (with rationale). Known risks and data gaps → `RISKS.md` (with mitigation). Never fabricate or interpolate missing data; document the gap instead.
- **One phase per session.** Fresh Claude Code context each phase. Claude Code reads `CLAUDE.md`, `DECISIONS.md`, `RISKS.md` at session start.
- **Exit criteria are checked by the human** against real output, not by trusting "tests pass."
