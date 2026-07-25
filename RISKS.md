# RISKS

Known risks and data gaps. Add a row the moment a gap is discovered — never silently
work around one.

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | **Feature leakage** — a feature encodes information from at or after kickoff (final scores, season-end aggregates, post-game rankings), inflating every downstream metric. | High | Critical — invalidates all results | Phase 4 leakage audit as a hard gate, running against production feature code; poisoned-feature test proving the audit can fail; season-forward splits everywhere; Vegas lines excluded from features by rule. |
| 2 | **CFBD API gaps** — missing games, missing closing lines, missing box-score stats for some seasons or FCS opponents. | High | Moderate — silently biases samples | Coverage assertions at ingestion with explicit per-season row counts; gaps excluded and documented here, never interpolated; `gold/` fixtures pin hand-verified rows. |
| 3 | **Small-sample opponents (FCS)** — teams with few or no rated games distort Elo and feature quality. | High | Moderate | Explicit FCS Elo policy defined in Phase 3 (documented in `DECISIONS.md`); FCS games excluded from training in Phase 5, retained for Elo bookkeeping only. |
| 4 | **Calibration drift across seasons** — a model calibrated on one season is miscalibrated on later ones (rule changes, transfer portal, scoring-environment shifts). | Medium | Moderate | Reliability curves reported per season in Phase 6, not pooled; isotonic calibration fit on a dedicated held-out season; drift stated in the model card rather than averaged away. |
| 5 | **API rate limits** — throttling or bans mid-backfill, producing partial data that looks complete. | Medium | Moderate | On-disk response cache in `cache/`; throttled client with backoff; ingestion is resumable and idempotent; row counts verified after every backfill. |
| 6 | **2020 season anomaly** — COVID-shortened, conference-only schedules, uneven game counts. | Certain | Moderate | Treated as a documented data gap, not normalized away. Per-season counts surfaced in Phase 1; inclusion/exclusion decided explicitly in Phase 5 and logged to `DECISIONS.md`. |
| 7 | **Over-claiming drift in prose** — READMEs, docstrings, and UI strings drifting toward "beats the market" language. | Medium | High — reputational, and the honest result is the point | `test_no_overclaiming.py` string scan in CI from Phase 6; the claim is fixed as *calibration approaching the de-vigged closing line*. |
| 8 | **De-vig / spread sign-convention error** — an inverted sign or a wrong vig removal makes the benchmark wrong while every test still passes. | Medium | Critical — the benchmark is the yardstick | Phase 2 verifies the conversion by hand against a known game held in `gold/`; sign convention documented in `DECISIONS.md` and asserted in tests. |
