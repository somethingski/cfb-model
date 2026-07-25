# Phase 2 — Vegas Line Handling (the Yardstick)

This phase produces the benchmark every result is measured against. It gets its own phase, its own tests, and slow human review. A silent sign error or bad de-vig here invalidates the entire evaluation without ever throwing.

## Preconditions
- Phase 1 complete; DB built; gold fixtures passing.
- Human has re-read the de-vig section below and confirmed the method choice.

## Exit criteria (human-verified)
1. Table `vegas_benchmark(game_id, provider, p_home_devig, source_type)` populated for every game with usable line data.
2. Hand-computed de-vig for 3 gold-fixture games matches the pipeline to 4 decimal places (human does the arithmetic independently — Sean-style exact check).
3. Sanity distribution printed: mean `p_home_devig` ≈ 0.55–0.60 (home teams win more often); vig removed (paired probs sum to 1.0 exactly).
   - **Amended 2026-07-25**, with evidence, from "no probabilities outside (0.01, 0.99)".
     That bound assumed a moneyline-derived benchmark; a spread-derived one puts 504 games
     (4.9%) outside it, reaching 0.99995 at spread −62. Clipping was rejected because it is
     the over-claim direction: among the 319 training games with `|spread| ≥ 37` the
     favourite won 318 (99.7%) against the normal model's ~99%, so clipping to 0.99 would
     understate a real effect and weaken the benchmark. Reported per build instead; RISKS #17.
   - The ≈0.55–0.60 band is met on the both-FBS population (0.5745, against an actual home
     win rate of 0.5784). Over all games the mean is 0.6146 against an actual 0.6200 — the
     gap is the 1,192 FBS-vs-FCS games averaging 0.92, which is the schedule, not miscalibration.
4. Coverage report: % of games per season in the benchmark set; games excluded are listed with reasons, never imputed.
5. Commit: `phase 2: vegas benchmark`.

## Method

### Source priority per game
1. **Moneylines** (preferred — direct market probabilities). Convert American odds:
   - Negative odds `-m`: implied = `m / (m + 100)`
   - Positive odds `+p`: implied = `100 / (p + 100)`
2. **Spread fallback** when no moneyline exists: convert closing spread to win probability via a normal-margin model, `p_home = Φ(-spread / σ)`, with `σ` estimated **from training seasons only (2014–2021)** by fitting actual margins against spreads. This is a leakage boundary: σ must never see validation/test seasons. `source_type` column records `moneyline` vs `spread` so Phase 6 can report sensitivity.

### De-vig: multiplicative normalization
`p_home = imp_home / (imp_home + imp_away)`, likewise away.

**Why multiplicative over Shin/power methods:** it is the standard, transparent, interview-defensible baseline; Shin corrects for insider-trading asymmetry that matters more in thin markets than in heavily-traded CFB closing lines; and since we use the benchmark for scoring (Brier/log loss) rather than betting, second-order de-vig refinements move the yardstick by less than model noise. Log to `DECISIONS.md`; note Shin as a possible robustness check in Phase 6, not a requirement.

### Provider selection
Prefer a single consistent provider (consensus if present, else the provider with the highest coverage across all seasons — measure and pick, don't guess). One provider per game; record which. Mixing books per game invites subtle inconsistencies.

## Files
- `src/cfb/vegas/odds.py` — pure functions: `american_to_implied()`, `devig_multiplicative()`, `spread_to_prob(spread, sigma)`. No I/O; fully unit-tested.
- `src/cfb/vegas/benchmark.py` — builds `vegas_benchmark` table; provider selection; σ estimation (train seasons only, value logged to `DECISIONS.md`).
- `gold/vegas_fixture.json` — 3+ games with hand-computed de-vigged probabilities (human computes these by hand).

## Tests
- Unit: known odds pairs → exact implied probabilities (e.g. -110/-110 → 0.5/0.5 after de-vig; -200/+170 → hand-checked values).
- Property: for any odds pair, de-vigged probs sum to 1.0 and each ∈ (0,1).
- `test_gold_vegas.py`: fixture regression.
- **Leakage test:** σ estimation function raises if passed any season > 2021.
- Sign-convention test: a heavy home favorite (e.g. spread −21) must map to `p_home > 0.9`. Sign errors in spreads are the classic silent killer here.

## Human decision / review points
- **Slow review, line by line:** the American-odds conversion, the spread sign convention (CFBD spreads are home-relative negative-favorite — verify against a known game, don't trust memory), and the σ fit.
- Confirm provider choice from the coverage report.

## Assumptions
- Moneyline coverage is partial, especially pre-2018; spread fallback carries real weight in early seasons. Reported, not hidden.
- "Closing" line = last recorded line per provider per game in CFBD. CFBD does not always timestamp line movement; document this limitation in `RISKS.md` (the benchmark may be slightly softer than true closers — which makes "approaching the benchmark" a *conservative* claim, the right direction to err).
