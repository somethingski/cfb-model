# gold

Committed hand-verified fixtures. These are the only values in the project that a human
has checked against the source rather than the pipeline checking itself.

## `games_fixture.json`

Six to ten games spanning eras and edge cases — an early season, the COVID season, a
neutral site, an FBS-vs-FCS matchup, a bowl game, the most recent season, and a line with
moneylines on each side of the favourite.

Candidates are proposed from the built database:

```bash
python scripts/make_gold_fixture.py
```

That script produces a worksheet, not evidence. Because the values come from the same
ingestion path the tests exercise, a freshly generated fixture would pass its own test
while proving nothing. It becomes meaningful only once a human has done this:

1. Open each `game_id` on collegefootballdata.com (or the team's game page).
2. Confirm by eye: both teams, the final score, the kickoff time, and the spread /
   over-under / moneylines for the recorded provider.
3. Set `human_verified` to `true` and fill in `verified_by` and `verified_on`.

`tests/test_gold_games.py` fails while `human_verified` is false. That failure is the
point: a skipped gate is a forgotten gate.

Once verified, the fixture is a regression test — if a later change to the client, the
transforms, or the schema alters any of these values, that test is what notices.

## `vegas_fixture.json`

Five games pinning the de-vig arithmetic: a home favourite and an away favourite with
near-mirrored numbers (−7.0 with −300/+250 against +8.0 with +250/−300, so an inverted
sign convention cannot satisfy both), a pick 'em, a pre-moneyline-era spread-only game,
and the widest spread in the database.

```bash
python scripts/make_gold_vegas_fixture.py
```

This one works differently from `games_fixture.json`, and the difference matters. A game's
score can be checked against an external source; a de-vigged probability cannot — there is
nowhere to look it up. So the generator emits `hand_computed` as **null** and you fill it
in yourself:

1. For each game, compute `p_home_devig = Φ(-spread / sigma)` with a calculator and a
   Z-table. `sigma` is at the top of the fixture.
2. Where moneylines are present, compute `p_home_moneyline`: convert each side with
   `-m → m/(m+100)` and `+p → 100/(p+100)`, then divide the home implied probability by
   the sum of the two.
3. Fill in every `hand_computed` value, then set `human_verified`, `verified_by`,
   `verified_on`.

Do **not** compute these by running this project's code. The pipeline's numbers are in the
`pipeline` block already; the whole point is that yours arrive independently. The test
compares the two to 4 decimal places.

Re-running the generator preserves any `hand_computed` values already filled in — that
arithmetic is the one thing here that cannot be regenerated.

## `eval_fixture.json`

One week of the held-out period — 2023 week 1, 52 games — with the model's probability, the
de-vigged line's probability, and the outcome for each. Phase 6's exit criterion 4.

```bash
python scripts/make_eval_fixture.py
```

Same shape as `vegas_fixture.json`: `hand_computed` is emitted null and you fill it in.

1. Paste the `p_model`, `p_vegas` and `home_win` columns into a spreadsheet.
2. Compute the mean of `(p_model - home_win)^2`, and the same for `p_vegas`.
3. Count the home wins — one cell, and it isolates the label from the probabilities.
4. Fill in `hand_computed`, then set `human_verified`, `verified_by`, `verified_on`.

What this catches that the unit tests cannot: `tests/test_metrics.py` proves the Brier
formula is right on four games written into the test file, but it cannot prove the pipeline
applied that formula to the right games, with the right labels, in the right order. That is
the aggregation rather than the arithmetic, and it is exactly what survives a green suite.
A spreadsheet catches it because a spreadsheet is not built out of the same parts.

The probabilities handed over are the pipeline's own, so this is not a check on the model —
it is a check that the headline in `results/results_table.md` is the mean squared error of
the numbers the model actually produced on the games it actually scored.

`tests/test_gold_eval.py` fails while `hand_computed` is blank. The test compares to 6
decimal places, tighter than the Phase 2 fixture's 4, because averaging a spreadsheet column
needs no Z-table and the two should agree exactly.
