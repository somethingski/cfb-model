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
