"""Generate candidate rows for ``gold/games_fixture.json`` from the built database.

This script only *proposes* the fixture. The values it writes come from the same
ingestion path the tests check, so on its own it proves nothing — it is a worksheet, not
evidence. The fixture becomes meaningful when a human opens each game on
collegefootballdata.com, confirms the score, kickoff, and line by eye, and flips
``human_verified`` to true. ``tests/test_gold_games.py`` fails until that happens.

Games are chosen deterministically (lowest ``game_id`` matching each case) so re-running
proposes the same set rather than quietly swapping in different games.

Usage:
    python scripts/make_gold_fixture.py
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

from cfb import config
from cfb.ingest.schema import connect

# Each case pins a different way ingestion could go wrong: an old era, the COVID season,
# a neutral site, an FBS-vs-FCS matchup, a bowl game, and the most recent season. Seasons
# are pinned per case so the fixture actually spans eras — without that, ordering by
# game_id lands almost every case in week 1 of 2014.
#
# The two moneyline cases are a matched pair on either side of the favourite. Together
# they pin the spread sign convention (negative = home favoured) in hand-checked data,
# which is what RISKS #8 asks for.
CASES: list[tuple[str, str]] = [
    ("oldest season", "g.season = 2014 AND g.neutral_site = 0"),
    ("covid season", "g.season = 2020"),
    ("neutral site", "g.season = 2018 AND g.neutral_site = 1 AND g.season_type = 'regular'"),
    (
        "fbs hosting fcs",
        """g.season = 2022 AND EXISTS (SELECT 1 FROM team_seasons ts
                   WHERE ts.team_id = g.away_team_id AND ts.season = g.season
                     AND ts.classification != 'fbs')""",
    ),
    ("postseason", "g.season = 2019 AND g.season_type = 'postseason'"),
    ("most recent season", "g.season = (SELECT MAX(season) FROM games)"),
    (
        "home favourite with moneylines",
        "g.season = 2021 AND l.home_moneyline IS NOT NULL AND l.spread <= -3",
    ),
    (
        "away favourite with moneylines",
        "g.season = 2023 AND l.home_moneyline IS NOT NULL AND l.spread >= 3",
    ),
]

# Prefer providers a human can look up easily. Deterministic, and it keeps the fixture off
# model-derived lines like numberfire when a book or the consensus is available.
PROVIDER_PREFERENCE = """
    CASE l.provider
        WHEN 'consensus'  THEN 0
        WHEN 'DraftKings' THEN 1
        WHEN 'Bovada'     THEN 2
        WHEN 'ESPN Bet'   THEN 3
        ELSE 4
    END
"""

# Provider choices made by a human during verification, recorded so that regenerating
# reproduces the committed fixture rather than silently reverting the decision.
PROVIDER_OVERRIDE: dict[str, str] = {
    # Bovada quotes both sides of this game; DraftKings' moneylines were the alternative.
    "away favourite with moneylines": "Bovada",
}

SELECT = """
SELECT g.game_id, g.season, g.week, g.season_type, g.start_date, g.neutral_site,
       home.school AS home_team, away.school AS away_team,
       g.home_points, g.away_points,
       l.provider, l.spread, l.over_under, l.home_moneyline, l.away_moneyline
FROM games g
JOIN teams home ON home.team_id = g.home_team_id
JOIN teams away ON away.team_id = g.away_team_id
JOIN lines l    ON l.game_id = g.game_id
WHERE g.completed = 1 AND l.spread IS NOT NULL AND {predicate}
ORDER BY g.game_id, {provider_preference}, l.provider
LIMIT 1
"""


def pick(conn: sqlite3.Connection, label: str, predicate: str) -> dict | None:
    """Select the lowest-id completed game matching a case, with one provider's line.

    Args:
        conn: Connection to the built database.
        label: Human-readable description of the edge case being pinned.
        predicate: SQL fragment narrowing to that case.

    Returns:
        A fixture entry, or None if no game matches.
    """
    provider = PROVIDER_OVERRIDE.get(label)
    if provider:
        predicate = f"{predicate} AND l.provider = ?"
    row = conn.execute(
        SELECT.format(predicate=predicate, provider_preference=PROVIDER_PREFERENCE),
        (provider,) if provider else (),
    ).fetchone()
    if row is None:
        return None
    entry = {"case": label}
    entry.update({key: row[key] for key in row.keys()})
    return entry


def print_worksheet(conn: sqlite3.Connection, entries: list[dict]) -> None:
    """Print what a human needs in order to check each game against a website.

    Two things make hand-checking error-prone, and both are addressed here rather than
    left for the reader to do in their head:

    * ``start_date`` is stored in UTC, but sites render kickoff in local time. The Eastern
      rendering is shown alongside so a 7:30pm on screen is recognisably a 7:30pm here.
    * The fixture pins **one** provider, but a site may show a different book's number.
      Every provider CFBD returned for the game is listed, with the pinned one marked, so
      a mismatch reads as "wrong row" rather than "wrong data".

    Args:
        conn: Connection to the built database.
        entries: The fixture entries just written.
    """
    print("\n" + "=" * 78)
    print("VERIFICATION WORKSHEET — check each game, then set human_verified to true")
    print("=" * 78)
    for entry in entries:
        kickoff = datetime.fromisoformat(entry["start_date"])
        eastern = kickoff.astimezone(ZoneInfo("America/New_York"))
        print(f"\n[{entry['case']}]  game_id {entry['game_id']}")
        print(
            f"  {entry['away_team']} @ {entry['home_team']}   final {entry['away_points']}-"
            f"{entry['home_points']}   {entry['season']} week {entry['week']} "
            f"({entry['season_type']}){'  NEUTRAL SITE' if entry['neutral_site'] else ''}"
        )
        print(f"  kickoff  {entry['start_date']}  (UTC, this is what the test asserts)")
        print(
            f"           {eastern.strftime('%Y-%m-%d %I:%M %p')} US Eastern  (for comparing "
            "against a site)"
        )
        print(f"  {'provider':30} {'spread':>8} {'O/U':>7} {'home ML':>9} {'away ML':>9}")
        for line in conn.execute(
            "SELECT provider, spread, over_under, home_moneyline, away_moneyline "
            "FROM lines WHERE game_id = ? ORDER BY provider",
            (entry["game_id"],),
        ):
            mark = " <-- pinned" if line["provider"] == entry["provider"] else ""
            print(
                f"  {line['provider']:30} {str(line['spread']):>8} {str(line['over_under']):>7} "
                f"{str(line['home_moneyline']):>9} {str(line['away_moneyline']):>9}{mark}"
            )


def main() -> int:
    """Write the proposed fixture to ``gold/games_fixture.json``."""
    conn = connect(config.DB_PATH)
    try:
        entries = []
        seen: set[int] = set()
        for label, predicate in CASES:
            entry = pick(conn, label, predicate)
            if entry is None:
                print(f"no game matched case {label!r}")
                continue
            if entry["game_id"] in seen:
                print(f"case {label!r} matched an already-included game; skipping")
                continue
            seen.add(entry["game_id"])
            entries.append(entry)
        print_worksheet(conn, entries)
    finally:
        conn.close()

    out = config.GOLD_DIR / "games_fixture.json"
    payload = {
        "human_verified": False,
        "verified_by": None,
        "verified_on": None,
        "source": "https://collegefootballdata.com — check each game by eye before flipping "
        "human_verified to true",
        "games": entries,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(entries)} candidate games to {out}")
    print("Now verify each one by hand, then set human_verified to true.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
