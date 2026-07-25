"""Regression test against hand-verified games.

The fixture is the one place in the pipeline where a human has looked at the source and
said "yes, that is what happened". Everything else is the ingestion code checking its own
homework. If a future change to the client, the transforms, or the schema alters any of
these values, this test is what notices.

``gold/games_fixture.json`` is proposed by ``scripts/make_gold_fixture.py`` and is not
evidence of anything until a human has checked each game against collegefootballdata.com
and set ``human_verified`` to true.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from cfb import config

FIXTURE_PATH = config.GOLD_DIR / "games_fixture.json"


def load_fixture() -> dict:
    """Read the gold fixture, or an empty stand-in when it has not been generated."""
    if not FIXTURE_PATH.exists():
        return {"human_verified": False, "games": []}
    return json.loads(FIXTURE_PATH.read_text())


FIXTURE = load_fixture()
GAMES = FIXTURE.get("games", [])

pytestmark = pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason=f"no fixture at {FIXTURE_PATH}; run scripts/make_gold_fixture.py",
)


def test_human_has_verified_the_fixture() -> None:
    """The gate for exit criterion 3. Red until a human has actually checked the games.

    This is deliberately a failure rather than a skip: a skipped gate is one that gets
    forgotten, and the whole value of the fixture is that a person looked at it.
    """
    if not FIXTURE.get("human_verified"):
        pytest.fail(
            f"{FIXTURE_PATH} has human_verified=false. Open each game on "
            "collegefootballdata.com, confirm the teams, final score, kickoff and line by "
            "eye, then set human_verified/verified_by/verified_on in the fixture."
        )


def test_fixture_spans_the_documented_edge_cases() -> None:
    cases = {game["case"] for game in GAMES}
    assert len(GAMES) >= 6, "the plan asks for 6-10 hand-checked games"
    assert {"covid season", "neutral site", "postseason"} <= cases


@pytest.mark.integration
@pytest.mark.parametrize("expected", GAMES, ids=[game.get("case", "?") for game in GAMES])
class TestFixtureMatchesDatabase:
    def test_game_row_matches(self, built_db: sqlite3.Connection, expected: dict) -> None:
        row = built_db.execute(
            """
            SELECT g.season, g.week, g.season_type, g.start_date, g.neutral_site,
                   home.school AS home_team, away.school AS away_team,
                   g.home_points, g.away_points
            FROM games g
            JOIN teams home ON home.team_id = g.home_team_id
            JOIN teams away ON away.team_id = g.away_team_id
            WHERE g.game_id = ?
            """,
            (expected["game_id"],),
        ).fetchone()
        assert row is not None, f"game {expected['game_id']} is missing from the database"

        for column in (
            "season",
            "week",
            "season_type",
            "start_date",
            "neutral_site",
            "home_team",
            "away_team",
            "home_points",
            "away_points",
        ):
            assert row[column] == expected[column], f"{column} differs for {expected['game_id']}"

    def test_line_row_matches(self, built_db: sqlite3.Connection, expected: dict) -> None:
        row = built_db.execute(
            """
            SELECT spread, over_under, home_moneyline, away_moneyline
            FROM lines WHERE game_id = ? AND provider = ?
            """,
            (expected["game_id"], expected["provider"]),
        ).fetchone()
        assert row is not None, f"no {expected['provider']} line for game {expected['game_id']}"

        for column in ("spread", "over_under", "home_moneyline", "away_moneyline"):
            assert row[column] == expected[column], f"{column} differs for {expected['game_id']}"
