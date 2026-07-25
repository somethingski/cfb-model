"""Leakage-relevant: the kickoff clock cannot have holes.

``games.start_date`` is the canonical ordering key for every later phase. If a single
game carries a missing, unparseable, or non-UTC kickoff, every "strictly before kickoff"
guarantee downstream is unenforceable for that game — silently.

The poisoned cases below exist so this file can fail. A clock test that only ever sees
good data proves nothing.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC

import pytest

from cfb.ingest import quality, transform
from tests.conftest import add_game


def parse_all_start_dates(conn: sqlite3.Connection) -> list[str]:
    """Return every ``start_date`` in the database, asserting each parses as UTC.

    Args:
        conn: Connection to a database with the ``games`` table.

    Returns:
        The raw stored strings.

    Raises:
        AssertionError: If a value does not parse or is not UTC.
    """
    stored = [row["start_date"] for row in conn.execute("SELECT start_date FROM games")]
    for value in stored:
        from datetime import datetime

        parsed = datetime.fromisoformat(value)
        assert parsed.tzinfo is not None, f"{value!r} has no timezone"
        assert parsed.utcoffset() == UTC.utcoffset(None), f"{value!r} is not UTC"
    return stored


class TestPoisonedClock:
    def test_a_blank_kickoff_is_detected(self, toy_db: sqlite3.Connection) -> None:
        add_game(toy_db, game_id=1, start_date=" ")
        with pytest.raises(quality.DataQualityError):
            quality.check_no_null_start_dates(toy_db)

    def test_a_local_time_kickoff_is_detected(self, toy_db: sqlite3.Connection) -> None:
        """A naive local timestamp would order correctly *most* of the time — the worst case."""
        add_game(toy_db, game_id=1, start_date="2023-08-26T14:30:00")
        with pytest.raises(AssertionError, match="no timezone"):
            parse_all_start_dates(toy_db)

    def test_a_non_utc_kickoff_is_detected(self, toy_db: sqlite3.Connection) -> None:
        add_game(toy_db, game_id=1, start_date="2023-08-26T14:30:00-04:00")
        with pytest.raises(AssertionError, match="not UTC"):
            parse_all_start_dates(toy_db)

    def test_the_transform_refuses_to_store_a_game_with_no_kickoff(self) -> None:
        with pytest.raises(ValueError, match="leakage clock"):
            transform.game_row(
                {
                    "id": 1,
                    "season": 2023,
                    "week": 1,
                    "seasonType": "regular",
                    "startDate": None,
                    "homeId": 1,
                    "awayId": 2,
                }
            )

    def test_clean_data_passes(self, toy_db: sqlite3.Connection) -> None:
        add_game(toy_db, game_id=1)
        quality.check_no_null_start_dates(toy_db)
        assert parse_all_start_dates(toy_db) == ["2023-08-26T18:30:00+00:00"]


@pytest.mark.integration
class TestBuiltDatabaseClock:
    def test_every_game_has_a_parseable_utc_kickoff(self, built_db: sqlite3.Connection) -> None:
        stored = parse_all_start_dates(built_db)
        assert len(stored) > 0

    def test_no_completed_game_is_missing_a_kickoff(self, built_db: sqlite3.Connection) -> None:
        quality.check_no_null_start_dates(built_db)

    def test_kickoffs_are_stored_in_one_canonical_spelling(
        self, built_db: sqlite3.Connection
    ) -> None:
        """Later phases order by this column as text; mixed spellings would sort wrongly."""
        (odd,) = built_db.execute(
            "SELECT COUNT(*) FROM games WHERE start_date NOT LIKE '____-__-__T__:__:__+00:00'"
        ).fetchone()
        assert odd == 0

    def test_kickoffs_fall_inside_their_season(self, built_db: sqlite3.Connection) -> None:
        """A game stamped outside its own season window means the spine is mislabelled."""
        (odd,) = built_db.execute(
            """
            SELECT COUNT(*) FROM games
            WHERE CAST(SUBSTR(start_date, 1, 4) AS INTEGER) NOT IN (season, season + 1)
            """
        ).fetchone()
        assert odd == 0
