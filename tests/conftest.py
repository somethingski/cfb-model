"""Shared fixtures: a hand-built toy database, and the real one when it exists."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from cfb import config
from cfb.ingest.schema import connect, init_db


@pytest.fixture
def toy_db(tmp_path: Path) -> sqlite3.Connection:
    """An empty database with the production schema applied."""
    conn = connect(tmp_path / "toy.sqlite")
    init_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def built_db() -> sqlite3.Connection:
    """The real ingested database, skipping the test when the backfill has not been run."""
    if not config.DB_PATH.exists():
        pytest.skip(f"no database at {config.DB_PATH}; run `make ingest` first")
    conn = connect(config.DB_PATH)
    yield conn
    conn.close()


def add_team(conn: sqlite3.Connection, team_id: int, school: str) -> None:
    """Insert a minimal team so foreign keys resolve."""
    conn.execute("INSERT OR IGNORE INTO teams (team_id, school) VALUES (?, ?)", (team_id, school))


def add_team_season(
    conn: sqlite3.Connection, team_id: int, season: int, classification: str = "fbs"
) -> None:
    """Record a team's subdivision for one season.

    The Elo walk reads subdivision per season, so a toy database needs these rows for a
    team to be rated at all.
    """
    add_team(conn, team_id, f"Team {team_id}")
    conn.execute(
        "INSERT OR REPLACE INTO team_seasons (team_id, season, classification) VALUES (?, ?, ?)",
        (team_id, season, classification),
    )


def add_game(conn: sqlite3.Connection, **overrides) -> int:
    """Insert a game with sane defaults, overridable per test.

    Returns:
        The inserted ``game_id``.
    """
    row = {
        "game_id": 1,
        "season": 2023,
        "week": 1,
        "season_type": "regular",
        "start_date": "2023-08-26T18:30:00+00:00",
        "start_time_tbd": 0,
        "neutral_site": 0,
        "conference_game": 0,
        "home_team_id": 87,
        "away_team_id": 2426,
        "home_points": 42,
        "away_points": 3,
        "completed": 1,
    }
    row.update(overrides)
    add_team(conn, row["home_team_id"], f"Home {row['home_team_id']}")
    add_team(conn, row["away_team_id"], f"Away {row['away_team_id']}")
    conn.execute(
        """
        INSERT OR REPLACE INTO games (
            game_id, season, week, season_type, start_date, start_time_tbd,
            neutral_site, conference_game, home_team_id, away_team_id,
            home_points, away_points, completed
        ) VALUES (
            :game_id, :season, :week, :season_type, :start_date, :start_time_tbd,
            :neutral_site, :conference_game, :home_team_id, :away_team_id,
            :home_points, :away_points, :completed
        )
        """,
        row,
    )
    return row["game_id"]
