"""SQLite schema for ingested CFBD data.

``start_date`` on ``games`` is the canonical leakage clock for the entire project.
Every later phase orders by it, so it is ``NOT NULL`` at the storage layer rather than
merely asserted afterwards: a game with no kickoff time must fail at insert time, not
survive to become a silent hole in the clock.

Deliberately absent: the pregame/postgame Elo, win-probability, and excitement-index
columns that ``/games`` also returns. Post-kickoff information that is never stored
cannot leak into a feature by accident.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS teams (
    team_id      INTEGER PRIMARY KEY,
    school       TEXT NOT NULL UNIQUE,
    mascot       TEXT,
    abbreviation TEXT
);

CREATE TABLE IF NOT EXISTS team_seasons (
    team_id        INTEGER NOT NULL,
    season         INTEGER NOT NULL,
    conference     TEXT,
    division       TEXT,
    classification TEXT,
    PRIMARY KEY (team_id, season),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS games (
    game_id         INTEGER PRIMARY KEY,
    season          INTEGER NOT NULL,
    week            INTEGER NOT NULL,
    season_type     TEXT NOT NULL,
    start_date      TEXT NOT NULL,
    start_time_tbd  INTEGER NOT NULL DEFAULT 0,
    neutral_site    INTEGER NOT NULL DEFAULT 0,
    conference_game INTEGER,
    home_team_id    INTEGER NOT NULL,
    away_team_id    INTEGER NOT NULL,
    home_points     INTEGER,
    away_points     INTEGER,
    completed       INTEGER NOT NULL,
    FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS game_team_stats (
    game_id    INTEGER NOT NULL,
    team_id    INTEGER NOT NULL,
    is_home    INTEGER NOT NULL,
    stat_name  TEXT NOT NULL,
    stat_value REAL,
    stat_raw   TEXT NOT NULL,
    PRIMARY KEY (game_id, team_id, stat_name),
    FOREIGN KEY (game_id) REFERENCES games(game_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS lines (
    game_id         INTEGER NOT NULL,
    provider        TEXT NOT NULL,
    spread          REAL,
    spread_open     REAL,
    over_under      REAL,
    over_under_open REAL,
    home_moneyline  INTEGER,
    away_moneyline  INTEGER,
    PRIMARY KEY (game_id, provider),
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

CREATE INDEX IF NOT EXISTS idx_games_season_week ON games(season, week);
CREATE INDEX IF NOT EXISTS idx_games_start_date  ON games(start_date);
CREATE INDEX IF NOT EXISTS idx_stats_team_game   ON game_team_stats(team_id, game_id);
CREATE INDEX IF NOT EXISTS idx_lines_game        ON lines(game_id);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with foreign keys enforced and rows accessible by name.

    Args:
        db_path: Path to the SQLite file. Parent directories are created if needed.

    Returns:
        An open connection. The caller owns closing it.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create tables and indexes if they do not already exist.

    Args:
        conn: An open connection.
    """
    conn.executescript(SCHEMA_SQL)
    conn.commit()
