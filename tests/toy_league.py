"""A small synthetic league with a built database, for tests that need one end to end.

Real-database tests skip when the backfill has not been run, which is exactly the wrong
property for the tests that prove the leakage audit can fail. Those have to run every time.
The league here is deterministic — scores come from a seeded generator — so a failure is
reproducible.

Nothing in this module is production code. It exists to be fed to production code.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass

from cfb.elo.pipeline import build as build_elo
from cfb.elo.pipeline import load_params
from cfb.ingest.schema import init_db

TEAMS: tuple[int, ...] = (101, 102, 103, 104, 105, 106)
SEASONS: tuple[int, ...] = (2014, 2015, 2016)
SEED: int = 17


@dataclass(frozen=True)
class ToyGame:
    """One scheduled toy game."""

    game_id: int
    season: int
    week: int
    start_date: str
    home: int
    away: int
    home_points: int
    away_points: int


def schedule() -> list[ToyGame]:
    """A single round robin per season, one game per matchup, one matchup per week slot.

    Returns:
        Fifteen games per season across three seasons, in kickoff order.
    """
    rng = random.Random(SEED)
    games: list[ToyGame] = []
    game_id = 1
    for season in SEASONS:
        week = 1
        for index, home in enumerate(TEAMS):
            for away in TEAMS[index + 1 :]:
                # A team's strength is stable across seasons, so ratings have something to
                # find; the noise is large enough that no feature tracks the label exactly.
                edge = (home - away) * 1.5
                home_points = max(0, int(rng.gauss(28 + edge, 10)))
                away_points = max(0, int(rng.gauss(28 - edge, 10)))
                if home_points == away_points:
                    home_points += 3
                games.append(
                    ToyGame(
                        game_id=game_id,
                        season=season,
                        week=week,
                        start_date=f"{season}-09-{week:02d}T18:00:00+00:00",
                        home=home,
                        away=away,
                        home_points=home_points,
                        away_points=away_points,
                    )
                )
                game_id += 1
                week += 1
    return games


def populate(conn: sqlite3.Connection, with_box_scores: bool = True) -> list[ToyGame]:
    """Build a complete toy database: teams, games, box scores and pre-game Elo.

    Args:
        conn: An open connection to an empty database.
        with_box_scores: Whether to write ``game_team_stats``. False leaves every
            yards-per-play and pace feature null, which is its own useful fixture.

    Returns:
        The schedule that was written.
    """
    init_db(conn)
    rng = random.Random(SEED + 1)
    games = schedule()

    for team_id in TEAMS:
        conn.execute(
            "INSERT OR IGNORE INTO teams (team_id, school) VALUES (?, ?)",
            (team_id, f"Toy {team_id}"),
        )
        for season in SEASONS:
            conn.execute(
                "INSERT OR REPLACE INTO team_seasons (team_id, season, classification) "
                "VALUES (?, ?, 'fbs')",
                (team_id, season),
            )

    for game in games:
        conn.execute(
            """
            INSERT INTO games (
                game_id, season, week, season_type, start_date, start_time_tbd,
                neutral_site, conference_game, home_team_id, away_team_id,
                home_points, away_points, completed
            ) VALUES (?, ?, ?, 'regular', ?, 0, 0, 1, ?, ?, ?, ?, 1)
            """,
            (
                game.game_id,
                game.season,
                game.week,
                game.start_date,
                game.home,
                game.away,
                game.home_points,
                game.away_points,
            ),
        )
        if not with_box_scores:
            continue
        for team_id, points in ((game.home, game.home_points), (game.away, game.away_points)):
            rushes = rng.randint(25, 45)
            attempts = rng.randint(20, 40)
            yards = points * 12 + rng.randint(-60, 60)
            for name, value, raw in (
                ("totalYards", float(yards), str(yards)),
                ("rushingAttempts", float(rushes), str(rushes)),
                ("completionAttempts", None, f"{attempts // 2}-{attempts}"),
            ):
                conn.execute(
                    "INSERT INTO game_team_stats (game_id, team_id, is_home, stat_name, "
                    "stat_value, stat_raw) VALUES (?, ?, ?, ?, ?, ?)",
                    (game.game_id, team_id, int(team_id == game.home), name, value, raw),
                )
    conn.commit()

    params, _ = load_params()
    build_elo(conn, params)
    return games
