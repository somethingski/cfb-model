"""The leakage boundary of Phase 3, tested from both directions it can fail.

Elo has exactly two ways to see the future, and they need different tests:

1. **A future game reaches an earlier rating.** Caught by rebuilding on a database with
   every later game deleted and demanding byte-identical pre-game ratings.
2. **A game's own result reaches its own pre-game rating.** Truncation cannot catch this —
   the game is present in both databases — so it is caught separately by demanding that
   the very first game of the run carries the initial ratings.

Neither check is worth anything unless it can fail, so each is run against a deliberately
broken walk defined in this file. ``CLAUDE.md``: a test that cannot fail is not a test.
The broken walks live here and never in ``src/``.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from cfb import config
from cfb.elo.engine import EloParams, update
from cfb.elo.pipeline import (
    Game,
    PregameRow,
    RatingBook,
    load_classifications,
    load_games,
    load_params,
    run_elo,
)
from cfb.ingest.schema import connect

PARAMS = EloParams()

CUTOFF = "2018-10-20T00:00:00+00:00"
"""Mid-season week 8 of 2018: late enough that ratings carry real history, early enough
that a large future remains to be deleted."""


# --- deliberately broken walks, for proving the checks can fail ----------------


def leaky_walk(
    games: list[Game],
    classification: dict[tuple[int, int], str],
    params: EloParams,
    mode: str,
) -> dict[int, PregameRow]:
    """A wrong Elo walk, used only to prove the tests below can fail.

    Args:
        games: Games to process.
        classification: ``(team_id, season)`` to subdivision.
        params: Rating parameters.
        mode: ``"future"`` walks each season backwards, so a game is rated using games
            played after it. ``"own_result"`` walks correctly but applies each game's
            result before snapshotting, so every rating knows its own score.

    Returns:
        Pre-game rows by game id, in the same shape :func:`run_elo` produces.
    """
    if mode == "future":
        ordered = sorted(games, key=lambda g: g.start_date, reverse=True)
        ordered.sort(key=lambda g: g.season)
    else:
        ordered = sorted(games, key=lambda g: (g.start_date, g.game_id))

    book = RatingBook(classification, params)
    first_season = min(game.season for game in ordered)
    current_season = ordered[0].season
    rows: dict[int, PregameRow] = {}

    for game in ordered:
        if game.season != current_season:
            book.start_season(game.season)
            current_season = game.season

        home_pre = book.rating_for(game.home_team_id, game.season, first_season)
        away_pre = book.rating_for(game.away_team_id, game.season, first_season)

        if game.has_result:
            home_post, away_post = update(
                home_pre,
                away_pre,
                game.home_points,
                game.away_points,
                params,
                neutral_site=game.neutral_site,
            )
            book.apply(game.home_team_id, game.season, home_post)
            book.apply(game.away_team_id, game.season, away_post)
            if mode == "own_result":
                # The bug: re-read after applying, so the snapshot contains the result.
                home_pre = book.rating_for(game.home_team_id, game.season, first_season)
                away_pre = book.rating_for(game.away_team_id, game.season, first_season)

        rows[game.game_id] = PregameRow(
            game_id=game.game_id,
            season=game.season,
            home_team_id=game.home_team_id,
            away_team_id=game.away_team_id,
            home_elo_pre=home_pre,
            away_elo_pre=away_pre,
            home_is_fbs=book.is_fbs(game.home_team_id, game.season),
            away_is_fbs=book.is_fbs(game.away_team_id, game.season),
            neutral_site=game.neutral_site,
            home_points=game.home_points,
            away_points=game.away_points,
        )
    return rows


def synthetic_league() -> tuple[list[Game], dict[tuple[int, int], str]]:
    """A three-team, two-season league with enough games for order to matter."""
    schedule = [
        (1, 2014, "2014-09-06", 1, 2, 42, 7),
        (2, 2014, "2014-09-13", 2, 3, 21, 20),
        (3, 2014, "2014-09-20", 3, 1, 3, 45),
        (4, 2014, "2014-10-04", 1, 2, 14, 17),
        (5, 2014, "2014-10-11", 2, 3, 35, 31),
        (6, 2015, "2015-09-05", 3, 1, 28, 24),
        (7, 2015, "2015-09-12", 1, 2, 10, 38),
    ]
    games = [
        Game(
            game_id=game_id,
            season=season,
            start_date=f"{date}T18:00:00+00:00",
            neutral_site=False,
            home_team_id=home,
            away_team_id=away,
            home_points=home_points,
            away_points=away_points,
            completed=True,
        )
        for game_id, season, date, home, away, home_points, away_points in schedule
    ]
    classification = {(team, season): "fbs" for team in (1, 2, 3) for season in (2014, 2015)}
    return games, classification


def pregame_map(games: list[Game], classification: dict, params: EloParams) -> dict:
    """Production pre-game rows keyed by game id."""
    return {row.game_id: row for row in run_elo(games, classification, params).rows}


# --- 1. a future game must not reach an earlier rating -------------------------


@pytest.mark.integration
def test_deleting_the_future_changes_nothing_about_the_past(tmp_path: Path) -> None:
    """The plan's chronology test, on the real database.

    Build Elo on the full schedule, then build it again on a copy with every game after
    the cutoff deleted, and demand the surviving rows are identical. If any future game
    influences a rating, the two runs disagree.
    """
    if not config.DB_PATH.exists():
        pytest.skip(f"no database at {config.DB_PATH}; run `make ingest` first")

    full = connect(config.DB_PATH)
    try:
        full_rows = pregame_map(load_games(full), load_classifications(full), PARAMS)
    finally:
        full.close()

    truncated_path = tmp_path / "truncated.sqlite"
    shutil.copy(config.DB_PATH, truncated_path)
    truncated = connect(truncated_path)
    try:
        for table in ("lines", "game_team_stats"):
            truncated.execute(
                f"DELETE FROM {table} WHERE game_id IN "
                "(SELECT game_id FROM games WHERE start_date >= ?)",
                (CUTOFF,),
            )
        truncated.execute("DROP TABLE IF EXISTS vegas_benchmark")
        truncated.execute("DROP TABLE IF EXISTS elo_pregame")
        deleted = truncated.execute("DELETE FROM games WHERE start_date >= ?", (CUTOFF,)).rowcount
        truncated.commit()
        truncated_rows = pregame_map(load_games(truncated), load_classifications(truncated), PARAMS)
    finally:
        truncated.close()

    assert deleted > 3000, "the cutoff deleted too little to prove anything"
    assert len(truncated_rows) > 3000, "the cutoff kept too little to prove anything"
    differing = [
        game_id
        for game_id, row in truncated_rows.items()
        if (row.home_elo_pre, row.away_elo_pre)
        != (full_rows[game_id].home_elo_pre, full_rows[game_id].away_elo_pre)
    ]
    assert not differing, (
        f"{len(differing)} games are rated differently once later games exist, "
        f"e.g. {differing[:5]} — a future game is reaching an earlier rating"
    )


def test_the_truncation_check_fires_on_a_walk_that_reads_the_future() -> None:
    """Poisoned input. Without this, the test above could be passing vacuously."""
    games, classification = synthetic_league()
    kept = [game for game in games if game.start_date < "2014-10-01T00:00:00+00:00"]

    honest_full = pregame_map(games, classification, PARAMS)
    honest_truncated = pregame_map(kept, classification, PARAMS)
    assert all(
        honest_truncated[game_id].home_elo_pre == honest_full[game_id].home_elo_pre
        for game_id in honest_truncated
    ), "the honest walk must survive its own check first"

    leaky_full = leaky_walk(games, classification, PARAMS, mode="future")
    leaky_truncated = leaky_walk(kept, classification, PARAMS, mode="future")
    differing = [
        game_id
        for game_id in leaky_truncated
        if leaky_truncated[game_id].home_elo_pre != leaky_full[game_id].home_elo_pre
    ]
    assert differing, "the future-reading walk went undetected; the check cannot fail"


# --- 2. a game's own result must not reach its own rating ----------------------


def test_the_first_game_carries_the_initial_ratings() -> None:
    """Catches snapshot-after-update, which truncation cannot see.

    Nothing has happened yet when the first game is rated, so both ratings must be exactly
    the initial value whatever the score was.
    """
    games, classification = synthetic_league()
    rows = pregame_map(games, classification, PARAMS)
    first = rows[1]
    assert (first.home_elo_pre, first.away_elo_pre) == (PARAMS.initial, PARAMS.initial)


def test_the_first_game_check_fires_when_the_update_comes_first() -> None:
    """Poisoned input for the check above."""
    games, classification = synthetic_league()
    rows = leaky_walk(games, classification, PARAMS, mode="own_result")
    first = rows[1]
    assert (first.home_elo_pre, first.away_elo_pre) != (PARAMS.initial, PARAMS.initial), (
        "a walk that updates before snapshotting went undetected; the check cannot fail"
    )


@pytest.mark.integration
def test_no_stored_rating_anticipates_its_own_result(built_db: sqlite3.Connection) -> None:
    """The same guarantee, read back off the built table rather than from memory.

    A team's stored pre-game rating for its opening game of the data must be the initial
    rating, not something already nudged by that game's score. Read against the parameters
    the table was actually built with, not the defaults.
    """
    params, _ = load_params()
    exists = built_db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='elo_pregame'"
    ).fetchone()
    if not exists:
        pytest.skip("elo_pregame not built; run `make elo`")

    row = built_db.execute(
        """
        SELECT e.home_elo_pre, e.away_elo_pre
        FROM games g
        JOIN elo_pregame e ON e.game_id = g.game_id
        JOIN team_seasons th ON th.team_id = g.home_team_id AND th.season = g.season
        JOIN team_seasons ta ON ta.team_id = g.away_team_id AND ta.season = g.season
        WHERE g.season = ? AND th.classification = 'fbs' AND ta.classification = 'fbs'
        ORDER BY g.start_date, g.game_id
        LIMIT 1
        """,
        (config.FIRST_SEASON,),
    ).fetchone()
    assert row is not None
    assert (row["home_elo_pre"], row["away_elo_pre"]) == (params.initial, params.initial)
