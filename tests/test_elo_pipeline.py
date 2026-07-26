"""The rules layered on top of the pure engine: subdivision policy, season boundaries,
unplayed games, and the ordering the walk depends on.

The chronology guarantee itself is tested separately in ``test_elo_chronology.py``.
"""

from __future__ import annotations

import random
import sqlite3

import pytest

from cfb.elo.engine import EloParams, expected
from cfb.elo.pipeline import (
    Game,
    load_classifications,
    load_games,
    run_elo,
    score,
    scoreable,
    write_elo_pregame,
)
from tests.conftest import add_game, add_team_season

PARAMS = EloParams()

ALPHA, BRAVO, CHARLIE, FCS_TEAM = 1, 2, 3, 4


def game(
    game_id: int,
    season: int,
    start_date: str,
    home: int,
    away: int,
    home_points: int | None = 21,
    away_points: int | None = 14,
    neutral_site: bool = False,
    completed: bool = True,
) -> Game:
    """Build a synthetic game with the fields the walk reads."""
    return Game(
        game_id=game_id,
        season=season,
        start_date=start_date,
        neutral_site=neutral_site,
        home_team_id=home,
        away_team_id=away,
        home_points=home_points,
        away_points=away_points,
        completed=completed,
    )


def fbs_everywhere(*team_ids: int, seasons: tuple[int, ...] = (2014, 2015)) -> dict:
    """Classification map marking the given teams FBS in the given seasons."""
    return {(team_id, season): "fbs" for team_id in team_ids for season in seasons}


# --- subdivision policy -------------------------------------------------------


def test_fcs_opponent_is_a_fixed_rating_that_never_moves() -> None:
    """The confirmed policy: 1200, constant, and never tracked."""
    classification = fbs_everywhere(ALPHA) | {(FCS_TEAM, 2014): "fcs", (FCS_TEAM, 2015): "fcs"}
    run = run_elo(
        [
            game(1, 2014, "2014-09-06T18:00:00+00:00", ALPHA, FCS_TEAM, 49, 0),
            game(2, 2014, "2014-09-13T18:00:00+00:00", FCS_TEAM, ALPHA, 3, 45),
        ],
        classification,
        PARAMS,
    )
    assert run.rows[0].away_elo_pre == PARAMS.fcs
    assert run.rows[1].home_elo_pre == PARAMS.fcs, "the FCS rating moved after a 49-0 loss"
    assert FCS_TEAM not in run.ratings


def test_losing_to_an_fcs_team_hurts() -> None:
    """The reason the games are kept rather than excluded."""
    classification = fbs_everywhere(ALPHA) | {(FCS_TEAM, 2014): "fcs"}
    run = run_elo(
        [game(1, 2014, "2014-09-06T18:00:00+00:00", ALPHA, FCS_TEAM, 17, 24)],
        classification,
        PARAMS,
    )
    assert run.ratings[ALPHA] < PARAMS.initial


def test_incumbents_start_at_1500_and_later_arrivals_at_1300() -> None:
    classification = {
        (ALPHA, 2014): "fbs",
        (ALPHA, 2015): "fbs",
        (BRAVO, 2014): "fcs",
        (BRAVO, 2015): "fbs",
    }
    run = run_elo(
        [
            game(1, 2014, "2014-09-06T18:00:00+00:00", ALPHA, BRAVO),
            game(2, 2015, "2015-09-05T18:00:00+00:00", ALPHA, BRAVO),
        ],
        classification,
        PARAMS,
    )
    assert run.rows[0].home_elo_pre == PARAMS.initial
    assert run.rows[0].away_elo_pre == PARAMS.fcs, "an FCS season is an FCS season"
    assert run.rows[1].away_elo_pre == PARAMS.newcomer
    assert (2015, BRAVO, PARAMS.newcomer) in run.promotions


def test_a_team_leaving_fbs_is_dropped_and_restarts_on_return() -> None:
    """Idaho's case, plus the return branch no team takes within 2014-2025.

    Confirmed rule: a rating from before a spell in FCS is too stale to resume, so a
    returning team re-enters at the newcomer prior.
    """
    classification = {
        (ALPHA, 2014): "fbs",
        (ALPHA, 2015): "fbs",
        (ALPHA, 2016): "fbs",
        (BRAVO, 2014): "fbs",
        (BRAVO, 2015): "fcs",
        (BRAVO, 2016): "fbs",
    }
    run = run_elo(
        [
            game(1, 2014, "2014-09-06T18:00:00+00:00", ALPHA, BRAVO, 49, 0),
            game(2, 2015, "2015-09-05T18:00:00+00:00", ALPHA, BRAVO, 49, 0),
            game(3, 2016, "2016-09-03T18:00:00+00:00", ALPHA, BRAVO, 49, 0),
        ],
        classification,
        PARAMS,
    )
    assert run.rows[1].away_elo_pre == PARAMS.fcs
    assert (2015, BRAVO) in run.demotions
    assert run.rows[2].away_elo_pre == PARAMS.newcomer


# --- season boundaries --------------------------------------------------------


def test_regression_is_applied_once_per_season_boundary() -> None:
    """Applied to the rating carried out of the previous season, not to a fresh 1500."""
    classification = fbs_everywhere(ALPHA, BRAVO, seasons=(2014, 2015))
    run = run_elo(
        [
            game(1, 2014, "2014-09-06T18:00:00+00:00", ALPHA, BRAVO, 42, 7),
            game(2, 2015, "2015-09-05T18:00:00+00:00", ALPHA, BRAVO, 42, 7),
        ],
        classification,
        PARAMS,
    )
    after_2014 = run.season_end_ratings[2014][ALPHA]
    assert run.rows[1].home_elo_pre == pytest.approx(
        after_2014 + (PARAMS.mean - after_2014) * PARAMS.regression, abs=1e-12
    )


def test_a_bowl_game_belongs_to_its_own_season() -> None:
    """Season S's postseason kicks off in January of S+1, before season S+1 regresses."""
    classification = fbs_everywhere(ALPHA, BRAVO, seasons=(2014, 2015))
    run = run_elo(
        [
            game(1, 2014, "2014-09-06T18:00:00+00:00", ALPHA, BRAVO, 42, 7),
            game(2, 2014, "2015-01-01T18:00:00+00:00", ALPHA, BRAVO, 42, 7),
            game(3, 2015, "2015-09-05T18:00:00+00:00", ALPHA, BRAVO, 42, 7),
        ],
        classification,
        PARAMS,
    )
    rating_after_bowl = run.season_end_ratings[2014][ALPHA]
    assert run.rows[1].home_elo_pre != PARAMS.initial, "the bowl was regressed as a new season"
    assert run.rows[2].home_elo_pre == pytest.approx(
        rating_after_bowl + (PARAMS.mean - rating_after_bowl) * PARAMS.regression, abs=1e-12
    )


# --- the games that are not games ---------------------------------------------


def test_a_cancelled_game_gets_ratings_but_moves_nothing() -> None:
    """RISKS #12. The matchup was scheduled, so Phase 4 may want the ratings; it was never
    played, so it cannot inform them."""
    classification = fbs_everywhere(ALPHA, BRAVO)
    run = run_elo(
        [
            game(
                1,
                2014,
                "2014-09-06T18:00:00+00:00",
                ALPHA,
                BRAVO,
                home_points=None,
                away_points=None,
                completed=False,
            ),
            game(2, 2014, "2014-09-13T18:00:00+00:00", ALPHA, BRAVO, 21, 14),
        ],
        classification,
        PARAMS,
    )
    assert len(run.rows) == 2, "a cancelled game still needs pre-game ratings"
    assert run.rows[1].home_elo_pre == PARAMS.initial, "an unplayed game moved a rating"


def test_neutral_site_probability_drops_the_home_edge() -> None:
    """The plan's neutral-site requirement, at the level the report and tuner read."""
    from cfb.elo.pipeline import elo_probability

    classification = fbs_everywhere(ALPHA, BRAVO)
    run = run_elo(
        [
            game(1, 2014, "2014-09-06T18:00:00+00:00", ALPHA, BRAVO, neutral_site=True),
            game(2, 2014, "2014-09-06T18:00:00+00:00", BRAVO, ALPHA, neutral_site=False),
        ],
        classification,
        PARAMS,
    )
    assert elo_probability(run.rows[0], PARAMS) == 0.5
    assert elo_probability(run.rows[1], PARAMS) == pytest.approx(
        expected(run.rows[1].home_elo_pre, run.rows[1].away_elo_pre, PARAMS.hfa), abs=1e-12
    )


# --- ordering -----------------------------------------------------------------


def test_the_walk_sorts_its_own_input() -> None:
    """A caller that happens to pass rows in order must not be what makes this correct."""
    classification = fbs_everywhere(ALPHA, BRAVO, CHARLIE, seasons=(2014, 2015))
    games = [
        game(1, 2014, "2014-09-06T18:00:00+00:00", ALPHA, BRAVO, 42, 7),
        game(2, 2014, "2014-09-13T18:00:00+00:00", BRAVO, CHARLIE, 21, 20),
        game(3, 2014, "2014-11-01T18:00:00+00:00", CHARLIE, ALPHA, 3, 45),
        game(4, 2015, "2015-09-05T18:00:00+00:00", ALPHA, CHARLIE, 17, 24),
    ]
    ordered = {row.game_id: row for row in run_elo(games, classification, PARAMS).rows}

    shuffled = list(games)
    random.Random(0).shuffle(shuffled)
    reshuffled = {row.game_id: row for row in run_elo(shuffled, classification, PARAMS).rows}
    assert ordered == reshuffled


def test_simultaneous_games_cannot_affect_each_other() -> None:
    """Why the ``game_id`` tie-break is safe rather than merely deterministic.

    4,082 distinct kickoff timestamps cover 10,374 games, so ties are everywhere. They are
    harmless because no team plays two games at once: simultaneous games involve disjoint
    teams, and an update to one cannot reach the other's pre-game snapshot.
    """
    classification = fbs_everywhere(ALPHA, BRAVO, CHARLIE, 5)
    same_time = "2014-09-06T18:00:00+00:00"
    forward = run_elo(
        [
            game(1, 2014, same_time, ALPHA, BRAVO, 42, 7),
            game(2, 2014, same_time, CHARLIE, 5, 42, 7),
        ],
        classification,
        PARAMS,
    )
    backward = run_elo(
        [
            game(2, 2014, same_time, CHARLIE, 5, 42, 7),
            game(1, 2014, same_time, ALPHA, BRAVO, 42, 7),
        ],
        classification,
        PARAMS,
    )
    assert {row.game_id: row for row in forward.rows} == {row.game_id: row for row in backward.rows}


# --- database edges -----------------------------------------------------------


def test_load_games_rejects_a_non_utc_kickoff(toy_db: sqlite3.Connection) -> None:
    """Ordering is lexicographic on ``start_date``; a mixed offset would reorder the walk."""
    add_game(toy_db, game_id=1, start_date="2023-08-26T18:30:00+00:00")
    add_game(toy_db, game_id=2, start_date="2023-08-26T14:30:00-04:00")
    with pytest.raises(ValueError, match="non-UTC"):
        load_games(toy_db)


def test_write_elo_pregame_leaves_no_stale_rows(toy_db: sqlite3.Connection) -> None:
    """A rebuild under new parameters must not sit next to rows from the old ones."""
    add_team_season(toy_db, ALPHA, 2014)
    add_team_season(toy_db, BRAVO, 2014)
    for game_id in (1, 2):
        add_game(
            toy_db,
            game_id=game_id,
            season=2014,
            start_date=f"2014-09-0{game_id}T18:00:00+00:00",
            home_team_id=ALPHA,
            away_team_id=BRAVO,
        )
    run = run_elo(load_games(toy_db), load_classifications(toy_db), PARAMS)
    assert write_elo_pregame(toy_db, run.rows) == 2
    assert write_elo_pregame(toy_db, run.rows[:1]) == 1
    assert toy_db.execute("SELECT COUNT(*) FROM elo_pregame").fetchone()[0] == 1


def test_load_classifications_reads_the_season_not_the_team(toy_db: sqlite3.Connection) -> None:
    add_team_season(toy_db, ALPHA, 2014, "fcs")
    add_team_season(toy_db, ALPHA, 2015, "fbs")
    classification = load_classifications(toy_db)
    assert classification[(ALPHA, 2014)] == "fcs"
    assert classification[(ALPHA, 2015)] == "fbs"


# --- scoring filters ----------------------------------------------------------


def test_scoreable_drops_fcs_games_and_unplayed_games() -> None:
    classification = fbs_everywhere(ALPHA, BRAVO) | {(FCS_TEAM, 2014): "fcs"}
    run = run_elo(
        [
            game(1, 2014, "2014-09-06T18:00:00+00:00", ALPHA, BRAVO),
            game(2, 2014, "2014-09-13T18:00:00+00:00", ALPHA, FCS_TEAM),
            game(
                3,
                2014,
                "2014-09-20T18:00:00+00:00",
                ALPHA,
                BRAVO,
                home_points=None,
                away_points=None,
                completed=False,
            ),
        ],
        classification,
        PARAMS,
    )
    assert [row.game_id for row in scoreable(run.rows)] == [1]
    assert [row.game_id for row in scoreable(run.rows, both_fbs=False)] == [1, 2]
    assert scoreable(run.rows, seasons=[2015]) == []


def test_score_refuses_an_empty_sample() -> None:
    """Zero games score perfectly. That has to raise rather than quietly return a number."""
    with pytest.raises(ValueError, match="no scoreable games"):
        score([], PARAMS)
