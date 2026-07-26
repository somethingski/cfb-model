"""The canonical leakage bug, attacked directly.

A rolling statistic must average the games *before* the one being predicted. Drop the shift
by one and every rolling column quietly contains the result it is supposed to predict —
nothing raises, no metric looks strange, and the model gets much better overnight.

``src/cfb/features/build.py`` spells the shift once, in :func:`priors_before`, and asserts
it a second time inside :func:`team_features`. This file attacks both, on a three-game
season small enough to check by eye, and then proves the attack works by running it against
a selector with the shift removed.
"""

from __future__ import annotations

import pytest

from cfb.features.build import (
    MIN_PRIOR_GAMES,
    ScheduledGame,
    TeamGame,
    priors_before,
    team_features,
)

TEAM = 1
OPPONENT = 2


def game(day: int, points_for: int, points_against: int, yards: int, plays: int) -> TeamGame:
    """One completed game in the toy season.

    Args:
        day: Day of September 2014 it kicked off.
        points_for: Points the team scored.
        points_against: Points it allowed.
        yards: Yards it gained.
        plays: Plays it ran.

    Returns:
        The team's side of that game. The opponent is given 300 yards on 60 plays every
        time, so the defensive rate is a round 5.0 and a wrong window is obvious.
    """
    return TeamGame(
        game_id=day,
        season=2014,
        start_date=f"2014-09-{day:02d}T18:00:00+00:00",
        team_id=TEAM,
        opponent_id=OPPONENT,
        opponent_is_fcs=False,
        points_for=points_for,
        points_against=points_against,
        yards_for=float(yards),
        plays_for=float(plays),
        yards_against=300.0,
        plays_against=60.0,
    )


SEASON = [
    game(6, 20, 10, 350, 70),
    game(13, 30, 24, 450, 75),
    game(20, 40, 17, 400, 80),
]
"""Three games, one week apart. Games 1 and 2 are the window for game 3."""


def target(day: int) -> ScheduledGame:
    """The game being predicted, on the given day of September 2014."""
    return ScheduledGame(
        game_id=100 + day,
        season=2014,
        week=day // 7 + 1,
        season_type="regular",
        start_date=f"2014-09-{day:02d}T18:00:00+00:00",
        neutral_site=False,
        conference_game=True,
        home_team_id=TEAM,
        away_team_id=OPPONENT,
        home_points=7,
        away_points=3,
        completed=True,
    )


# --- the shift, from the front ------------------------------------------------


def test_the_first_game_of_a_season_has_null_rolling_stats() -> None:
    """Nothing has happened yet, so there is nothing to average.

    The nulls are the correct answer, not a gap to be filled. Back-filling them from later
    games is the same bug wearing a helpful expression.
    """
    features = team_features(target(6), priors_before(SEASON, "2014-09-06T18:00:00+00:00"))
    assert features.prior_games == 0
    assert features.off_ppg_roll is None
    assert features.def_ppg_roll is None
    assert features.off_ypp_roll is None
    assert features.def_ypp_roll is None
    assert features.pace_roll is None
    assert features.prev_season_win_pct is None


def test_the_third_game_averages_exactly_the_first_two() -> None:
    """The plan's test, hand-computed.

    Points: (20 + 30) / 2 = 25. Allowed: (10 + 24) / 2 = 17.
    Yards per play: (350 + 450) / (70 + 75) = 800 / 145.
    Pace: (70 + 75) / 2 = 72.5. Defensive rate: 600 / 120 = 5.0.
    """
    features = team_features(target(20), priors_before(SEASON, "2014-09-20T18:00:00+00:00"))
    assert features.prior_games == 2
    assert features.off_ppg_roll == pytest.approx(25.0)
    assert features.def_ppg_roll == pytest.approx(17.0)
    assert features.off_ypp_roll == pytest.approx(800 / 145)
    assert features.def_ypp_roll == pytest.approx(5.0)
    assert features.pace_roll == pytest.approx(72.5)
    assert features.as_of == "2014-09-13T18:00:00+00:00"


def test_a_game_at_the_same_instant_is_not_prior() -> None:
    """The comparison is strict.

    Two games kicking off together are not ordered by the clock, so neither may inform the
    other. The window for game 2 is game 1 alone, never game 2 itself.
    """
    simultaneous = priors_before(SEASON, SEASON[1].start_date)
    assert [g.game_id for g in simultaneous] == [SEASON[0].game_id]


def test_the_minimum_window_is_one_prior_game() -> None:
    """A single prior game is enough; zero is not. Documented, so it is pinned."""
    one = team_features(target(13), priors_before(SEASON, "2014-09-13T18:00:00+00:00"))
    assert one.prior_games == MIN_PRIOR_GAMES
    assert one.off_ppg_roll == pytest.approx(20.0)


# --- proving the checks above can fail ----------------------------------------


def priors_including_this_game(team_games: list[TeamGame], kickoff: str) -> list[TeamGame]:
    """A selector with the shift dropped. Lives here and never in ``src/``.

    Args:
        team_games: The team's games.
        kickoff: The target kickoff.

    Returns:
        Games up to *and including* the kickoff — the bug.
    """
    return sorted((g for g in team_games if g.start_date <= kickoff), key=lambda g: g.start_date)


def test_dropping_the_shift_changes_the_answer() -> None:
    """Without this, the assertions above could be true of a broken implementation too."""
    honest = priors_before(SEASON, "2014-09-20T18:00:00+00:00")
    leaky = priors_including_this_game(SEASON, "2014-09-20T18:00:00+00:00")
    assert len(leaky) == len(honest) + 1, "the poisoned selector is not actually leaking"

    # Mean of 20, 30, 40 is 30 — the third game's own score has moved the window.
    assert sum(g.points_for for g in leaky) / len(leaky) == pytest.approx(30.0)
    assert sum(g.points_for for g in honest) / len(honest) == pytest.approx(25.0)


def test_team_features_refuses_a_window_that_reaches_its_own_game() -> None:
    """The second line of defence, proved live.

    ``priors_before`` is the shift; this assertion inside ``team_features`` is what stops a
    future caller from building the window some other way and getting away with it.
    """
    leaky = priors_including_this_game(SEASON, "2014-09-20T18:00:00+00:00")
    with pytest.raises(ValueError, match="the shift has been dropped"):
        team_features(target(20), leaky)
