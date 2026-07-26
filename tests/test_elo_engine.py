"""Unit tests for the pure Elo arithmetic.

Every expected value here is written as the formula that produces it, evaluated with
``math`` in the test rather than copied from a run of the code under test. A test that
asserts ``engine.f(x) == engine.f(x)`` is not a test.
"""

from __future__ import annotations

import math

import pytest

from cfb.elo.engine import (
    EloParams,
    brier_score,
    expected,
    log_loss,
    mov_multiplier,
    run_season_regression,
    update,
)

PARAMS = EloParams()


# --- expected score -----------------------------------------------------------


def test_expected_is_symmetric() -> None:
    """E_home + E_away = 1, with home-field advantage flipping sign for the away side."""
    for rating, opponent, hfa in ((1500, 1500, 0), (1800, 1200, 65), (1350, 1720, 65)):
        assert expected(rating, opponent, hfa) + expected(opponent, rating, -hfa) == pytest.approx(
            1.0, abs=1e-12
        )


def test_equal_ratings_with_no_home_field_is_a_coin_flip() -> None:
    assert expected(1500, 1500, 0.0) == 0.5


def test_home_field_advantage_favours_the_home_team() -> None:
    """Sign check. An inverted HFA would still produce plausible-looking ratings."""
    assert expected(1500, 1500, PARAMS.hfa) == pytest.approx(
        1.0 / (1.0 + 10.0 ** (-65.0 / 400.0)), abs=1e-12
    )
    assert expected(1500, 1500, PARAMS.hfa) > 0.5


def test_four_hundred_points_is_ten_to_one() -> None:
    """The definition of the Elo scale, stated as a test so a changed SCALE is loud."""
    assert expected(1900, 1500) == pytest.approx(10.0 / 11.0, abs=1e-12)


# --- margin-of-victory multiplier ---------------------------------------------


@pytest.mark.parametrize(
    ("margin", "winner_elo_diff"),
    [(14, 65), (21, -300), (1, 0), (35, 500), (35, -500)],
)
def test_mov_multiplier_matches_the_formula(margin: int, winner_elo_diff: float) -> None:
    assert mov_multiplier(margin, winner_elo_diff) == pytest.approx(
        math.log(abs(margin) + 1) * 2.2 / (0.001 * winner_elo_diff + 2.2), abs=1e-12
    )


def test_multiplier_damps_expected_blowouts_and_amplifies_upsets() -> None:
    """The whole point of the signed denominator, and the reason it is not an absolute value.

    Same 35-point margin: worth less when the favourite delivered it, more when the
    underdog did. Under the plan's literal ``|elo_diff|`` spelling these two would be
    equal, and the anti-autocorrelation property the plan asks for would be gone.
    """
    favourite = mov_multiplier(35, 500)
    underdog = mov_multiplier(35, -500)
    even = mov_multiplier(35, 0)
    assert favourite < even < underdog


def test_multiplier_is_sublinear_in_margin() -> None:
    """A 42-point win counts for more than a 3-point win, nowhere near fourteen times more."""
    small = mov_multiplier(3, 0)
    large = mov_multiplier(42, 0)
    assert large > small
    assert large < 3 * small


def test_a_tie_moves_nothing() -> None:
    """ln(0 + 1) = 0. Documented behaviour, unreachable in this dataset (0 ties in 10,373).

    Pinned rather than patched with an invented minimum margin: inventing one would be
    fabricating data to fill a gap nobody has.
    """
    assert mov_multiplier(0, 0.0) == 0.0
    assert update(1800.0, 1200.0, 21, 21, PARAMS) == (1800.0, 1200.0)


def test_multiplier_refuses_a_negative_denominator() -> None:
    """Guard against a silent sign flip if ratings ever diverge past 2200 points."""
    with pytest.raises(ValueError, match="diverged"):
        mov_multiplier(10, -2300)


# --- the update ---------------------------------------------------------------


def test_update_is_zero_sum() -> None:
    home_post, away_post = update(1600.0, 1450.0, 31, 17, PARAMS)
    assert (home_post - 1600.0) == pytest.approx(-(away_post - 1450.0), abs=1e-12)


def test_winning_raises_a_rating_and_losing_lowers_it() -> None:
    home_post, away_post = update(1500.0, 1500.0, 28, 14, PARAMS)
    assert home_post > 1500.0 > away_post


def test_beating_a_much_weaker_team_moves_less_than_the_reverse() -> None:
    """A favourite's expected win is nearly priced in; an upset is not."""
    favourite_gain = update(1900.0, 1300.0, 42, 0, PARAMS)[0] - 1900.0
    underdog_gain = update(1300.0, 1900.0, 42, 0, PARAMS)[0] - 1300.0
    assert underdog_gain > favourite_gain > 0


def test_neutral_site_removes_home_field_advantage() -> None:
    """The plan's neutral-site test. HFA must be absent from E, not merely reduced."""
    neutral = update(1500.0, 1500.0, 24, 21, PARAMS, neutral_site=True)
    hosted = update(1500.0, 1500.0, 24, 21, PARAMS, neutral_site=False)
    assert neutral[0] > hosted[0], "an unfavoured home win should be worth less than a neutral one"

    # With equal ratings and no home-field advantage, E is exactly 0.5.
    gain = neutral[0] - 1500.0
    assert gain == pytest.approx(PARAMS.k * mov_multiplier(3, 0.0) * 0.5, abs=1e-12)


def test_a_bigger_k_moves_ratings_proportionally_further() -> None:
    slow = update(1500.0, 1500.0, 28, 14, EloParams(k=20.0))[0] - 1500.0
    fast = update(1500.0, 1500.0, 28, 14, EloParams(k=40.0))[0] - 1500.0
    assert fast == pytest.approx(2 * slow, abs=1e-12)


# --- pre-season regression ----------------------------------------------------


def test_season_regression_is_exact() -> None:
    assert run_season_regression(1800.0, EloParams(regression=0.5)) == 1650.0
    assert run_season_regression(1200.0, EloParams(regression=0.5)) == 1350.0
    assert run_season_regression(1800.0, PARAMS) == pytest.approx(1700.0, abs=1e-9)


def test_season_regression_leaves_the_mean_alone_and_never_overshoots() -> None:
    assert run_season_regression(1500.0, PARAMS) == 1500.0
    for coefficient in (0.25, 1 / 3, 0.5):
        regressed = run_season_regression(2000.0, EloParams(regression=coefficient))
        assert 1500.0 < regressed < 2000.0


# --- scoring ------------------------------------------------------------------


def test_brier_and_log_loss_on_known_values() -> None:
    assert brier_score([0.5, 0.5], [1.0, 0.0]) == 0.25
    assert brier_score([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert log_loss([0.5, 0.5], [1.0, 0.0]) == pytest.approx(math.log(2), abs=1e-12)


def test_log_loss_survives_a_confident_miss() -> None:
    """Clipped, so one bad prediction costs a large finite number rather than infinity."""
    value = log_loss([0.0], [1.0])
    assert math.isfinite(value)
    assert value > 30


def test_metrics_refuse_empty_and_mismatched_input() -> None:
    """An empty sample scores perfectly. That must raise, not return 0."""
    for metric in (brier_score, log_loss):
        with pytest.raises(ValueError, match="no predictions"):
            metric([], [])
        with pytest.raises(ValueError, match="length mismatch"):
            metric([0.5], [1.0, 0.0])


# --- parameters ---------------------------------------------------------------


def test_params_round_trip_and_ignore_metadata() -> None:
    """``elo_params.json`` carries provenance alongside the values; loading must not choke."""
    params = EloParams(k=27.5, hfa=55.0, regression=0.25)
    assert EloParams.from_dict(params.to_dict()) == params
    assert EloParams.from_dict({"k": 27.5, "fitted_on": "2014-2021"}).k == 27.5
