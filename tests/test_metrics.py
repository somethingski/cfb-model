"""Phase 6's scoring vocabulary, checked against arithmetic done outside the code.

Every expected value in the first section was worked out by hand and written down here as a
literal. That is the point: a test that computed its expectation with the same function it
is testing would pass whatever the function did. The four-game set is small enough that the
arithmetic is in the docstring and a reader can check it without running anything.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cfb.eval import metrics

# --- The four-game toy set ----------------------------------------------------
#
#   game  outcome  predicted
#   1     1        0.9
#   2     0        0.2
#   3     1        0.6
#   4     0        0.5
#
# Brier  = ((0.9-1)^2 + (0.2-0)^2 + (0.6-1)^2 + (0.5-0)^2) / 4
#        = (0.01 + 0.04 + 0.16 + 0.25) / 4 = 0.46 / 4 = 0.115
#
# Log loss = -(ln 0.9 + ln 0.8 + ln 0.6 + ln 0.5) / 4
#          = -(-0.10536052 - 0.22314355 - 0.51082562 - 0.69314718) / 4
#          =   1.53247687 / 4 = 0.38311922

TOY_OUTCOMES = np.array([1.0, 0.0, 1.0, 0.0])
TOY_PREDICTIONS = np.array([0.9, 0.2, 0.6, 0.5])
TOY_BRIER = 0.115
TOY_LOG_LOSS = 0.38311921782449324


def test_brier_matches_the_hand_computed_value():
    assert metrics.brier_score(TOY_OUTCOMES, TOY_PREDICTIONS) == pytest.approx(TOY_BRIER, abs=1e-12)


def test_log_loss_matches_the_hand_computed_value():
    assert metrics.log_loss_score(TOY_OUTCOMES, TOY_PREDICTIONS) == pytest.approx(
        TOY_LOG_LOSS, abs=1e-12
    )


def test_the_hand_computed_log_loss_is_what_the_arithmetic_says():
    """Guard the literal above against a typo, without using the code under test.

    Spelled out with ``math.log`` term by term, so if someone mistypes a digit in
    ``TOY_LOG_LOSS`` this fails rather than the expectation silently drifting to match a
    broken implementation.
    """
    by_hand = -(math.log(0.9) + math.log(0.8) + math.log(0.6) + math.log(0.5)) / 4
    assert by_hand == pytest.approx(TOY_LOG_LOSS, abs=1e-15)


def test_a_perfect_forecast_scores_zero_on_both():
    outcomes = np.array([1.0, 0.0])
    assert metrics.brier_score(outcomes, np.array([1.0, 0.0])) == 0.0
    assert metrics.log_loss_score(outcomes, np.array([1.0, 0.0])) == pytest.approx(0.0, abs=1e-13)


def test_a_confident_miss_is_finite_but_large():
    """Log loss clips, so one certain wrong call costs a lot and not infinity."""
    loss = metrics.log_loss_score(np.array([1.0]), np.array([0.0]))
    assert math.isfinite(loss)
    assert loss > 30


# --- Reliability binning ------------------------------------------------------


def test_bins_have_equal_counts():
    outcomes = np.array([float(index % 2) for index in range(100)])
    predictions = np.linspace(0.01, 0.99, 100)
    bins = metrics.reliability_table(outcomes, predictions, n_bins=10)
    assert len(bins) == 10
    assert [current.n for current in bins] == [10] * 10
    assert sum(current.n for current in bins) == 100


def test_bins_stay_equal_when_every_prediction_ties():
    """The reason bins are cut on the sorted order rather than on quantile values.

    Ten identical predictions have identical quantiles, so quantile edges would collapse
    into one bin holding everything and nine empty ones. Splitting the order cannot do that.
    """
    outcomes = np.array([1.0, 0.0] * 10)
    predictions = np.full(20, 0.42)
    bins = metrics.reliability_table(outcomes, predictions, n_bins=5)
    assert [current.n for current in bins] == [4] * 5


def test_bins_are_ascending_and_report_the_right_frequency():
    outcomes = np.array([0.0, 0.0, 1.0, 1.0])
    predictions = np.array([0.1, 0.2, 0.8, 0.9])
    bins = metrics.reliability_table(outcomes, predictions, n_bins=2)
    assert [current.n for current in bins] == [2, 2]
    assert bins[0].mean_predicted == pytest.approx(0.15)
    assert bins[0].empirical == 0.0
    assert bins[1].mean_predicted == pytest.approx(0.85)
    assert bins[1].empirical == 1.0
    assert bins[0].hi <= bins[1].lo


def test_a_bin_reports_its_gap_signed_toward_underconfidence():
    outcomes = np.array([1.0, 1.0, 1.0, 0.0])
    predictions = np.array([0.5, 0.5, 0.5, 0.5])
    (only_bin,) = metrics.reliability_table(outcomes, predictions, n_bins=1)
    assert only_bin.gap == pytest.approx(0.25), "won more often than predicted -> positive"


def test_more_bins_than_games_returns_one_bin_per_game_not_empty_bins():
    outcomes = np.array([1.0, 0.0])
    predictions = np.array([0.7, 0.3])
    bins = metrics.reliability_table(outcomes, predictions, n_bins=10)
    assert [current.n for current in bins] == [1, 1]


@pytest.mark.parametrize(
    "outcomes, predictions, n_bins",
    [
        (np.array([1.0, 0.0]), np.array([0.5]), 10),
        (np.array([]), np.array([]), 10),
        (np.array([1.0]), np.array([0.5]), 0),
    ],
)
def test_binning_refuses_inputs_it_cannot_bin(outcomes, predictions, n_bins):
    with pytest.raises(ValueError):
        metrics.reliability_table(outcomes, predictions, n_bins=n_bins)


# --- Resolution ---------------------------------------------------------------


def test_resolution_counts_distinct_values_and_the_largest_plateau():
    described = metrics.resolution(np.array([0.5, 0.5, 0.5, 0.2, 0.9]))
    assert described == {
        "n": 5,
        "n_distinct": 3,
        "largest_plateau": 3,
        "largest_plateau_value": 0.5,
    }


def test_resolution_of_a_continuous_forecast_is_its_own_length():
    described = metrics.resolution(np.linspace(0.1, 0.9, 50))
    assert described["n_distinct"] == 50
    assert described["largest_plateau"] == 1


# --- Skill fraction -----------------------------------------------------------


def test_skill_fraction_matches_the_hand_computed_value():
    """naive 0.25, model 0.20, line 0.18 -> (0.25-0.20)/(0.25-0.18) = 0.05/0.07."""
    assert metrics.skill_fraction(0.25, 0.20, 0.18) == pytest.approx(0.05 / 0.07)


def test_a_model_equal_to_the_line_closes_the_whole_distance():
    assert metrics.skill_fraction(0.25, 0.18, 0.18) == pytest.approx(1.0)


def test_a_model_equal_to_the_baseline_closes_none_of_it():
    assert metrics.skill_fraction(0.25, 0.25, 0.18) == pytest.approx(0.0)


def test_beating_the_line_pushes_the_fraction_past_one():
    """Not an error in itself — the tripwire is what treats it as an alarm."""
    assert metrics.skill_fraction(0.25, 0.17, 0.18) > 1.0


def test_skill_fraction_refuses_a_baseline_that_is_not_worse_than_the_line():
    with pytest.raises(ValueError, match="not worse than"):
        metrics.skill_fraction(0.18, 0.19, 0.18)
