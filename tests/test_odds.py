"""Unit and property tests for the odds arithmetic.

This is the module ``RISKS.md`` #8 is about: a sign error or a wrong vig removal here
produces a benchmark that is silently wrong while every other test in the project stays
green. So the expected values below are written out longhand rather than computed by
calling the function under test.
"""

from __future__ import annotations

import math

import pytest

from cfb.vegas.odds import (
    american_to_implied,
    devig_multiplicative,
    moneyline_to_prob,
    normal_cdf,
    spread_to_prob,
)

SIGMA = 16.0
"""Representative sigma for the pure tests; the fitted value is asserted separately."""


class TestAmericanToImplied:
    @pytest.mark.parametrize(
        ("odds", "expected"),
        [
            (-110, 110 / 210),  # standard juice
            (-100, 0.5),  # even money, negative spelling
            (100, 0.5),  # even money, positive spelling
            (-200, 2 / 3),
            (170, 100 / 270),
            (250, 2 / 7),
            (-2000, 2000 / 2100),
        ],
    )
    def test_known_pairs(self, odds: int, expected: float) -> None:
        assert american_to_implied(odds) == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize("odds", [0, 99, -99, 1, -1])
    def test_rejects_impossible_quotes(self, odds: int) -> None:
        """Odds inside (-100, 100) are not a quote the formula describes."""
        with pytest.raises(ValueError, match="magnitude"):
            american_to_implied(odds)

    def test_includes_vig(self) -> None:
        """A -110/-110 market implies 104.8%, not 100%. If this is 1.0, the vig is gone."""
        total = american_to_implied(-110) + american_to_implied(-110)
        assert total == pytest.approx(1.047619, abs=1e-6)


class TestDevigMultiplicative:
    def test_balanced_market_is_a_coin_flip(self) -> None:
        assert devig_multiplicative(*[american_to_implied(-110)] * 2) == (0.5, 0.5)

    def test_hand_checked_asymmetric_market(self) -> None:
        """-200 / +170, worked by hand: (2/3) / (2/3 + 10/27) = 9/14."""
        p_home, p_away = devig_multiplicative(american_to_implied(-200), american_to_implied(170))
        assert p_home == pytest.approx(9 / 14, abs=1e-12)
        assert p_away == pytest.approx(5 / 14, abs=1e-12)

    @pytest.mark.parametrize(("home", "away"), [(0.0, 0.5), (1.0, 0.5), (0.5, -0.1), (0.5, 1.2)])
    def test_rejects_non_probabilities(self, home: float, away: float) -> None:
        with pytest.raises(ValueError, match=r"must be in \(0, 1\)"):
            devig_multiplicative(home, away)

    def test_probabilities_sum_to_exactly_one(self) -> None:
        """Exit criterion 3 asks for an exact sum, not an approximate one.

        Sweeps every combination of a wide grid of real-world quotes. Two independent
        divisions would fail this on some pairs at the last bit; computing the away side
        as ``1 - p_home`` is what makes it exact.
        """
        negative = [-100000, -5000, -2000, -450, -200, -115, -110, -105, -100]
        quotes = negative + [-odds for odds in negative]
        for home in quotes:
            for away in quotes:
                p_home, p_away = devig_multiplicative(
                    american_to_implied(home), american_to_implied(away)
                )
                assert p_home + p_away == 1.0, f"{home}/{away} summed to {p_home + p_away!r}"
                assert 0.0 < p_home < 1.0
                assert 0.0 < p_away < 1.0

    def test_moneyline_to_prob_matches_the_two_step_form(self) -> None:
        assert moneyline_to_prob(-200, 170) == pytest.approx(9 / 14, abs=1e-12)


class TestSpreadToProb:
    def test_pick_em_is_a_coin_flip(self) -> None:
        assert spread_to_prob(0.0, SIGMA) == 0.5

    def test_heavy_home_favourite_maps_above_ninety_percent(self) -> None:
        """The sign-convention test. Spreads are home-relative: negative favours home.

        An inverted sign is the classic silent killer for this phase — it would still
        produce probabilities in (0, 1), still sum to 1, and still pass every other test
        in this file. This is the assertion that catches it.
        """
        assert spread_to_prob(-21.0, SIGMA) > 0.9

    def test_heavy_home_underdog_maps_below_ten_percent(self) -> None:
        assert spread_to_prob(21.0, SIGMA) < 0.1

    def test_is_symmetric_about_pick_em(self) -> None:
        assert spread_to_prob(-14.0, SIGMA) == pytest.approx(1.0 - spread_to_prob(14.0, SIGMA))

    def test_is_monotonic_in_the_spread(self) -> None:
        """More favoured must never mean less likely to win."""
        spreads = [s / 2 for s in range(-124, 125)]
        probs = [spread_to_prob(s, SIGMA) for s in spreads]
        assert all(earlier > later for earlier, later in zip(probs, probs[1:], strict=False))

    @pytest.mark.parametrize("sigma", [0.0, -1.0])
    def test_rejects_non_positive_sigma(self, sigma: float) -> None:
        with pytest.raises(ValueError, match="sigma must be positive"):
            spread_to_prob(-3.0, sigma)

    def test_normal_cdf_matches_known_quantiles(self) -> None:
        assert normal_cdf(0.0) == 0.5
        assert normal_cdf(1.0) == pytest.approx(0.8413447461, abs=1e-9)
        assert normal_cdf(1.959963985) == pytest.approx(0.975, abs=1e-9)
        assert normal_cdf(-3.0) == pytest.approx(0.0013498980, abs=1e-9)

    def test_extreme_spreads_stay_inside_the_open_interval(self) -> None:
        """The database's widest real spreads are -62 and +54."""
        for spread in (-62.0, 54.0):
            probability = spread_to_prob(spread, SIGMA)
            assert 0.0 < probability < 1.0
            assert math.isfinite(probability)
