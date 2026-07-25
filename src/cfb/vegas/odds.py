"""Pure odds arithmetic for the Vegas benchmark.

No I/O, no database, no configuration. Every function here is total and unit-tested in
isolation, because this is the arithmetic that decides what the whole project is measured
against: a sign error or a botched vig removal produces a benchmark that is wrong in
silence, with every downstream test still green (``RISKS.md`` #8).

Two conventions are load-bearing and are stated once, here:

* **American odds.** Negative odds ``-m`` stake ``m`` to win 100; positive odds ``+p``
  stake 100 to win ``p``. Implied probability includes the book's vig, so a pair of
  implied probabilities sums to more than 1.
* **Spreads are home-relative**, so a negative spread means the home team is favoured.
  Verified against a real game in ``DECISIONS.md`` (2023 Massachusetts vs Merrimack,
  ``formattedSpread`` "Massachusetts -16.5" with ``spread: -16.5``).
"""

from __future__ import annotations

import math

MIN_ODDS_MAGNITUDE = 100
"""American odds are undefined between -100 and +100; both extremes mean even money."""


def american_to_implied(odds: int) -> float:
    """Convert American odds to the implied probability, vig included.

    Args:
        odds: American odds, e.g. ``-110`` or ``+250``. Magnitude must be at least 100.

    Returns:
        The implied probability in (0, 1). Not de-vigged: pair this with the other side
        and pass both to :func:`devig_multiplicative`.

    Raises:
        ValueError: If ``odds`` has magnitude below 100, which is not a valid quote.
            Refusing is deliberate — the alternative is a plausible-looking number
            derived from a value the formula does not describe.
    """
    if abs(odds) < MIN_ODDS_MAGNITUDE:
        raise ValueError(f"American odds must have magnitude >= 100, got {odds!r}")
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def devig_multiplicative(implied_home: float, implied_away: float) -> tuple[float, float]:
    """Remove the vig by normalising a pair of implied probabilities to sum to 1.

    Multiplicative normalisation is the standard, transparent baseline. It assumes the
    book's margin is spread proportionally across both sides — Shin and power methods
    relax that to model insider-trading asymmetry, which matters more in thin markets
    than in heavily-traded CFB closing lines. Since the benchmark is used for scoring
    rather than betting, second-order refinements move the yardstick by less than model
    noise. Logged in ``DECISIONS.md``; Shin is noted as a Phase 6 robustness check.

    Args:
        implied_home: Home implied probability from :func:`american_to_implied`.
        implied_away: Away implied probability.

    Returns:
        ``(p_home, p_away)``, summing to exactly 1.0. The away side is computed as
        ``1 - p_home`` rather than by its own division: two independent divisions can
        sum to 0.9999999999999999 in binary floating point, and the exit criterion for
        this phase asks for an exact sum.

    Raises:
        ValueError: If either input is outside (0, 1).
    """
    for name, value in (("implied_home", implied_home), ("implied_away", implied_away)):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {value!r}")
    p_home = implied_home / (implied_home + implied_away)
    return p_home, 1.0 - p_home


def moneyline_to_prob(home_moneyline: int, away_moneyline: int) -> float:
    """Convert a two-sided moneyline quote to a de-vigged home win probability.

    Args:
        home_moneyline: American odds on the home team.
        away_moneyline: American odds on the away team.

    Returns:
        The de-vigged probability that the home team wins.

    Raises:
        ValueError: If either quote is not valid American odds.
    """
    return devig_multiplicative(
        american_to_implied(home_moneyline),
        american_to_implied(away_moneyline),
    )[0]


def normal_cdf(z: float) -> float:
    """Standard normal cumulative distribution function.

    Written out with :func:`math.erf` rather than pulling in SciPy: it is one line, exact
    to machine precision, and keeps the one module the benchmark depends on dependency-free.

    Args:
        z: A standard-normal quantile.

    Returns:
        P(Z <= z).
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def spread_to_prob(spread: float, sigma: float) -> float:
    """Convert a home-relative point spread to a home win probability.

    Models the game margin as ``Normal(-spread, sigma)``: the spread is the market's
    expected margin with the sign flipped, and ``sigma`` is the typical error of that
    expectation, fitted on training seasons only. The home team wins when the margin
    exceeds 0, so ``p_home = P(margin > 0) = Phi(-spread / sigma)``.

    A negative spread means the home team is favoured and must therefore produce a
    probability above 0.5 — the direction asserted in the sign-convention test.

    Args:
        spread: The closing spread from the home team's perspective.
        sigma: Standard deviation of margin-versus-spread residuals. Must be positive.

    Returns:
        The home win probability in (0, 1).

    Raises:
        ValueError: If ``sigma`` is not positive.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma!r}")
    return normal_cdf(-spread / sigma)
