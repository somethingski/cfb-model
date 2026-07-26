"""Pure Elo arithmetic. No database, no configuration, no I/O.

Everything here is a total function of its arguments, because the alternative — Elo
arithmetic tangled up with the chronological walk that feeds it — makes the two failure
modes of this phase indistinguishable. A wrong K and a game processed out of order both
show up as "the ratings look a bit off". Keeping the arithmetic here and the ordering in
:mod:`cfb.elo.pipeline` means each can be tested on its own.

The update rule is standard Elo with FiveThirtyEight's margin-of-victory multiplier::

    E_home = 1 / (1 + 10^(-(R_home - R_away + HFA) / 400))
    delta  = K * mov_multiplier * (S_home - E_home)

Ratings are zero-sum *in this module*: the home team gains exactly what the away team
loses. The pipeline breaks that on purpose for FCS opponents, whose rating is a fixed
constant that never moves (see :data:`EloParams.fcs`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

SCALE = 400.0
"""Elo's rating scale: a 400-point edge is 10:1 odds. Definitional, never tuned."""

MOV_SHAPE = 2.2
"""FiveThirtyEight's autocorrelation-damping constant. Copied from their form, not fitted."""

LOG_LOSS_EPS = 1e-15
"""Clip bound for log loss. Elo probabilities never reach 0 or 1, but the objective should
not be one bad input away from returning infinity."""


@dataclass(frozen=True)
class EloParams:
    """The rating system's parameters.

    Defaults are the Phase 3 plan's starting values, confirmed by Sean before any code was
    written. ``k``, ``hfa``, and ``regression`` are tuned by :mod:`cfb.elo.tune` on training
    seasons only and then frozen into ``elo_params.json``; the rest are priors, not fitted
    quantities, and are deliberately left out of the grid.

    Attributes:
        k: Step size of a rating update.
        hfa: Elo points added to the home side inside the expected-score calculation. Zero
            at neutral sites.
        regression: Fraction of the gap to ``mean`` closed at each new season, so
            ``1/3`` means ``R <- (2/3)*R + (1/3)*1500``.
        mean: The rating seasons regress toward. The global mean, not a conference mean —
            conference realignment makes conference means noisy.
        initial: Rating for an FBS team in the first season of the data.
        newcomer: Rating for a team entering FBS after the first season. Below ``initial``
            because a promoted team is a below-average FBS team, not an average one.
        fcs: Fixed rating for every FCS opponent. Never updates.
    """

    k: float = 35.0
    hfa: float = 65.0
    regression: float = 1.0 / 3.0
    mean: float = 1500.0
    initial: float = 1500.0
    newcomer: float = 1300.0
    fcs: float = 1200.0

    def to_dict(self) -> dict[str, float]:
        """Return the parameters as a JSON-serialisable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> EloParams:
        """Build parameters from a dict, ignoring unrelated keys.

        Args:
            values: Mapping that contains at least some parameter names. Keys that are not
                parameters (metadata written alongside them in ``elo_params.json``) are
                ignored so the file can carry provenance without a separate schema.

        Returns:
            An ``EloParams`` with the given values, defaults elsewhere.
        """
        fields = {field for field in cls().to_dict()}
        return cls(**{key: float(value) for key, value in values.items() if key in fields})


def expected(rating: float, opponent_rating: float, hfa: float = 0.0) -> float:
    """Expected score for a team, in [0, 1].

    "Expected score" is a win probability only because college football has no ties in
    practice (0 in 10,373 completed games in this database). With ties it would be
    ``P(win) + P(tie)/2``.

    Args:
        rating: The team's pre-game rating.
        opponent_rating: The opponent's pre-game rating.
        hfa: Home-field advantage in Elo points, from ``rating``'s perspective. Positive
            when ``rating`` is the home team, zero at a neutral site, and negative when
            ``rating`` is the away team.

    Returns:
        The expected score. Symmetric by construction:
        ``expected(a, b, h) + expected(b, a, -h) == 1``.
    """
    return 1.0 / (1.0 + 10.0 ** (-(rating - opponent_rating + hfa) / SCALE))


def mov_multiplier(margin: float, winner_elo_diff: float) -> float:
    """FiveThirtyEight's margin-of-victory multiplier.

    Two things it does at once. The logarithm means a 40-point win counts for more than a
    3-point win but nowhere near thirteen times more. The denominator shrinks the
    multiplier when the *winner* was already the stronger side and grows it when an
    underdog wins, which is what stops good teams from ratcheting upward forever on
    blowouts of bad teams (autocorrelation).

    That second effect requires ``winner_elo_diff`` to be **signed** from the winner's
    perspective. The Phase 3 plan wrote it with an absolute value, which would damp
    upsets exactly as hard as expected blowouts and defeat the stated purpose; this
    follows the named FiveThirtyEight form instead. Logged in ``DECISIONS.md``.

    Args:
        margin: Final margin. Only its magnitude is used; the sign of the rating change
            comes from ``S - E`` in :func:`update`.
        winner_elo_diff: Winner's pre-game rating minus the loser's, home-field advantage
            included. Negative when the underdog won.

    Returns:
        The multiplier, ``0.0`` for a tie because ``ln(0 + 1) = 0``. A tie therefore moves
        no ratings at all. That branch is unreachable in this dataset — college football
        has had overtime since 1996 — so it is documented and pinned by a test rather than
        patched with an invented minimum margin.

    Raises:
        ValueError: If ``winner_elo_diff`` is below -2200, where the denominator turns
            negative and the multiplier would silently flip sign. Unreachable with these
            ratings (the widest gap the system can produce is roughly 850 points), so a
            hit means the ratings have diverged and the run should stop rather than
            produce plausible-looking numbers.
    """
    denominator = 0.001 * winner_elo_diff + MOV_SHAPE
    if denominator <= 0.0:
        raise ValueError(
            f"winner_elo_diff={winner_elo_diff!r} makes the MOV denominator non-positive; "
            "ratings have diverged far beyond anything this system should produce"
        )
    return math.log(abs(margin) + 1.0) * MOV_SHAPE / denominator


def update(
    home_rating: float,
    away_rating: float,
    home_points: int,
    away_points: int,
    params: EloParams,
    neutral_site: bool = False,
) -> tuple[float, float]:
    """Apply one game to a pair of ratings.

    Args:
        home_rating: Home team's pre-game rating.
        away_rating: Away team's pre-game rating.
        home_points: Final home score.
        away_points: Final away score.
        params: Rating parameters.
        neutral_site: True to zero out home-field advantage.

    Returns:
        ``(home_post, away_post)``. Zero-sum: the home team's gain is the away team's loss.
        The caller decides which of the two to actually keep — the pipeline discards the
        FCS side.
    """
    hfa = 0.0 if neutral_site else params.hfa
    expected_home = expected(home_rating, away_rating, hfa)

    margin = home_points - away_points
    if margin > 0:
        score_home = 1.0
        winner_elo_diff = (home_rating + hfa) - away_rating
    elif margin < 0:
        score_home = 0.0
        winner_elo_diff = away_rating - (home_rating + hfa)
    else:
        score_home = 0.5
        winner_elo_diff = 0.0

    delta = params.k * mov_multiplier(margin, winner_elo_diff) * (score_home - expected_home)
    return home_rating + delta, away_rating - delta


def brier_score(probabilities: Sequence[float], outcomes: Sequence[float]) -> float:
    """Mean squared error of probabilistic predictions.

    Lives here rather than in a scoring module because Phase 3 has two callers for it —
    the tuning objective and the sanity report — and no third one yet. Phase 6 owns
    evaluation properly and may well take these over.

    Args:
        probabilities: Predicted probabilities of the outcome coded as 1.
        outcomes: Realised outcomes, each 0.0 or 1.0.

    Returns:
        The Brier score; lower is better.

    Raises:
        ValueError: If the inputs differ in length or are empty.
    """
    if len(probabilities) != len(outcomes):
        raise ValueError(f"length mismatch: {len(probabilities)} vs {len(outcomes)}")
    if not probabilities:
        raise ValueError("no predictions to score")
    return sum((p - y) ** 2 for p, y in zip(probabilities, outcomes, strict=True)) / len(outcomes)


def log_loss(probabilities: Sequence[float], outcomes: Sequence[float]) -> float:
    """Mean negative log likelihood of probabilistic predictions.

    Args:
        probabilities: Predicted probabilities of the outcome coded as 1. Clipped away
            from 0 and 1 by ``LOG_LOSS_EPS`` so that one confident miss costs a large
            finite number rather than infinity, which would make the tuning objective
            unorderable.
        outcomes: Realised outcomes, each 0.0 or 1.0.

    Returns:
        The mean log loss; lower is better.

    Raises:
        ValueError: If the inputs differ in length or are empty.
    """
    if len(probabilities) != len(outcomes):
        raise ValueError(f"length mismatch: {len(probabilities)} vs {len(outcomes)}")
    if not probabilities:
        raise ValueError("no predictions to score")
    total = 0.0
    for p, y in zip(probabilities, outcomes, strict=True):
        clipped = min(max(p, LOG_LOSS_EPS), 1.0 - LOG_LOSS_EPS)
        total -= y * math.log(clipped) + (1.0 - y) * math.log(1.0 - clipped)
    return total / len(outcomes)


def run_season_regression(rating: float, params: EloParams) -> float:
    """Pull a rating toward the league mean at the start of a new season.

    Rosters turn over, coaches leave, and last November's rating is a stale estimate of
    this September's team. Regression is the system's admission that it knows less at a
    season boundary than it did at the final whistle.

    Args:
        rating: End-of-previous-season rating.
        params: Rating parameters; uses ``regression`` and ``mean``.

    Returns:
        The pre-season rating.
    """
    return rating + (params.mean - rating) * params.regression
