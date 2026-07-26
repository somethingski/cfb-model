"""The systems being compared, and the two things Phase 6 is allowed to fit.

Four systems are scored (the plan's a-d), plus two secondary series:

======================  ==========================================================
(a) model               the calibrated GBT from Phase 5
(b) vegas               the de-vigged closing line from Phase 2
(c) naive home          a constant, the training-season home-win rate
(d) elo only            a logistic of the Elo difference plus home-field advantage
secondary: model raw    the same GBT without the calibrator
secondary: model platt  the same GBT with the Phase 5 alternative calibrator
======================  ==========================================================

Phase 6 reads test-season labels, which no other phase may do, so the rule about what may
be *fitted* has to be stated rather than assumed: **nothing here is fitted on a test
season.** Two quantities are fitted at all — the Elo logistic scale (training seasons) and
the Platt calibrator (the validation season) — and each calls the matching guard from
:mod:`cfb.model.splits` on its own input before it fits, so the check sits on the only path
to the fitted value rather than beside it.

Even the baselines respect the split. The naive home rate is the *training* seasons' rate,
not the test seasons' own, because a constant fitted on the games it is scored on is a
season-level aggregate applied inside that season — the leakage pattern ``CLAUDE.md`` names,
and one that would flatter the baseline the model is measured against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cfb.model import splits
from cfb.model.train import log_loss_score, outcomes

ELO_DEFINITIONAL_SCALE: float = 400.0
"""Elo's own scale, the one the ratings were built under (``cfb.elo.engine.SCALE``).

Reported alongside the fitted scale rather than replaced by it. The rating updates use 400
by definition, so a fitted value far from it would say the ratings and their probability
reading disagree — worth seeing, not worth hiding behind a single tuned number.
"""

SCALE_SEARCH_BOUNDS: tuple[float, float] = (50.0, 2000.0)
SCALE_SEARCH_POINTS: int = 400
SCALE_SEARCH_PASSES: int = 3
"""Grid-refinement settings for the one-dimensional scale fit.

Three passes of 400 points over a shrinking window resolves the scale to well under a tenth
of a point, which is far finer than the objective can distinguish. A grid rather than an
optimiser because it needs no new dependency, cannot fail to converge, and gives the same
answer on every run — the same reasoning that put a grid in Phase 3.
"""


@dataclass(frozen=True)
class EloOnly:
    """The Elo-only baseline: a logistic of the rating difference plus home-field.

    Attributes:
        scale: Fitted logistic scale, in Elo points per decade of odds.
        hfa: Home-field advantage in Elo points, taken from the frozen ``elo_params.json``
            rather than re-fitted. Phase 3 already fitted it on training seasons; fitting it
            again here would create a second definition of a frozen parameter.
        fitted_on: Seasons the scale was fitted on, for the report.
        train_log_loss: The objective value at the fitted scale.
    """

    scale: float
    hfa: float
    fitted_on: tuple[int, ...]
    train_log_loss: float

    def probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict home-win probabilities for a model frame.

        Args:
            frame: Rows carrying ``elo_diff`` and ``neutral_site``.

        Returns:
            One probability per row, in the frame's order.
        """
        return elo_probabilities(frame, self.scale, self.hfa)


def elo_probabilities(frame: pd.DataFrame, scale: float, hfa: float) -> np.ndarray:
    """Convert Elo differences into home-win probabilities.

    ``p = 1 / (1 + 10 ** (-(elo_diff + hfa) / scale))``, with the home-field term dropped at
    neutral sites — the same rule :func:`cfb.elo.engine.expected` applies inside the rating
    walk, so the baseline is the rating system's own reading of a matchup and not a
    differently-shaped model wearing its ratings.

    Args:
        frame: Rows carrying ``elo_diff`` and ``neutral_site``.
        scale: Elo points per decade of odds.
        hfa: Home-field advantage in Elo points.

    Returns:
        One probability per row, in the frame's order.

    Raises:
        ValueError: If ``scale`` is not positive, which would invert the curve.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale!r}")
    edge = frame["elo_diff"].to_numpy(dtype=float) + hfa * (
        1.0 - frame["neutral_site"].to_numpy(dtype=float)
    )
    return 1.0 / (1.0 + 10.0 ** (-edge / scale))


def fit_elo_scale(train: pd.DataFrame, hfa: float) -> EloOnly:
    """Fit the Elo-only logistic scale on training seasons.

    The guard runs before the search, not after it, so a caller that hands this function a
    frame containing 2024 raises instead of quietly fitting the baseline on the games the
    baseline is about to be scored on. That would not make the *model* look better — it
    would make the yardstick under it look better, which is the subtler direction and the
    one nobody double-checks.

    Args:
        train: Training-season rows of the model frame.
        hfa: Home-field advantage in Elo points, from ``elo_params.json``.

    Returns:
        The fitted baseline.

    Raises:
        LeakageError: If any test-season row is present.
        ValueError: If the frame is empty.
    """
    splits.assert_no_test_rows(train, "the Elo-only scale fit")
    if train.empty:
        raise ValueError("no training rows to fit the Elo-only scale on")

    actual = outcomes(train)
    lo, hi = SCALE_SEARCH_BOUNDS
    best_scale = float(np.mean(SCALE_SEARCH_BOUNDS))
    best_loss = float("inf")

    for _ in range(SCALE_SEARCH_PASSES):
        grid = np.linspace(lo, hi, SCALE_SEARCH_POINTS)
        for scale in grid:
            loss = log_loss_score(actual, elo_probabilities(train, float(scale), hfa))
            if loss < best_loss:
                best_loss, best_scale = loss, float(scale)
        step = (hi - lo) / (SCALE_SEARCH_POINTS - 1)
        lo, hi = max(SCALE_SEARCH_BOUNDS[0], best_scale - step), best_scale + step

    return EloOnly(
        scale=best_scale,
        hfa=hfa,
        fitted_on=splits.seasons_in(train),
        train_log_loss=best_loss,
    )


def naive_home_rate(train: pd.DataFrame) -> float:
    """The constant the naive baseline predicts: the training-season home-win rate.

    Computed the same way :mod:`cfb.model.train` computes it, on the same rows, so the
    baseline in ``results_table.md`` is the same object as the baseline in
    ``train_report.json`` rather than a second one that happens to be close.

    Args:
        train: Training-season rows of the model frame.

    Returns:
        The home-win rate.

    Raises:
        LeakageError: If any test-season row is present.
        ValueError: If the frame is empty.
    """
    splits.assert_no_test_rows(train, "the naive home-rate baseline")
    if train.empty:
        raise ValueError("no training rows to compute the naive home rate from")
    return float(outcomes(train).mean())


def fit_platt(validation: pd.DataFrame, raw: np.ndarray) -> LogisticRegression:
    """Fit the Platt-scaling alternative to Phase 5's isotonic calibrator.

    ``RISKS.md`` #4 and #25 both name Platt scaling as the robustness comparison for a
    calibrator fitted on one thin season, and this is where that comparison gets made. It is
    fitted on the same 776 validation games as the isotonic one, so the two differ in method
    and in nothing else.

    The logistic is fitted on the raw model's **log-odds** rather than on its probability,
    which is what makes it the standard Platt form: a monotone rescaling of the model's
    score with two parameters, against isotonic's arbitrary monotone step function. Two
    parameters cannot map a tail to certainty the way isotonic can, which is precisely the
    failure mode the clip in Phase 5 exists to bound.

    Args:
        validation: The validation rows the predictions came from.
        raw: Raw model probabilities for those rows, in the same order.

    Returns:
        The fitted logistic.

    Raises:
        LeakageError: If ``validation`` holds any season other than the validation season.
        ValueError: If the arrays are not aligned.
    """
    splits.assert_validation_only(validation, "the Platt calibrator fit")
    if len(validation) != len(raw):
        raise ValueError(f"frame has {len(validation)} rows but {len(raw)} predictions were given")

    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(_log_odds(raw), outcomes(validation))
    return model


def apply_platt(model: LogisticRegression, raw: np.ndarray) -> np.ndarray:
    """Apply the Platt calibrator to raw model probabilities.

    Args:
        model: A fitted logistic from :func:`fit_platt`.
        raw: Raw model probabilities.

    Returns:
        Calibrated probabilities. Not clipped: a two-parameter logistic cannot reach 0 or 1,
        so the clip that isotonic needs would do nothing here except hide that difference.
    """
    return np.asarray(model.predict_proba(_log_odds(raw))[:, 1])


def _log_odds(probabilities: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Return probabilities as a column of log-odds, clipped away from the asymptotes."""
    clipped = np.clip(np.asarray(probabilities, dtype=float), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
