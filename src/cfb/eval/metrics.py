"""Scoring vocabulary for Phase 6: reliability bins and the skill fraction.

Two things this module deliberately does **not** contain.

First, **Brier and log loss are imported, not reimplemented.** They already exist in
:mod:`cfb.model.train`, which is the module that produced Phase 5's published validation
numbers, and :mod:`cfb.eval.evaluate` imports that module anyway for ``predict`` and
``apply_calibrator``. A third copy here would mean the number in ``train_report.json`` and
the number in ``results_table.md`` came from different code, and the day they disagreed the
first suspicion would fall on the model rather than on the arithmetic.

Be careful when reaching for a scorer elsewhere in this project: :mod:`cfb.elo.engine` has
same-named functions with the **arguments in the opposite order** —
``engine.log_loss(probabilities, outcomes)`` against
``train.log_loss_score(outcomes, probabilities)``. Brier is symmetric in its two arguments
and would survive the mix-up silently; log loss would not. Phase 6 uses the ``train``
convention (outcomes first, as in scikit-learn) everywhere, and Phase 3's copy is left
alone rather than refactored inside an evaluation change.

Second, there is **no calibration summary statistic** here — no ECE, no MCE. A single
number that says "well calibrated" invites exactly the over-claim this project is built to
avoid, and the honest artefact is the curve with its bin counts attached, where a bin
holding 12 games looks like a bin holding 12 games. The reliability table below is what the
plot draws and what the results table prints.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cfb.model.train import brier_score, log_loss_score

__all__ = [
    "ReliabilityBin",
    "brier_score",
    "log_loss_score",
    "reliability_table",
    "resolution",
    "skill_fraction",
]

N_BINS: int = 10
"""The plan's bin count: 10 equal-count bins.

Equal-*count* rather than equal-*width*. Predicted probabilities on college football games
pile up in the middle and thin out at the tails, so ten equal-width bins would put a few
thousand games in the middle bins and a handful in the outer ones, and the curve's endpoints
— the part everyone looks at — would be the noisiest part of the picture while looking like
the rest of it.
"""


@dataclass(frozen=True)
class ReliabilityBin:
    """One bin of a reliability curve.

    Attributes:
        lo: Lowest predicted probability in the bin.
        hi: Highest predicted probability in the bin.
        n: How many games fell in it. Carried everywhere the bin goes, because a point
            plotted from 12 games and a point plotted from 240 are not the same evidence
            and the curve cannot show the difference on its own.
        mean_predicted: Mean predicted probability across the bin — the x coordinate.
        empirical: Fraction of those games the home team actually won — the y coordinate.
    """

    lo: float
    hi: float
    n: int
    mean_predicted: float
    empirical: float

    @property
    def gap(self) -> float:
        """Empirical frequency minus mean prediction.

        Positive means the system was underconfident in this bin (the home team won more
        often than it said), negative means overconfident.
        """
        return self.empirical - self.mean_predicted


def reliability_table(
    outcomes: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    n_bins: int = N_BINS,
) -> list[ReliabilityBin]:
    """Bin predictions into equal-count bins and measure each bin's empirical frequency.

    Bins are formed by sorting on the predicted probability and splitting the *sorted
    order* into ``n_bins`` near-equal pieces, rather than by cutting on quantile values.
    The distinction matters when predictions tie: quantile edges collapse when many games
    share a probability, silently producing empty bins and unequal ones, whereas splitting
    the order keeps the counts equal by construction and lets the tie land wherever the sort
    puts it. The sort is stable, so the same input always produces the same table.

    Args:
        outcomes: 0/1 outcomes, one per game.
        probabilities: Predicted probability of the 1 outcome, same order.
        n_bins: How many bins to form.

    Returns:
        The bins, ascending by predicted probability. Bins are non-empty by construction;
        with fewer games than bins, the surplus bins are simply not returned.

    Raises:
        ValueError: If the inputs differ in length, are empty, or ``n_bins`` is not
            positive.
    """
    actual = np.asarray(outcomes, dtype=float)
    predicted = np.asarray(probabilities, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(
            f"length mismatch: {actual.shape} outcomes vs {predicted.shape} predictions"
        )
    if actual.size == 0:
        raise ValueError("no predictions to bin")
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins!r}")

    order = np.argsort(predicted, kind="stable")
    bins = []
    for chunk in np.array_split(order, min(n_bins, actual.size)):
        if chunk.size == 0:
            continue
        chunk_predicted = predicted[chunk]
        bins.append(
            ReliabilityBin(
                lo=float(chunk_predicted.min()),
                hi=float(chunk_predicted.max()),
                n=int(chunk.size),
                mean_predicted=float(chunk_predicted.mean()),
                empirical=float(actual[chunk].mean()),
            )
        )
    return bins


def resolution(probabilities: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    """Describe how many distinct values a system's predictions actually take.

    A diagnostic for step-function calibrators. Isotonic regression fitted on one thin
    season maps wide stretches of the model's output onto a single fitted value, so hundreds
    of genuinely different games come out with an identical probability. Brier and log loss
    both notice that eventually, but neither one *says* it, and a reliability curve drawn
    over equal-count bins hides it completely — the bin means still move.

    Args:
        probabilities: Predicted probabilities.

    Returns:
        ``n``, ``n_distinct``, ``largest_plateau`` (games sharing one value) and
        ``largest_plateau_value``.

    Raises:
        ValueError: If there are no predictions.
    """
    values = np.asarray(probabilities, dtype=float)
    if values.size == 0:
        raise ValueError("no predictions to describe")
    distinct, counts = np.unique(values, return_counts=True)
    largest = int(counts.argmax())
    return {
        "n": int(values.size),
        "n_distinct": int(distinct.size),
        "largest_plateau": int(counts[largest]),
        "largest_plateau_value": float(distinct[largest]),
    }


def skill_fraction(naive_brier: float, model_brier: float, vegas_brier: float) -> float:
    """Fraction of the naive-to-Vegas distance the model closed.

    ``(naive - model) / (naive - vegas)``. This is the honest headline the plan asks for:
    "the model closes X% of the gap between a naive baseline and the closing line" says
    where the model sits between a floor anyone could reach and a ceiling it is not
    expected to reach, which a raw Brier of 0.20 does not.

    Read it with its own limits in mind. It is a ratio of two small differences, so it moves
    a lot on little; it is not a percentage of anything anyone can bet on; and it exceeds
    1.0 exactly when the model has scored better than the line, which in this project is a
    leakage alarm rather than a good result.

    Args:
        naive_brier: Brier of the constant home-rate baseline.
        model_brier: Brier of the model.
        vegas_brier: Brier of the de-vigged closing line.

    Returns:
        The fraction, on the same identical game set all three were scored on.

    Raises:
        ValueError: If the naive baseline is not worse than the line. The denominator is
            then zero or negative and the ratio is meaningless — and the situation itself
            (a constant prediction matching the closing line) means something upstream is
            badly wrong, which is worth stopping for rather than dividing through.
    """
    denominator = naive_brier - vegas_brier
    if denominator <= 0:
        raise ValueError(
            f"naive Brier {naive_brier!r} is not worse than the line's {vegas_brier!r}; "
            "the naive-to-Vegas distance is zero or negative and the skill fraction has no "
            "meaning. Check that both were scored on the same games."
        )
    return (naive_brier - model_brier) / denominator
