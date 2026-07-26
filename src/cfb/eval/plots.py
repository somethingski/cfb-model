"""Reliability curves — the one picture Phase 6 produces.

A reliability curve plots what a system *said* against what actually *happened*: for each
bin of predicted probability, the mean prediction on the x axis and the fraction of those
games the home team won on the y axis. A perfectly calibrated system sits on the diagonal.

Two choices about what the plot is allowed to do:

* **The bin size is stated on every panel.** A point built from 12 games and a point built
  from 240 look identical on a reliability curve, and the eye goes to the tails, which are
  usually the thin ones. Equal-count binning makes this one caption rather than ten
  annotations — every point in a panel rests on the same number of games — so a wobbly
  endpoint can be read as a small sample rather than as miscalibration.
* **The diagonal is drawn first and in grey**, so nothing about the styling makes a curve
  look closer to it than it is. The expected honest picture is the line hugging the diagonal
  tightest, the model close behind, and Elo-only visibly looser; if the model's curve looks
  *better* than the line's, that is a red flag before it is an achievement.

Matplotlib is used with the non-interactive ``Agg`` backend so the figure renders the same
way in a terminal, in CI, and on a machine with no display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before pyplot is imported)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cfb.eval.metrics import ReliabilityBin, reliability_table  # noqa: E402
from cfb.model.train import outcomes  # noqa: E402

SERIES: tuple[tuple[str, str, str], ...] = (
    ("vegas", "de-vigged closing line", "#1f77b4"),
    ("model", "model (calibrated)", "#d62728"),
    ("elo_only", "Elo only", "#7f7f7f"),
)
"""Which systems are drawn, their labels, and their colours.

The naive baseline is not drawn: it predicts one constant, so its curve is a single point.
Drawn in benchmark-first order so the line is the reference the eye lands on.
"""


def _bin_size_caption(tables: dict[str, list[ReliabilityBin]]) -> str:
    """Describe how many games each plotted point rests on.

    Equal-count binning makes this one sentence rather than ten annotations: every point in
    a panel is built from the same number of games, give or take the remainder when the
    total does not divide evenly. Writing it once beats labelling ten points that all say
    240 and overlap the curve doing it.

    Args:
        tables: System key to its bins.

    Returns:
        A short caption.
    """
    counts = {current.n for bins in tables.values() for current in bins}
    if not counts:
        return ""
    if len(counts) == 1:
        return f"each point = {counts.pop()} games"
    return f"each point = {min(counts)}–{max(counts)} games"


def _draw_panel(axis, tables: dict[str, list[ReliabilityBin]], title: str, n_games: int) -> None:
    """Draw one reliability panel.

    Args:
        axis: Matplotlib axes.
        tables: System key to its bins.
        title: Panel title.
        n_games: Games behind the panel, for the title.
    """
    axis.plot([0, 1], [0, 1], color="#bbbbbb", linewidth=1, linestyle="--", zorder=1)
    for key, label, colour in SERIES:
        bins = tables.get(key)
        if not bins:
            continue
        axis.plot(
            [current.mean_predicted for current in bins],
            [current.empirical for current in bins],
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=colour,
            label=label,
            zorder=2,
        )
    axis.text(
        0.98,
        0.02,
        _bin_size_caption(tables),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#666666",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_aspect("equal")
    axis.set_title(f"{title} (n={n_games})", fontsize=10)
    axis.set_xlabel("predicted P(home win)", fontsize=8)
    axis.set_ylabel("observed frequency", fontsize=8)
    axis.tick_params(labelsize=7)
    axis.grid(True, alpha=0.15)


def reliability_figure(
    evaluation: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    path: Path,
    n_bins: int | None = None,
) -> Path:
    """Draw the overall panel and one panel per test season, and save the figure.

    Args:
        evaluation: The evaluation frame.
        predictions: System key to probabilities over that frame.
        path: Destination PNG.
        n_bins: Bins per panel; None uses :data:`cfb.eval.metrics.N_BINS`.

    Returns:
        The path written.
    """
    actual = outcomes(evaluation)
    seasons = evaluation["season"].to_numpy()
    panels: list[tuple[str, np.ndarray]] = [("all test seasons", np.ones(len(evaluation), bool))]
    for season in sorted({int(value) for value in seasons}):
        panels.append((str(season), seasons == season))

    figure, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6))
    axes = np.atleast_1d(axes)
    for axis, (title, mask) in zip(axes, panels, strict=True):
        kwargs = {} if n_bins is None else {"n_bins": n_bins}
        tables = {
            key: reliability_table(actual[mask], predictions[key][mask], **kwargs)
            for key, _, _ in SERIES
            if key in predictions
        }
        _draw_panel(axis, tables, title, int(mask.sum()))

    axes[0].legend(loc="upper left", fontsize=7, framealpha=0.9)
    figure.suptitle(
        "Reliability: predicted vs. observed home-win frequency, equal-count bins", fontsize=11
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
