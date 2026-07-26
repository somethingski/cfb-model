"""Season-forward splits, and the guards that keep the test seasons untouched.

This module is the phase's leakage boundary, and it is deliberately small: split
membership is *data*, defined once and imported everywhere, so there is no second place
for a season to be classified differently. Every boundary is derived from
:mod:`cfb.config` rather than restated, so advancing the project to a new completed season
extends the test set instead of silently shifting the train/validation line.

Why season-forward at all: a random split puts week 12 of 2019 in the training set and
week 3 of 2019 in the test set, so the model is fitted on games played *after* the ones it
is scored on. Nothing in the feature code leaks in that arrangement — the leak is in the
split itself, and it is invisible in every metric except the ones that get better.

The hyperparameter search uses forward-chaining cross-validation (:func:`forward_folds`)
for the same reason. ``KFold`` and ``train_test_split`` are bugs here, per ``CLAUDE.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pandas as pd

from cfb import config


class LeakageError(AssertionError):
    """Raised when data from a split reaches a place that split must never reach.

    Subclasses ``AssertionError`` so it reads as what it is — a violated invariant — and
    so a stray ``except ValueError`` around a fitting call cannot swallow it.
    """


TRAIN_SEASONS: tuple[int, ...] = tuple(
    season for season in config.SEASONS if season <= config.TRAIN_LAST_SEASON
)
"""Seasons the GBT may be fitted on: 2014-2021.

The boundary is :data:`cfb.config.TRAIN_LAST_SEASON`, the same constant Phase 2's sigma
and Phase 3's grid search are fitted under. One definition, three phases.
"""

VALIDATION_SEASONS: tuple[int, ...] = (config.TRAIN_LAST_SEASON + 1,)
"""The single season the isotonic calibrator is fitted on: 2022.

One season rather than two is an assumption the plan records: ~750 usable games is thin
for isotonic regression, which is why the calibrated output is clipped. The alternative
(calibrating on 2021-2022 and shrinking the training set) was considered and not taken.
"""

TEST_SEASONS: tuple[int, ...] = tuple(
    season for season in config.SEASONS if season > VALIDATION_SEASONS[-1]
)
"""Seasons that must not be read in this phase at all: 2023-2025.

Phase 6 evaluates on these. Nothing here loads them except the guards below, whose whole
job is to notice if they arrive somewhere they should not.
"""

SPLITS: Mapping[str, tuple[int, ...]] = {
    "train": TRAIN_SEASONS,
    "validation": VALIDATION_SEASONS,
    "test": TEST_SEASONS,
}
"""The split scheme as data. Imported; never re-derived at a call site."""


def split_of(season: int) -> str:
    """Name the split a season belongs to.

    Args:
        season: Season year.

    Returns:
        ``"train"``, ``"validation"`` or ``"test"``.

    Raises:
        ValueError: If the season is outside the project's range. A season with no split
            is a season with no rule, and defaulting it to anything is how a 2013 or 2026
            row would quietly join the training set.
    """
    for name, seasons in SPLITS.items():
        if season in seasons:
            return name
    raise ValueError(
        f"season {season} belongs to no split; the project covers "
        f"{config.FIRST_SEASON}-{config.LAST_SEASON} (see cfb.config.SEASONS)"
    )


def forward_folds(
    seasons: Sequence[int] = TRAIN_SEASONS,
    min_fit_seasons: int = 5,
) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Build forward-chaining cross-validation folds.

    Each fold fits on every season before the scored one, so the model is never fitted on
    games played after the games it is scored on. With the default training range this is
    the plan's scheme exactly: fit 2014-2018 score 2019, fit 2014-2019 score 2020, fit
    2014-2020 score 2021.

    Args:
        seasons: Seasons available for fitting, ascending.
        min_fit_seasons: How many seasons the first fold must fit on before it is allowed
            to score. Five is the plan's choice: with fewer, the early folds are scoring a
            model fitted on too little to say anything, and their noise dominates the
            average the grid is ranked by.

    Returns:
        ``(fit_seasons, score_season)`` pairs, chronological.

    Raises:
        ValueError: If the seasons are not strictly ascending, or if there are too few of
            them to make a single fold. Both mean the caller's idea of the training range
            disagrees with this module's, which is worth stopping for.
    """
    ordered = tuple(seasons)
    if list(ordered) != sorted(set(ordered)):
        raise ValueError(f"seasons must be strictly ascending with no repeats; got {ordered}")
    if len(ordered) <= min_fit_seasons:
        raise ValueError(
            f"cannot build a forward fold from {len(ordered)} seasons with "
            f"min_fit_seasons={min_fit_seasons}; need at least {min_fit_seasons + 1}"
        )
    return tuple(
        (ordered[:index], ordered[index]) for index in range(min_fit_seasons, len(ordered))
    )


def seasons_in(frame: pd.DataFrame) -> tuple[int, ...]:
    """List the distinct seasons present in a frame, ascending.

    Args:
        frame: Any frame carrying a ``season`` column.

    Returns:
        The seasons, ascending.

    Raises:
        KeyError: If there is no ``season`` column. A frame that cannot say which seasons
            it holds cannot be checked, and silently passing it would defeat every guard
            in this module.
    """
    if "season" not in frame.columns:
        raise KeyError("frame has no 'season' column; cannot check its split membership")
    return tuple(sorted(int(season) for season in frame["season"].unique()))


def assert_no_test_rows(frame: pd.DataFrame, where: str = "this frame") -> None:
    """Raise if any test-season row is present.

    Called inside the fitting functions rather than beside them, so the check sits on the
    only path to ``fit()`` instead of relying on every future caller remembering it. This
    is exit criterion 3 of the phase.

    Args:
        frame: Frame to check.
        where: What is being checked, for the error message.

    Raises:
        LeakageError: If any row belongs to a test season.
    """
    present = sorted(set(seasons_in(frame)) & set(TEST_SEASONS))
    if present:
        raise LeakageError(
            f"test-season rows reached {where}: {present}. Seasons "
            f"{TEST_SEASONS[0]}-{TEST_SEASONS[-1]} are Phase 6's held-out set and must "
            "never reach fit(). This is a leak, not a configuration problem."
        )


def assert_validation_only(frame: pd.DataFrame, where: str = "this frame") -> None:
    """Raise if the frame holds anything other than validation-season rows.

    The calibrator is part of the model, so fitting it on training rows would calibrate
    against outcomes the GBT has already memorised, and fitting it on test rows would tune
    the final predictions on the very games the headline number is computed from. Both are
    silent, and both make a real number a false claim.

    Args:
        frame: Frame to check.
        where: What is being checked, for the error message.

    Raises:
        LeakageError: If any row is outside :data:`VALIDATION_SEASONS`.
    """
    present = seasons_in(frame)
    intruders = sorted(set(present) - set(VALIDATION_SEASONS))
    if intruders:
        raise LeakageError(
            f"non-validation rows reached {where}: {intruders}. The calibrator is fitted "
            f"on season {VALIDATION_SEASONS[0]} only — it is part of the model, so the "
            "seasons it sees are seasons the model has been tuned on."
        )
    if not present:
        raise LeakageError(f"{where} received no rows at all; nothing to calibrate on")


def rows_for(frame: pd.DataFrame, seasons: Iterable[int]) -> pd.DataFrame:
    """Select the rows belonging to the given seasons.

    Args:
        frame: The model frame.
        seasons: Seasons to keep.

    Returns:
        A copy holding only those seasons, in the frame's existing order.
    """
    wanted = list(seasons)
    return frame[frame["season"].isin(wanted)].copy()
