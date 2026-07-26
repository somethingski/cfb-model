"""Fit the LightGBM classifier and the isotonic calibrator, and report what happened.

The shape of this module is set by one requirement: **the test seasons must not reach
``fit()``**, and that must be true by mechanism rather than by care. So every function
that fits anything calls a guard from :mod:`cfb.model.splits` on its own inputs, before
LightGBM or scikit-learn sees them. A future caller that assembles a frame some other way
raises instead of leaking.

The second requirement is that the numbers be *believable*, which in this project means
worse than Vegas. The report therefore scores the model and the de-vigged closing line on
the same validation games and states both. If the model wins, that is a leakage alarm and
the run fails — the prior is bug, not breakthrough (``CLAUDE.md``).

Three choices worth knowing before reading the code:

* **Nulls are handed to LightGBM as nulls.** Early-season rolling features are null by
  design (a team with no prior games this season has no rolling mean), and LightGBM sends
  missing values down whichever side of each split reduces the loss. Imputing them would
  invent a season history; dropping the rows would delete every week 1.
* **The final model is fitted on 2014-2021 for a fixed number of rounds** — the mean of
  the rounds the forward-CV folds early-stopped at — rather than early-stopping against
  2022. Early stopping on 2022 would make 2022 a fitting input, and 2022 is also the
  calibration season and the reported validation season. It would be used three times.
* **One thread, ``deterministic=True``.** LightGBM's multithreaded histogram construction
  can reorder floating-point sums between runs. On 9k rows the cost is seconds and the
  benefit is that "two runs produce identical metrics" is a property rather than a hope.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from cfb import config
from cfb.features.build import FEATURE_COLUMNS, read_frame
from cfb.ingest.schema import connect
from cfb.model import splits

SEED: int = 20140830
"""Every seeded operation in this phase uses this one value.

Chosen as the date of the first game in the data rather than as a lucky number, so it is
obvious that it was fixed once and not searched over.
"""

EXCLUDED_FROM_MODEL: tuple[str, ...] = ("fcs_opponent", "as_of")
"""Feature-store columns that exist but are not model inputs.

``fcs_opponent`` is constant zero once FCS games are filtered out, so it carries no
information; keeping it would mean shipping a feature whose value the model has never seen
vary. The *window* counts (``fcs_games_in_window_*``) are kept, because a team's prior
games may well have included an FCS opponent even when this game does not.

``as_of`` is Phase 4's audit witness — the kickoff of the latest game the row's features
read — and it is a timestamp, not a measurement of either team. It is legitimate
pre-kickoff information, but as a model input it is a thinly disguised calendar date, and
a tree that learns "games after this timestamp behave differently" has learned the season
it is in rather than anything about football. Kept in the store, kept out of the matrix.
"""

MODEL_FEATURES: tuple[str, ...] = tuple(
    name for name in FEATURE_COLUMNS if name not in EXCLUDED_FROM_MODEL
)
"""The model matrix, derived from the Phase 4 spec table rather than restated here.

A feature added in Phase 4 reaches the model automatically; a feature renamed there breaks
this import rather than silently disappearing from training.
"""

CATEGORICAL_FEATURES: tuple[str, ...] = ("season_type",)
SEASON_TYPE_CATEGORIES: tuple[str, ...] = ("regular", "postseason", "spring_regular")
"""Fixed category order.

Pandas assigns category codes by order of appearance unless told otherwise, so a frame
filtered differently would encode ``postseason`` as a different integer and the saved
model would quietly mean something else at prediction time. Stating the order makes the
encoding a property of this module rather than of whatever rows were loaded.
"""

FIXED_PARAMS: dict[str, object] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting": "gbdt",
    "verbosity": -1,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
    "bagging_freq": 0,
}
"""Parameters held constant across the search. Only the four in :data:`PARAM_GRID` vary."""

PARAM_GRID: dict[str, tuple] = {
    "num_leaves": (15, 31),
    "learning_rate": (0.03, 0.05, 0.1),
    "min_data_in_leaf": (50, 100),
    "feature_fraction": (0.7, 0.9),
}
"""The plan's grid: 24 combinations, regularised hard for a small dataset.

~7,000 training rows and 27 features is not a lot of signal, and the failure mode of a
GBT here is memorising the training seasons rather than underfitting them. Every axis is
therefore biased toward the conservative end.
"""

MAX_BOOST_ROUNDS: int = 2000
EARLY_STOPPING_ROUNDS: int = 50

CLIP_LO: float = 0.02
CLIP_HI: float = 0.98
"""Bounds applied to the calibrated probability.

Isotonic regression on ~750 games is free to map an entire tail to exactly 0 or 1, which
is a claim no model fitted on this much data can support, and which log loss punishes with
an infinity. The clip is the plan's; Platt scaling is the alternative Phase 6 may compare.
"""

PLAUSIBLE_BRIER: tuple[float, float] = (0.19, 0.21)
"""The plan's plausibility band for validation Brier.

Numbers *below* this band are the failure mode, not numbers above it.
"""


@dataclass(frozen=True)
class Candidate:
    """One point of the hyperparameter grid."""

    num_leaves: int
    learning_rate: float
    min_data_in_leaf: int
    feature_fraction: float

    def to_params(self) -> dict[str, object]:
        """Return the full LightGBM parameter dict for this candidate.

        Returns:
            :data:`FIXED_PARAMS` with this candidate's four values applied.
        """
        return {**FIXED_PARAMS, **self.to_dict()}

    def to_dict(self) -> dict[str, float | int]:
        """Return only the searched values, for the report."""
        return {
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
        }


@dataclass(frozen=True)
class FoldResult:
    """What one forward-chaining fold scored."""

    fit_seasons: tuple[int, ...]
    score_season: int
    log_loss: float
    brier: float
    best_iteration: int
    n_fit: int
    n_score: int


@dataclass(frozen=True)
class SearchResult:
    """One candidate and its average across the folds."""

    candidate: Candidate
    folds: tuple[FoldResult, ...]
    mean_log_loss: float
    mean_brier: float
    mean_best_iteration: float

    @property
    def at_grid_edge(self) -> list[str]:
        """Names of searched parameters sitting on an endpoint of their grid.

        Phase 3 established the habit: a winner at the edge of the range is evidence the
        grid is wrong, not a result. It needs one amendment here. Three of these four axes
        have only two candidate values, so *every* winner sits at their edge and the flag
        would fire on every run while meaning nothing. Only axes with an interior to miss —
        three values or more — can report an edge.
        """
        return [
            name
            for name, values in PARAM_GRID.items()
            if len(values) >= 3 and getattr(self.candidate, name) in (values[0], values[-1])
        ]


@dataclass(frozen=True)
class ValidationMetrics:
    """Scores on the validation season, model and benchmark side by side."""

    n: int
    raw_brier: float
    raw_log_loss: float
    calibrated_brier: float
    calibrated_log_loss: float
    home_rate: float
    baseline_brier: float
    n_with_line: int
    n_without_line: int
    vegas_brier: float | None
    vegas_log_loss: float | None
    model_brier_on_lined: float | None
    model_log_loss_on_lined: float | None

    @property
    def beats_vegas(self) -> bool:
        """Whether the calibrated model scored better than the line on the same games.

        True is the alarm condition. The claim of this project is calibration *approaching*
        the de-vigged closing line; a model built from public box scores and a home-grown
        Elo beating a market that has absorbed injury news, weather and line movement is
        evidence of a bug in the features, not of an edge.
        """
        if self.vegas_brier is None or self.model_brier_on_lined is None:
            return False
        return self.model_brier_on_lined < self.vegas_brier


def brier_score(outcomes: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean squared error of the predicted probability.

    Args:
        outcomes: 0/1 outcomes.
        probabilities: Predicted probability of the 1 outcome.

    Returns:
        The Brier score. Lower is better.
    """
    return float(np.mean((probabilities - outcomes) ** 2))


def log_loss_score(outcomes: np.ndarray, probabilities: np.ndarray, eps: float = 1e-15) -> float:
    """Mean negative log likelihood.

    Args:
        outcomes: 0/1 outcomes.
        probabilities: Predicted probability of the 1 outcome.
        eps: Clip applied before taking the log, so a confident miss is finite. Only the
            metric is clipped; nothing about the model or the report is.

    Returns:
        The log loss. Lower is better.
    """
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    return float(-np.mean(outcomes * np.log(clipped) + (1 - outcomes) * np.log(1 - clipped)))


def model_frame(store: pd.DataFrame) -> pd.DataFrame:
    """Reduce the feature store to the rows and encoding this phase models.

    Two things happen here and nowhere else: FBS-vs-FCS games are dropped, and
    ``season_type`` is given a fixed categorical encoding.

    FCS games are excluded from training and evaluation because the target market is FBS
    games, and because every FCS opponent shares one fixed 1200 Elo rating (``RISKS.md``
    #3 and #19) — those rows would teach the model about a fictional average opponent.
    They remain in the database, and in the rolling windows, exactly as Phase 4 built them.

    Kept pure (a frame in, a frame out) so the tests can exercise it on a hand-built frame
    without a database or a parquet file in sight.

    Args:
        store: The Phase 4 feature store, as read by :func:`cfb.features.build.read_frame`.

    Returns:
        The model frame: keys, features, and the label.

    Raises:
        ValueError: If any label is null, which would mean the store contains a game with
            no result and the Phase 4 invariant has broken; or if ``season_type`` holds a
            value this module has not been told about.
    """
    frame = store[store["fcs_opponent"] == 0].copy()
    if frame["label_home_win"].isna().any():
        raise ValueError("feature store has a null label; the store holds completed games only")

    dtype = pd.CategoricalDtype(categories=list(SEASON_TYPE_CATEGORIES), ordered=False)
    unknown = set(frame["season_type"].unique()) - set(SEASON_TYPE_CATEGORIES)
    if unknown:
        raise ValueError(
            f"unknown season_type values {sorted(unknown)}; add them to "
            "SEASON_TYPE_CATEGORIES deliberately rather than letting them encode as null"
        )
    frame["season_type"] = frame["season_type"].astype(dtype)
    return frame.reset_index(drop=True)


def load_model_frame(path: Path | None = None) -> tuple[pd.DataFrame, int]:
    """Read the feature store from disk and reduce it to the model frame.

    Args:
        path: Feature store location; defaults to :data:`cfb.config.FEATURE_STORE_PATH`.

    Returns:
        ``(model_frame, fcs_rows_excluded)``. The excluded count is returned rather than
        printed, because it belongs in the report: a row count that shrank is the kind of
        thing that should be visible in the artefact, not only in a terminal that scrolled.
    """
    store = read_frame(path)
    frame = model_frame(store)
    return frame, len(store) - len(frame)


def design_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the model inputs from a model frame.

    Args:
        frame: A frame from :func:`load_model_frame`.

    Returns:
        The columns in :data:`MODEL_FEATURES`, in that order.

    Raises:
        KeyError: If a feature is missing, which means the store and this module disagree
            about what the model is fitted on.
    """
    missing = [name for name in MODEL_FEATURES if name not in frame.columns]
    if missing:
        raise KeyError(f"model frame is missing features {missing}; rebuild with `make features`")
    return frame[list(MODEL_FEATURES)]


def outcomes(frame: pd.DataFrame) -> np.ndarray:
    """Return the 0/1 home-win labels as an array."""
    return frame["label_home_win"].to_numpy(dtype=float)


def _dataset(frame: pd.DataFrame, reference: lgb.Dataset | None = None) -> lgb.Dataset:
    """Wrap a model frame as a LightGBM dataset with the categorical feature declared."""
    return lgb.Dataset(
        design_matrix(frame),
        label=outcomes(frame),
        categorical_feature=list(CATEGORICAL_FEATURES),
        reference=reference,
        free_raw_data=False,
    )


def fit_booster(
    train: pd.DataFrame,
    params: dict[str, object],
    num_boost_round: int,
    valid: pd.DataFrame | None = None,
    early_stopping_rounds: int | None = None,
) -> lgb.Booster:
    """Fit one LightGBM booster.

    Every path to a fitted GBT goes through this function, which is why the leakage guard
    lives here: both frames are checked before LightGBM sees either of them.

    Args:
        train: Rows to fit on.
        params: Full LightGBM parameter dict.
        num_boost_round: Maximum boosting rounds.
        valid: Rows to early-stop against. None fits the full round count.
        early_stopping_rounds: Rounds without improvement before stopping.

    Returns:
        The fitted booster.

    Raises:
        LeakageError: If either frame contains a test season.
    """
    splits.assert_no_test_rows(train, "the GBT training set")
    if valid is not None:
        splits.assert_no_test_rows(valid, "the GBT early-stopping set")

    callbacks = []
    valid_sets = []
    train_set = _dataset(train)
    if valid is not None:
        valid_sets.append(_dataset(valid, reference=train_set))
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

    return lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets or None,
        callbacks=callbacks or None,
    )


def predict(booster: lgb.Booster, frame: pd.DataFrame) -> np.ndarray:
    """Predict home-win probabilities for a model frame.

    Args:
        booster: A fitted booster.
        frame: Rows to predict.

    Returns:
        Probabilities, one per row.
    """
    return np.asarray(booster.predict(design_matrix(frame), num_iteration=booster.best_iteration))


def run_folds(frame: pd.DataFrame, candidate: Candidate) -> list[FoldResult]:
    """Score one candidate by forward-chaining cross-validation.

    Args:
        frame: The model frame, training seasons and possibly more.
        candidate: The grid point to score.

    Returns:
        One result per fold, chronological.
    """
    results = []
    for fit_seasons, score_season in splits.forward_folds():
        fit_rows = splits.rows_for(frame, fit_seasons)
        score_rows = splits.rows_for(frame, [score_season])
        booster = fit_booster(
            fit_rows,
            candidate.to_params(),
            num_boost_round=MAX_BOOST_ROUNDS,
            valid=score_rows,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        probabilities = predict(booster, score_rows)
        actual = outcomes(score_rows)
        results.append(
            FoldResult(
                fit_seasons=tuple(fit_seasons),
                score_season=score_season,
                log_loss=log_loss_score(actual, probabilities),
                brier=brier_score(actual, probabilities),
                best_iteration=int(booster.best_iteration or booster.current_iteration()),
                n_fit=len(fit_rows),
                n_score=len(score_rows),
            )
        )
    return results


def candidates(grid: dict[str, tuple] = PARAM_GRID) -> list[Candidate]:
    """Expand the grid into candidates, in a fixed order.

    Args:
        grid: Parameter name to candidate values.

    Returns:
        Every combination, ordered so two runs search in the same sequence.
    """
    names = list(grid)
    return [
        Candidate(**dict(zip(names, values, strict=True))) for values in product(*grid.values())
    ]


def search(frame: pd.DataFrame, grid: dict[str, tuple] = PARAM_GRID) -> list[SearchResult]:
    """Run the forward-CV search over the whole grid.

    Args:
        frame: The model frame.
        grid: The hyperparameter grid.

    Returns:
        Every candidate, sorted by mean fold log loss, best first. Ties break toward the
        smaller ``num_leaves`` and then the smaller ``learning_rate``: on a flat objective,
        prefer the less flexible model rather than whichever point iteration reached first.
    """
    results = []
    for candidate in candidates(grid):
        folds = run_folds(frame, candidate)
        results.append(
            SearchResult(
                candidate=candidate,
                folds=tuple(folds),
                mean_log_loss=float(np.mean([fold.log_loss for fold in folds])),
                mean_brier=float(np.mean([fold.brier for fold in folds])),
                mean_best_iteration=float(np.mean([fold.best_iteration for fold in folds])),
            )
        )
    return sorted(
        results,
        key=lambda r: (r.mean_log_loss, r.candidate.num_leaves, r.candidate.learning_rate),
    )


def sensitivity(results: Sequence[SearchResult], name: str) -> list[tuple[float, float]]:
    """Best achievable mean log loss at each value of one parameter.

    A search that reports only its winner hides whether the objective has any shape. If
    log loss is flat to four decimals across an axis, the "tuned" value on that axis is
    noise, and that is worth knowing before it is written down as a finding. Carried over
    from Phase 3, where it turned out to be the more informative half of the report.

    Args:
        results: Search results.
        name: A key of :data:`PARAM_GRID`.

    Returns:
        ``(value, best_mean_log_loss)`` pairs, ordered by value.
    """
    best: dict[float, float] = {}
    for result in results:
        value = getattr(result.candidate, name)
        best[value] = min(best.get(value, float("inf")), result.mean_log_loss)
    return sorted(best.items())


def fit_calibrator(frame: pd.DataFrame, raw: np.ndarray) -> IsotonicRegression:
    """Fit the isotonic calibrator on validation-season predictions.

    The calibrator is part of the model, which is why it gets its own guard rather than
    relying on the caller having passed the right rows: fitting it on training rows would
    calibrate against games the GBT has already fitted, and fitting it on test rows would
    tune the final predictions on the very games Phase 6's headline number comes from.
    That second one would still be real code producing a real number, and the number would
    be a false claim.

    Args:
        frame: The validation rows the predictions came from.
        raw: Raw model probabilities for those rows, in the same order.

    Returns:
        The fitted calibrator.

    Raises:
        LeakageError: If ``frame`` holds any season other than the validation season.
        ValueError: If the arrays are not aligned.
    """
    splits.assert_validation_only(frame, "the calibrator fit")
    if len(frame) != len(raw):
        raise ValueError(f"frame has {len(frame)} rows but {len(raw)} predictions were given")

    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
    calibrator.fit(raw, outcomes(frame))
    return calibrator


def apply_calibrator(calibrator: IsotonicRegression, raw: np.ndarray) -> np.ndarray:
    """Calibrate raw probabilities and clip them into the supportable range.

    Args:
        calibrator: A fitted calibrator.
        raw: Raw model probabilities.

    Returns:
        Calibrated probabilities, clipped to ``[CLIP_LO, CLIP_HI]``.
    """
    return np.clip(np.asarray(calibrator.predict(raw)), CLIP_LO, CLIP_HI)


def load_benchmark(conn: sqlite3.Connection, game_ids: Sequence[int]) -> dict[int, float]:
    """Read de-vigged closing-line probabilities for the given games.

    Args:
        conn: Open connection to the built database.
        game_ids: Games to look up.

    Returns:
        ``game_id`` to ``p_home_devig`` for the games that have a line. Games with no line
        from any provider (``RISKS.md`` #10) are simply absent, and the caller reports how
        many were dropped rather than filling them in.
    """
    if not len(game_ids):
        return {}
    placeholders = ",".join("?" for _ in game_ids)
    rows = conn.execute(
        f"SELECT game_id, p_home_devig FROM vegas_benchmark WHERE game_id IN ({placeholders})",
        list(game_ids),
    ).fetchall()
    return {int(row[0]): float(row[1]) for row in rows}


def evaluate(
    validation: pd.DataFrame,
    raw: np.ndarray,
    calibrated: np.ndarray,
    benchmark: dict[int, float],
    home_rate: float,
) -> ValidationMetrics:
    """Score the model and the benchmark on the validation season.

    Args:
        validation: Validation rows.
        raw: Raw model probabilities.
        calibrated: Calibrated probabilities.
        benchmark: De-vigged line probabilities, keyed by game id.
        home_rate: The training-season home-win rate, used for the naive baseline. Taken
            from training seasons rather than from 2022 itself: a baseline that knows the
            validation season's own home-win rate is a season-level aggregate applied
            inside that season, which is the leakage pattern ``CLAUDE.md`` names.

    Returns:
        The metrics, with the Vegas comparison restricted to games that have a line.
    """
    actual = outcomes(validation)
    game_ids = validation["game_id"].to_numpy()
    has_line = np.array([int(game_id) in benchmark for game_id in game_ids])
    vegas = np.array([benchmark[int(game_id)] for game_id in game_ids[has_line]])

    return ValidationMetrics(
        n=len(validation),
        raw_brier=brier_score(actual, raw),
        raw_log_loss=log_loss_score(actual, raw),
        calibrated_brier=brier_score(actual, calibrated),
        calibrated_log_loss=log_loss_score(actual, calibrated),
        home_rate=home_rate,
        baseline_brier=brier_score(actual, np.full_like(actual, home_rate)),
        n_with_line=int(has_line.sum()),
        n_without_line=int((~has_line).sum()),
        vegas_brier=brier_score(actual[has_line], vegas) if has_line.any() else None,
        vegas_log_loss=log_loss_score(actual[has_line], vegas) if has_line.any() else None,
        model_brier_on_lined=(
            brier_score(actual[has_line], calibrated[has_line]) if has_line.any() else None
        ),
        model_log_loss_on_lined=(
            log_loss_score(actual[has_line], calibrated[has_line]) if has_line.any() else None
        ),
    )


def save_artifacts(
    booster: lgb.Booster,
    calibrator: IsotonicRegression,
    report_payload: dict,
    models_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Write the booster, the calibrator and the report.

    Args:
        booster: The fitted final model.
        calibrator: The fitted calibrator.
        report_payload: The report as JSON-ready data.
        models_dir: Destination directory; defaults to :data:`cfb.config.MODELS_DIR`.

    Returns:
        The three paths written, in the order ``(gbt, calibrator, report)``.
    """
    directory = models_dir or config.MODELS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    gbt_path = directory / config.GBT_PATH.name
    calibrator_path = directory / config.CALIBRATOR_PATH.name
    report_path = directory / config.TRAIN_REPORT_PATH.name

    booster.save_model(str(gbt_path), num_iteration=booster.best_iteration or None)
    calibrator_path.write_bytes(pickle.dumps(calibrator))
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=False) + "\n")
    return gbt_path, calibrator_path, report_path


def build_report(
    results: Sequence[SearchResult],
    final_rounds: int,
    train_rows: int,
    metrics: ValidationMetrics,
    excluded_fcs: int,
) -> dict:
    """Assemble the machine-readable training report.

    Deliberately carries no timestamp: the report is an artefact whose whole point is that
    two runs of the same code on the same data produce the same file, and a clock in it
    would make every run differ.

    Args:
        results: Search results, best first.
        final_rounds: Boosting rounds used for the final fit.
        train_rows: Rows the final model was fitted on.
        metrics: Validation metrics.
        excluded_fcs: How many FCS rows were dropped before anything was fitted.

    Returns:
        JSON-ready report data.
    """
    winner = results[0]
    return {
        "claim": (
            "Calibration approaching the de-vigged Vegas closing line. The model is not "
            "expected to beat the line; a validation score better than the benchmark is "
            "treated as a leakage alarm, not as a result."
        ),
        "splits": {
            "train": list(splits.TRAIN_SEASONS),
            "validation": list(splits.VALIDATION_SEASONS),
            "test": list(splits.TEST_SEASONS),
            "scheme": "season-forward; random splits are a bug (CLAUDE.md)",
            "forward_cv_folds": [
                {"fit": list(fit), "score": score} for fit, score in splits.forward_folds()
            ],
        },
        "data": {
            "source": str(config.FEATURE_STORE_PATH.relative_to(config.PROJECT_ROOT)),
            "fcs_rows_excluded": excluded_fcs,
            "final_fit_rows": train_rows,
            "validation_rows": metrics.n,
            "features": list(MODEL_FEATURES),
            "features_excluded": list(EXCLUDED_FROM_MODEL),
            "null_policy": (
                "Nulls are passed to LightGBM as nulls and routed by the learner. Early-"
                "season rolling features are null by design; imputing them would invent a "
                "season history and dropping the rows would delete every week 1."
            ),
        },
        "seeds": {"seed": SEED, "deterministic": True, "num_threads": 1},
        "search": {
            "grid": {name: list(values) for name, values in PARAM_GRID.items()},
            "combinations": len(results),
            "objective": "mean binary log loss over the forward-chaining folds",
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "max_boost_rounds": MAX_BOOST_ROUNDS,
            "chosen": winner.candidate.to_dict(),
            "chosen_mean_log_loss": winner.mean_log_loss,
            "chosen_mean_brier": winner.mean_brier,
            "chosen_at_grid_edge": winner.at_grid_edge,
            "grid_edge_rule": (
                "Only axes with three or more candidate values can report an edge; on a "
                "two-value axis every winner is at an edge and the flag would mean nothing."
            ),
            "sensitivity": {
                name: [[value, loss] for value, loss in sensitivity(results, name)]
                for name in PARAM_GRID
            },
            "spread_across_grid": results[-1].mean_log_loss - results[0].mean_log_loss,
            "final_boost_rounds": final_rounds,
            "final_rounds_rule": (
                "mean of the rounds the folds early-stopped at, rounded. The final model "
                "is not early-stopped against 2022: 2022 is the calibration season and the "
                "reported validation season, and using it to choose an iteration count "
                "would make it a fitting input as well."
            ),
            "folds": [
                {
                    "fit": list(fold.fit_seasons),
                    "score": fold.score_season,
                    "log_loss": fold.log_loss,
                    "brier": fold.brier,
                    "best_iteration": fold.best_iteration,
                    "n_fit": fold.n_fit,
                    "n_score": fold.n_score,
                }
                for fold in winner.folds
            ],
            "all_candidates": [
                {
                    **result.candidate.to_dict(),
                    "mean_log_loss": result.mean_log_loss,
                    "mean_brier": result.mean_brier,
                    "mean_best_iteration": result.mean_best_iteration,
                }
                for result in results
            ],
        },
        "calibration": {
            "method": "isotonic regression",
            "fitted_on": list(splits.VALIDATION_SEASONS),
            "clip": [CLIP_LO, CLIP_HI],
            "why_not_test": (
                "Calibration is part of the model. Fitting it on test data would tune the "
                "final predictions on the very games used for the headline Brier score — "
                "real code producing a false claim."
            ),
            "known_tradeoff": (
                "Isotonic on ~750 validation games can overfit at the tails, which is what "
                "the clip is for. Platt scaling is the robustness alternative Phase 6 may "
                "compare against."
            ),
        },
        "validation": {
            "season": splits.VALIDATION_SEASONS[0],
            "n": metrics.n,
            "raw_brier": metrics.raw_brier,
            "raw_log_loss": metrics.raw_log_loss,
            "calibrated_brier": metrics.calibrated_brier,
            "calibrated_log_loss": metrics.calibrated_log_loss,
            "naive_home_baseline_brier": metrics.baseline_brier,
            "naive_home_rate": metrics.home_rate,
            "plausible_band": list(PLAUSIBLE_BRIER),
            "in_plausible_band": PLAUSIBLE_BRIER[0] <= metrics.raw_brier <= PLAUSIBLE_BRIER[1],
        },
        "benchmark": {
            "source": "vegas_benchmark.p_home_devig (Phase 2), spread-derived",
            "n_with_line": metrics.n_with_line,
            "n_without_line": metrics.n_without_line,
            "vegas_brier": metrics.vegas_brier,
            "vegas_log_loss": metrics.vegas_log_loss,
            "model_brier_on_lined_games": metrics.model_brier_on_lined,
            "model_log_loss_on_lined_games": metrics.model_log_loss_on_lined,
            "model_beats_vegas": metrics.beats_vegas,
            "gap_brier": (
                None
                if metrics.vegas_brier is None or metrics.model_brier_on_lined is None
                else metrics.model_brier_on_lined - metrics.vegas_brier
            ),
        },
        "lightgbm_version": lgb.__version__,
    }


def render(report_payload: dict, results: Sequence[SearchResult]) -> str:
    """Render the run for a human reading the terminal.

    Args:
        report_payload: The report from :func:`build_report`.
        results: Search results, best first.

    Returns:
        A printable report.
    """
    search_block = report_payload["search"]
    validation = report_payload["validation"]
    benchmark = report_payload["benchmark"]
    n_folds = len(report_payload["splits"]["forward_cv_folds"])

    out = [
        "",
        f"Search: {search_block['combinations']} candidates x {n_folds} forward folds",
        "-" * 78,
    ]
    out.append(
        f"{'rank':>4}  {'leaves':>6}  {'lr':>5}  {'min_leaf':>8}  {'frac':>5}  "
        f"{'log loss':>9}  {'brier':>7}  {'rounds':>7}"
    )
    for rank, result in enumerate(results[:5], start=1):
        c = result.candidate
        out.append(
            f"{rank:>4}  {c.num_leaves:>6}  {c.learning_rate:>5.2f}  {c.min_data_in_leaf:>8}  "
            f"{c.feature_fraction:>5.1f}  {result.mean_log_loss:>9.5f}  "
            f"{result.mean_brier:>7.5f}  {result.mean_best_iteration:>7.1f}"
        )
    worst = results[-1]
    out.append(
        f"{'worst':>4}  {worst.candidate.num_leaves:>6}  {worst.candidate.learning_rate:>5.2f}  "
        f"{worst.candidate.min_data_in_leaf:>8}  {worst.candidate.feature_fraction:>5.1f}  "
        f"{worst.mean_log_loss:>9.5f}  {worst.mean_brier:>7.5f}  "
        f"{worst.mean_best_iteration:>7.1f}"
    )

    out += ["", "Winning candidate, fold by fold", "-" * 78]
    for fold in search_block["folds"]:
        out.append(
            f"  fit {fold['fit'][0]}-{fold['fit'][-1]} ({fold['n_fit']:>5} games) "
            f"-> score {fold['score']} ({fold['n_score']:>4} games): "
            f"log loss {fold['log_loss']:.5f}  brier {fold['brier']:.5f}  "
            f"stopped at {fold['best_iteration']}"
        )
    out += ["", "Sensitivity: best mean log loss at each value", "-" * 78]
    for name, pairs in search_block["sensitivity"].items():
        rendered = "  ".join(f"{value:g}:{loss:.5f}" for value, loss in pairs)
        out.append(f"  {name:<18} {rendered}")
    out.append(
        f"  whole grid spans {search_block['spread_across_grid']:.5f} in mean log loss, "
        f"best to worst"
    )
    if search_block["chosen_at_grid_edge"]:
        edges = ", ".join(search_block["chosen_at_grid_edge"])
        out.append(f"  note: winner sits at a grid edge on {edges} (axes with an interior)")

    out += ["", f"Validation ({validation['season']}, {validation['n']} games)", "-" * 78]
    out.append(
        f"  raw model         brier {validation['raw_brier']:.5f}   "
        f"log loss {validation['raw_log_loss']:.5f}"
    )
    out.append(
        f"  calibrated        brier {validation['calibrated_brier']:.5f}   "
        f"log loss {validation['calibrated_log_loss']:.5f}"
    )
    out.append(
        f"  naive home rate   brier {validation['naive_home_baseline_brier']:.5f}   "
        f"(predicts {validation['naive_home_rate']:.4f} every game)"
    )

    out += [
        "",
        f"Benchmark, on the {benchmark['n_with_line']} games that have a closing line",
        "-" * 78,
    ]
    if benchmark["vegas_brier"] is None:
        out.append("  no benchmark rows found; run `make benchmark`")
    else:
        out.append(
            f"  de-vigged line    brier {benchmark['vegas_brier']:.5f}   "
            f"log loss {benchmark['vegas_log_loss']:.5f}"
        )
        out.append(
            f"  calibrated model  brier {benchmark['model_brier_on_lined_games']:.5f}   "
            f"log loss {benchmark['model_log_loss_on_lined_games']:.5f}"
        )
        out.append(
            f"  gap (model - line): {benchmark['gap_brier']:+.5f} brier"
            f"   [{benchmark['n_without_line']} games have no line and are excluded]"
        )

    out += ["", "Verdict", "-" * 78]
    if benchmark["model_beats_vegas"]:
        out.append(
            "  MODEL BEATS VEGAS ON VALIDATION. Exit criterion 2 treats this as a leakage "
            "alarm, not a result. Stop and investigate the features."
        )
    else:
        out.append("  Model is worse than the line, as expected.")
    if validation["in_plausible_band"]:
        out.append(
            f"  Raw validation brier {validation['raw_brier']:.5f} is inside the plan's "
            f"plausibility band {PLAUSIBLE_BRIER}."
        )
    else:
        direction = "below" if validation["raw_brier"] < PLAUSIBLE_BRIER[0] else "above"
        out.append(
            f"  Raw validation brier {validation['raw_brier']:.5f} is {direction} the plan's "
            f"band {PLAUSIBLE_BRIER}. Below the band is the failure mode; review before "
            "accepting."
        )
    return "\n".join(out)


def run(
    frame: pd.DataFrame,
    conn: sqlite3.Connection,
    excluded_fcs: int = 0,
) -> tuple[lgb.Booster, IsotonicRegression, dict, list[SearchResult]]:
    """Do the whole phase: search, fit, calibrate, evaluate.

    Args:
        frame: The model frame from :func:`load_model_frame`.
        conn: Open connection to the built database, for the Phase 2 benchmark.
        excluded_fcs: How many FCS rows the model frame dropped, for the report.

    Returns:
        ``(booster, calibrator, report_payload, search_results)``.
    """
    train = splits.rows_for(frame, splits.TRAIN_SEASONS)
    validation = splits.rows_for(frame, splits.VALIDATION_SEASONS)

    results = search(train)
    winner = results[0]
    final_rounds = max(1, int(round(winner.mean_best_iteration)))

    booster = fit_booster(train, winner.candidate.to_params(), num_boost_round=final_rounds)
    raw = predict(booster, validation)
    calibrator = fit_calibrator(validation, raw)
    calibrated = apply_calibrator(calibrator, raw)

    benchmark = load_benchmark(conn, validation["game_id"].tolist())
    home_rate = float(outcomes(train).mean())
    metrics = evaluate(validation, raw, calibrated, benchmark, home_rate)

    payload = build_report(results, final_rounds, len(train), metrics, excluded_fcs)
    return booster, calibrator, payload, results


def main(argv: Sequence[str] | None = None) -> int:
    """Train the model, calibrate it, and write the artefacts.

    Args:
        argv: Command-line arguments; None reads ``sys.argv``.

    Returns:
        Process exit status. Non-zero if the model beat the benchmark on validation, so
        that ``make train`` fails rather than a leaking model being treated as finished.
    """
    parser = argparse.ArgumentParser(description="Train and calibrate the Phase 5 model.")
    parser.add_argument(
        "--dry-run", action="store_true", help="run the search and report without writing models/"
    )
    args = parser.parse_args(argv)

    if not config.FEATURE_STORE_PATH.exists():
        raise SystemExit(f"no feature store at {config.FEATURE_STORE_PATH}; run `make audit` first")
    if not config.DB_PATH.exists():
        raise SystemExit(f"no database at {config.DB_PATH}; run `make ingest` first")

    frame, excluded_fcs = load_model_frame()
    print(
        f"model frame: {len(frame)} FBS-vs-FBS games, {excluded_fcs} FCS games excluded "
        f"({splits.TRAIN_SEASONS[0]}-{splits.TEST_SEASONS[-1]}), "
        f"{len(MODEL_FEATURES)} features"
    )
    print(
        f"  train {splits.TRAIN_SEASONS[0]}-{splits.TRAIN_SEASONS[-1]}  "
        f"validation {splits.VALIDATION_SEASONS[0]}  "
        f"test {splits.TEST_SEASONS[0]}-{splits.TEST_SEASONS[-1]} (not touched)"
    )

    conn = connect(config.DB_PATH)
    try:
        booster, calibrator, payload, results = run(frame, conn, excluded_fcs)
    finally:
        conn.close()

    print(render(payload, results))

    if args.dry_run:
        print(f"\ndry run: nothing written to {config.MODELS_DIR}")
    else:
        gbt_path, calibrator_path, report_path = save_artifacts(booster, calibrator, payload)
        print(f"\nwrote {gbt_path}\n      {calibrator_path}\n      {report_path}")

    if payload["benchmark"]["model_beats_vegas"]:
        print(
            "\nFAILED: the model scored better than the de-vigged closing line on "
            "validation. Per the phase plan this is a leakage alarm; the run is failed "
            "deliberately so nothing downstream treats it as a finished model."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
