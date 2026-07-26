"""Score the held-out seasons. The only code in this project that reads a test label.

Everything upstream was built so that this module could be run once and believed. What it
reports is therefore not "how good is the model" but **the gap between the model and the
de-vigged closing line**, stated as a gap, on one game set that every system is scored on.

Three properties the code enforces rather than assumes:

* **One frame, five probability vectors.** The evaluation frame is built once, sorted once,
  and every system predicts over it in that order. "Scored on the identical game set" is
  then true by construction instead of by four filters that have to agree.
* **The frame is test seasons only.** :func:`cfb.model.splits.assert_test_only` runs on it
  before anything is scored. A 2022 row here would score the model on games the calibrator
  was fitted on, and would move the headline in the flattering direction invisibly.
* **The model that is scored is the model Phase 5 fitted.** The booster and calibrator are
  reloaded from disk and checked against ``train_report.json``'s validation Brier before the
  test seasons are touched at all. A re-fitted or half-written artefact fails here rather
  than quietly producing a different headline.

And one property it enforces about the *result*: if the model scores better than the closing
line, the run fails with a non-zero exit code. Per ``CLAUDE.md`` the prior for that outcome
is a bug, not a breakthrough, and a printed warning is not a gate. The check is on the
pooled test set, which is the only comparison here with the sample size to carry it; a
single season beating the line is reported loudly and does not fail the run.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from cfb import config
from cfb.elo import pipeline as elo_pipeline
from cfb.eval import plots
from cfb.eval.metrics import (
    ReliabilityBin,
    brier_score,
    log_loss_score,
    reliability_table,
    resolution,
)
from cfb.eval.metrics import skill_fraction as skill_fraction_of
from cfb.eval.systems import EloOnly, apply_platt, fit_elo_scale, fit_platt, naive_home_rate
from cfb.ingest.schema import connect
from cfb.model import splits
from cfb.model.train import MODEL_FEATURES, apply_calibrator, load_model_frame, outcomes, predict
from cfb.vegas.benchmark import MONEYLINE_MISMATCH_SQL

TRIPWIRE_TOLERANCE: float = 0.002
"""How far below the line's Brier the model may score before the run fails.

The plan's figure. It is a tolerance for floating-point and sampling slop around a dead
heat, not a licence to be slightly better than the market: genuinely beating a closing line
on 2,000+ games with public box scores is extraordinary-claims territory, and the default
explanation is a leak.
"""

MODEL = "model"
MODEL_RAW = "model_raw"
MODEL_PLATT = "model_platt"
VEGAS = "vegas"
NAIVE = "naive"
ELO = "elo_only"

HEADLINE_SYSTEMS: tuple[str, ...] = (MODEL, VEGAS, NAIVE, ELO)
"""The plan's four systems, in the order they are reported.

The two secondary series (:data:`MODEL_RAW`, :data:`MODEL_PLATT`) are scored on the same
games and reported separately: ``RISKS.md`` #25 requires the raw and calibrated numbers to
be stated apart rather than pooled, because the calibrator's Phase 5 improvement was
measured in-sample and these seasons are the first out-of-sample test of it.
"""

SYSTEM_LABELS: dict[str, str] = {
    MODEL: "model (calibrated)",
    MODEL_RAW: "model (raw)",
    MODEL_PLATT: "model (Platt)",
    VEGAS: "de-vigged closing line",
    NAIVE: "naive home baseline",
    ELO: "Elo only",
}

LIMITATIONS: tuple[str, ...] = (
    "Not a betting tool. The model scores worse than the de-vigged closing line, and the "
    "line already has vig on top of it. Nothing here has been evaluated against closing-line "
    "value, bet sizing, or transaction costs, and no such claim is made.",
    "The benchmark is spread-derived, not a traded price. `p_home_devig` converts a closing "
    "spread through a normal margin model with a fixed sigma fitted on 2014-2021 (RISKS #15, "
    "#17). It is a good yardstick, not the market's own probability.",
    "One model, one split, one run. There is no repeated-seed study and no confidence "
    "interval on any gap reported here. Differences of a thousandth of a Brier point between "
    "systems are not resolvable at this sample size.",
    "Hyperparameters are barely tuned in any meaningful sense. The 24-point grid spans "
    "0.0053 in mean forward-CV log loss best to worst (RISKS #23); the chosen point is "
    "defensible, not discovered.",
    "The isotonic calibrator did not generalise. Fitted on a single season of 776 games "
    "(RISKS #4, #25), it improved 2022 in-sample and made every held-out season worse than "
    "leaving the model raw; the measured cost and the mechanism are in `results_table.md`. "
    "The headline is still the calibrated model, because re-picking the raw one on the "
    "strength of its test-season score would be selecting a model on the test set.",
    "A one-parameter logistic of the Elo difference scores better than the shipped "
    "calibrated model on these seasons. The 26-feature booster earns its keep only before "
    "calibration, and only narrowly, which says the rolling box-score features carry little "
    "signal that Elo does not already carry.",
    "Elo is cold-started in 2014 and every FCS opponent shares one fixed 1200 rating "
    "(RISKS #18, #19, #22), so the rating features are noisiest exactly where the training "
    "data begins and are inflated for teams fresh off an FCS game.",
    "Plays are reconstructed as rushing attempts plus pass attempts, not observed "
    "(RISKS #21), so every yards-per-play and pace feature rests on a derived denominator "
    "that will not match every published play count.",
    "Team strength is represented by Elo, rest, and rolling box-score rates only. There are "
    "no injuries, no weather, no personnel, no travel distance, and no market information of "
    "any kind — that last one by rule, since the line is the benchmark.",
)
"""What a reader has to know before quoting a number from this phase.

Written here rather than in the model card's template so the list is one reviewable object
that the tests can check is non-empty and that Phase 7 can import instead of paraphrasing.
"""


@dataclass(frozen=True)
class Scored:
    """One system's scores on one slice of games.

    Attributes:
        system: System key.
        n: Games scored.
        brier: Brier score, lower better.
        log_loss: Mean negative log likelihood, lower better.
    """

    system: str
    n: int
    brier: float
    log_loss: float


@dataclass(frozen=True)
class Slice:
    """A named set of games with every system scored on it.

    Attributes:
        name: ``"overall"`` or a season, rendered.
        n: Games in the slice — the same for every system by construction.
        scores: System key to its scores.
    """

    name: str
    n: int
    scores: dict[str, Scored]

    def gap_brier(self) -> float:
        """Model Brier minus the line's. Positive means the model is worse, as expected."""
        return self.scores[MODEL].brier - self.scores[VEGAS].brier

    def gap_log_loss(self) -> float:
        """Model log loss minus the line's."""
        return self.scores[MODEL].log_loss - self.scores[VEGAS].log_loss

    def skill_fraction(self) -> float:
        """Fraction of the naive-to-line distance the model closed on this slice."""
        return skill_fraction_of(
            self.scores[NAIVE].brier, self.scores[MODEL].brier, self.scores[VEGAS].brier
        )


def load_artifacts(models_dir: Path | None = None) -> tuple[lgb.Booster, IsotonicRegression]:
    """Load the Phase 5 booster and calibrator.

    Args:
        models_dir: Directory holding the artefacts; defaults to
            :data:`cfb.config.MODELS_DIR`.

    Returns:
        ``(booster, calibrator)``.

    Raises:
        SystemExit: If either artefact is missing. Evaluation cannot be run against a model
            that does not exist, and re-training silently here would mean ``make evaluate``
            could change the thing it is measuring.
    """
    directory = models_dir or config.MODELS_DIR
    gbt_path = directory / config.GBT_PATH.name
    calibrator_path = directory / config.CALIBRATOR_PATH.name
    for path in (gbt_path, calibrator_path):
        if not path.exists():
            raise SystemExit(f"no model artefact at {path}; run `make train` first")
    return lgb.Booster(model_file=str(gbt_path)), pickle.loads(calibrator_path.read_bytes())


def assert_artifacts_match_report(
    booster: lgb.Booster,
    calibrator: IsotonicRegression,
    frame: pd.DataFrame,
    report_path: Path | None = None,
    tolerance: float = 1e-12,
) -> float:
    """Check the reloaded model reproduces Phase 5's published validation Brier.

    This runs before any test season is read, and it is the cheapest possible guard against
    the two ways this phase could quietly measure the wrong thing: an artefact re-fitted
    after ``train_report.json`` was written, or one written by a different code path. Both
    would produce a perfectly plausible headline attached to a model nobody reviewed.

    Uses the validation season, which this phase is allowed to read and which the report
    states a number for.

    Args:
        booster: The reloaded booster.
        calibrator: The reloaded calibrator.
        frame: The full model frame.
        report_path: Location of ``train_report.json``; defaults to config.
        tolerance: Allowed absolute difference. Effectively exact — Phase 5 pinned
            LightGBM to one thread and ``deterministic=True`` precisely so this could be.

    Returns:
        The recomputed calibrated validation Brier.

    Raises:
        SystemExit: If the report is missing or the numbers disagree.
    """
    path = report_path or config.TRAIN_REPORT_PATH
    if not path.exists():
        raise SystemExit(f"no training report at {path}; run `make train` first")
    expected = json.loads(path.read_text())["validation"]["calibrated_brier"]

    validation = splits.rows_for(frame, splits.VALIDATION_SEASONS)
    recomputed = brier_score(
        outcomes(validation), apply_calibrator(calibrator, predict(booster, validation))
    )
    if abs(recomputed - expected) > tolerance:
        raise SystemExit(
            f"the model in {config.MODELS_DIR} does not reproduce the validation Brier in "
            f"{path.name}: recomputed {recomputed!r}, report says {expected!r}. The artefacts "
            "and the report disagree, so one of them is stale. Re-run `make train` before "
            "evaluating rather than publishing a number from an unreviewed model."
        )
    return recomputed


def load_benchmark_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    """Read the Phase 2 benchmark, including the moneyline view and its mismatch flag.

    The mismatch rule is imported from :mod:`cfb.vegas.benchmark` rather than restated, so
    the rows this phase excludes from the moneyline robustness check are exactly the rows
    ``RISKS.md`` #16 is about.

    Args:
        conn: Open connection to the built database.

    Returns:
        One row per benchmarked game: ``game_id``, ``p_home_devig``, ``p_home_moneyline``,
        ``provider``, ``source_type``, ``moneyline_mismatch``.
    """
    return pd.read_sql(
        f"""
        SELECT
            b.game_id,
            b.p_home_devig,
            b.p_home_moneyline,
            b.provider,
            b.source_type,
            CASE WHEN {MONEYLINE_MISMATCH_SQL} THEN 1 ELSE 0 END AS moneyline_mismatch
        FROM vegas_benchmark b
        """,
        conn,
    )


def evaluation_frame(frame: pd.DataFrame, benchmark: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Build the one frame every system is scored on.

    The intersection the plan specifies: test seasons, FBS-vs-FBS (already true of the model
    frame), and a Vegas benchmark present. Sorted by ``(start_date, game_id)`` so the row
    order is a property of the data rather than of whatever order parquet happened to return,
    which is what makes "same games, same order" checkable.

    Args:
        frame: The full model frame from :func:`cfb.model.train.load_model_frame`.
        benchmark: The frame from :func:`load_benchmark_frame`.

    Returns:
        ``(evaluation_frame, dropped_for_missing_line)``.

    Raises:
        LeakageError: If the result holds any non-test season, or no rows at all.
    """
    test = splits.rows_for(frame, splits.TEST_SEASONS)
    splits.assert_test_only(test, "the evaluation frame")

    merged = test.merge(benchmark, on="game_id", how="left")
    has_line = merged["p_home_devig"].notna()
    dropped = int((~has_line).sum())

    evaluation = merged[has_line].sort_values(["start_date", "game_id"]).reset_index(drop=True)
    splits.assert_test_only(evaluation, "the evaluation frame")
    return evaluation, dropped


def build_predictions(
    evaluation: pd.DataFrame,
    booster: lgb.Booster,
    calibrator: IsotonicRegression,
    platt,
    elo: EloOnly,
    home_rate: float,
) -> dict[str, np.ndarray]:
    """Predict every system over the evaluation frame, in the frame's order.

    Args:
        evaluation: The evaluation frame.
        booster: Phase 5 booster.
        calibrator: Phase 5 isotonic calibrator.
        platt: The Platt alternative from :func:`cfb.eval.systems.fit_platt`.
        elo: The fitted Elo-only baseline.
        home_rate: Training-season home-win rate for the naive baseline.

    Returns:
        System key to probabilities, every array the same length as ``evaluation`` and in
        its order.
    """
    raw = predict(booster, evaluation)
    return {
        MODEL: apply_calibrator(calibrator, raw),
        MODEL_RAW: raw,
        MODEL_PLATT: apply_platt(platt, raw),
        VEGAS: evaluation["p_home_devig"].to_numpy(dtype=float),
        NAIVE: np.full(len(evaluation), home_rate, dtype=float),
        ELO: elo.probabilities(evaluation),
    }


def score_slice(
    name: str, evaluation: pd.DataFrame, predictions: dict[str, np.ndarray], mask: np.ndarray | None
) -> Slice:
    """Score every system on one slice of the evaluation frame.

    Args:
        name: Slice name, for the report.
        evaluation: The evaluation frame.
        predictions: System key to probabilities over the whole frame.
        mask: Boolean row mask, or None for the whole frame.

    Returns:
        The slice with every system scored on exactly the same rows.
    """
    selector = np.ones(len(evaluation), dtype=bool) if mask is None else mask
    actual = outcomes(evaluation)[selector]
    scores = {
        system: Scored(
            system=system,
            n=int(selector.sum()),
            brier=brier_score(actual, probabilities[selector]),
            log_loss=log_loss_score(actual, probabilities[selector]),
        )
        for system, probabilities in predictions.items()
    }
    return Slice(name=name, n=int(selector.sum()), scores=scores)


def score_all(evaluation: pd.DataFrame, predictions: dict[str, np.ndarray]) -> list[Slice]:
    """Score the pooled test period and each test season.

    Args:
        evaluation: The evaluation frame.
        predictions: System key to probabilities.

    Returns:
        The overall slice first, then one per season ascending. Per-season columns are what
        expose the calibration drift ``RISKS.md`` #4 predicts; pooling would average it away.
    """
    seasons = evaluation["season"].to_numpy()
    slices = [score_slice("overall", evaluation, predictions, None)]
    for season in sorted(set(int(value) for value in seasons)):
        slices.append(score_slice(str(season), evaluation, predictions, seasons == season))
    return slices


def moneyline_robustness(
    evaluation: pd.DataFrame, predictions: dict[str, np.ndarray]
) -> dict[str, object]:
    """Score the model against the moneyline view of the line, where one exists.

    The plan's robustness bullet asked to split the benchmark into moneyline-sourced and
    spread-fallback games. That split does not exist: the Phase 2 decision made
    ``p_home_devig`` spread-derived for *every* season deliberately, so the benchmark's
    construction would not change at a split boundary (``RISKS.md`` #15). What can be asked
    instead — and is the same question underneath — is whether the *conclusion* depends on
    that choice: score the model against the de-vigged moneyline on the games that carry
    one, and see whether the gap moves.

    The ``RISKS.md`` #16 mismatch rows are excluded, as that risk row instructs: for those
    games CFBD's home/away moneyline assignment does not agree with the perspective the
    spread is stated from, and averaging over a known sign error would be worse than
    reporting the exclusion.

    Args:
        evaluation: The evaluation frame.
        predictions: System key to probabilities.

    Returns:
        A JSON-ready block, with ``n`` zero and the scores None if no game qualifies.
    """
    usable = (
        evaluation["p_home_moneyline"].notna() & (evaluation["moneyline_mismatch"] == 0)
    ).to_numpy()
    excluded_mismatch = int((evaluation["moneyline_mismatch"] == 1).sum())
    no_moneyline = int(evaluation["p_home_moneyline"].isna().sum())

    if not usable.any():
        return {
            "n": 0,
            "games_without_a_moneyline": no_moneyline,
            "games_excluded_as_mismatched": excluded_mismatch,
            "note": "no test game carries a usable moneyline; nothing to compare",
        }

    actual = outcomes(evaluation)[usable]
    moneyline = evaluation["p_home_moneyline"].to_numpy(dtype=float)[usable]
    model = predictions[MODEL][usable]
    spread = predictions[VEGAS][usable]
    return {
        "n": int(usable.sum()),
        "games_without_a_moneyline": no_moneyline,
        "games_excluded_as_mismatched": excluded_mismatch,
        "moneyline_brier": brier_score(actual, moneyline),
        "moneyline_log_loss": log_loss_score(actual, moneyline),
        "spread_benchmark_brier": brier_score(actual, spread),
        "spread_benchmark_log_loss": log_loss_score(actual, spread),
        "model_brier": brier_score(actual, model),
        "model_log_loss": log_loss_score(actual, model),
        "gap_vs_moneyline_brier": brier_score(actual, model) - brier_score(actual, moneyline),
        "gap_vs_spread_brier": brier_score(actual, model) - brier_score(actual, spread),
        "why_not_the_plan_s_split": (
            "The plan asked for moneyline-sourced vs spread-fallback benchmark games. Phase "
            "2 made p_home_devig spread-derived for every season on purpose (RISKS #15), so "
            "that split has no members. This asks the same question the other way round."
        ),
    }


CONFIDENT_THRESHOLD: float = 0.85
"""Where the "confident predictions" diagnostic starts, on the raw model's scale.

Chosen because it is where isotonic's top plateau begins to bite, not by searching for the
threshold that tells the best story. Stated so it can be checked.
"""


def calibrator_comparison(
    overall: Slice, predictions: dict[str, np.ndarray], evaluation: pd.DataFrame
) -> dict[str, object]:
    """Compare the shipped isotonic calibrator against raw and Platt, out of sample.

    This is the first honest test of ``RISKS.md`` #4 and #25. Phase 5 fitted isotonic on the
    776 games of 2022 and measured its improvement on those same games, and both risk rows
    say in as many words that the number is a fit and not an estimate.

    The diagnostic that explains whatever it finds is :func:`cfb.eval.metrics.resolution`:
    isotonic is a step function, so it collapses distinct model outputs onto shared fitted
    values, and with 776 games to fit on there are not many steps to go around. Brier notices
    that; it does not explain it.

    **What this block must not be used for.** If the shipped calibrator turns out to have
    hurt on the test seasons, the answer is not to drop it and re-report. Choosing between
    model variants on test-season scores is selecting a model on the test set — the same
    mistake as tuning on it, arriving one phase later and wearing a lab coat. The shipped
    model stays the shipped model, the cost is reported, and any change is a decision for a
    later phase fitted on data this one is not allowed to touch.

    Args:
        overall: The pooled slice.
        predictions: System key to probabilities.
        evaluation: The evaluation frame.

    Returns:
        A JSON-ready block.
    """
    actual = outcomes(evaluation)
    confident = predictions[MODEL_RAW] > CONFIDENT_THRESHOLD
    return {
        "note": (
            "Isotonic is the shipped calibrator; Platt is the alternative RISKS #4 and #25 "
            "name; raw is the booster with neither. All three were fitted on training data "
            "and 2022 only, so these are their first out-of-sample scores."
        ),
        "raw_brier": overall.scores[MODEL_RAW].brier,
        "isotonic_brier": overall.scores[MODEL].brier,
        "platt_brier": overall.scores[MODEL_PLATT].brier,
        "isotonic_cost_vs_raw": overall.scores[MODEL].brier - overall.scores[MODEL_RAW].brier,
        "platt_cost_vs_raw": overall.scores[MODEL_PLATT].brier - overall.scores[MODEL_RAW].brier,
        "in_sample_isotonic_gain_2022": (
            "Phase 5 measured 0.20244 -> 0.19510 on the season it was fitted on. That is a "
            "fit, not an estimate, and RISKS #25 said so before this was run."
        ),
        "resolution": {
            system: resolution(predictions[system]) for system in (MODEL_RAW, MODEL, MODEL_PLATT)
        },
        "confident_predictions": {
            "threshold": CONFIDENT_THRESHOLD,
            "n": int(confident.sum()),
            "observed_frequency": float(actual[confident].mean()) if confident.any() else None,
            "raw_mean_prediction": float(predictions[MODEL_RAW][confident].mean())
            if confident.any()
            else None,
            "isotonic_mean_prediction": float(predictions[MODEL][confident].mean())
            if confident.any()
            else None,
        },
        "must_not_be_used_to_reselect": (
            "Switching the shipped model on the strength of these numbers would be selecting "
            "a model on the test set. The shipped model stays shipped; the cost is reported."
        ),
    }


def season_coverage(evaluation: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Report what each test season actually contains, rather than assuming it is complete.

    The phase plan assumes 2025 is fully landed in CFBD "as it should be by now", and asks
    that if late-season games are missing the run report actual coverage instead of silently
    scoring a shrunken test set. A season that quietly lost its bowls would still produce a
    perfectly plausible Brier — on an easier set of games, since the postseason is where the
    mismatches and the neutral sites are.

    Args:
        evaluation: The evaluation frame.

    Returns:
        Per season: games by season type, the week range, and the last kickoff.
    """
    coverage = {}
    for season, rows in evaluation.groupby("season"):
        by_type = rows["season_type"].value_counts().to_dict()
        regular = rows[rows["season_type"] == "regular"]
        coverage[str(int(season))] = {
            "n": int(len(rows)),
            "by_season_type": {str(name): int(count) for name, count in by_type.items()},
            "regular_weeks": [int(regular["week"].min()), int(regular["week"].max())]
            if len(regular)
            else None,
            "last_kickoff": str(rows["start_date"].max())[:10],
        }
    return coverage


def reliability(
    evaluation: pd.DataFrame, predictions: dict[str, np.ndarray], mask: np.ndarray | None = None
) -> dict[str, list[ReliabilityBin]]:
    """Build reliability tables for the systems worth plotting.

    Args:
        evaluation: The evaluation frame.
        predictions: System key to probabilities.
        mask: Row mask, or None for the whole frame.

    Returns:
        System key to its bins. The naive baseline is omitted: it predicts one constant, so
        every game lands in one bin and the "curve" is a single point that says nothing
        about calibration.
    """
    selector = np.ones(len(evaluation), dtype=bool) if mask is None else mask
    actual = outcomes(evaluation)[selector]
    return {
        system: reliability_table(actual, predictions[system][selector])
        for system in (MODEL, VEGAS, ELO)
    }


def tripwire(slices: Sequence[Slice], tolerance: float = TRIPWIRE_TOLERANCE) -> dict[str, object]:
    """Decide whether the pooled result is a leakage alarm.

    Args:
        slices: All slices, the pooled one first.
        tolerance: How far below the line's Brier the model may score.

    Returns:
        A JSON-ready verdict block. ``tripped`` True means the run must fail.
        ``seasons_beating_vegas`` is reported and never fails the run.
    """
    overall = slices[0]
    model = overall.scores[MODEL].brier
    vegas = overall.scores[VEGAS].brier
    return {
        "rule": "model Brier must be >= the line's Brier minus the tolerance",
        "tolerance": tolerance,
        "model_brier": model,
        "vegas_brier": vegas,
        "gap_brier": model - vegas,
        "n": overall.n,
        "tripped": bool(model < vegas - tolerance),
        "scope": (
            "The pooled test period only. A single season beating the line is reported and "
            "not failed: ~800 games cannot separate a leak from noise, and a gate that fires "
            "on noise gets loosened, which is how gates die."
        ),
        "seasons_beating_vegas": [
            current.name
            for current in slices[1:]
            if current.scores[MODEL].brier < current.scores[VEGAS].brier - tolerance
        ],
    }


def build_metrics(
    slices: Sequence[Slice],
    evaluation: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    elo: EloOnly,
    home_rate: float,
    dropped_no_line: int,
    validation_brier: float,
) -> dict:
    """Assemble the machine-readable results.

    Carries no timestamp, following ``train_report.json``: two runs on the same data must
    produce a file that ``diff`` says is identical, and a clock would defeat that.

    Args:
        slices: Overall slice first, then per season.
        evaluation: The evaluation frame.
        predictions: System key to probabilities.
        elo: The fitted Elo-only baseline.
        home_rate: The naive baseline's constant.
        dropped_no_line: Games dropped for having no Vegas benchmark.
        validation_brier: The recomputed Phase 5 validation Brier, as an artefact witness.

    Returns:
        JSON-ready results.
    """
    overall = slices[0]
    providers = evaluation["provider"].value_counts().to_dict()
    return {
        "claim": (
            "Calibration approaching the de-vigged Vegas closing line. The model does not "
            "beat the line and is not a betting tool. A test score better than the benchmark "
            "is treated as a leakage alarm, not as a result."
        ),
        "headline": {
            "statement": (
                f"On {overall.n} held-out games the model's Brier is "
                f"{overall.scores[MODEL].brier:.4f} against the de-vigged closing line's "
                f"{overall.scores[VEGAS].brier:.4f} — short of the line by "
                f"{overall.gap_brier():.4f}."
            ),
            "gap_brier": overall.gap_brier(),
            "gap_log_loss": overall.gap_log_loss(),
            "skill_fraction": overall.skill_fraction(),
            "skill_fraction_definition": (
                "(brier_naive - brier_model) / (brier_naive - brier_vegas)"
            ),
        },
        "game_set": {
            "seasons": list(splits.TEST_SEASONS),
            "n": overall.n,
            "n_per_season": {
                current.name: current.n for current in slices if current.name != "overall"
            },
            "rule": (
                "test seasons, FBS vs FBS, and a Vegas benchmark present — the intersection. "
                "Every system is scored on this one frame, in one order."
            ),
            "dropped_for_missing_line": dropped_no_line,
            "fcs_games_excluded_upstream": (
                "FBS-vs-FCS games are excluded by the Phase 5 model frame (RISKS #3), before "
                "this phase sees the data."
            ),
            "benchmark_providers": {str(name): int(count) for name, count in providers.items()},
            "season_coverage": season_coverage(evaluation),
        },
        "systems": {
            MODEL: "Phase 5 LightGBM, isotonic-calibrated on 2022, output clipped to [0.02, 0.98]",
            MODEL_RAW: "the same booster without the calibrator",
            MODEL_PLATT: "the same booster with a Platt calibrator fitted on 2022 (RISKS #25)",
            VEGAS: "vegas_benchmark.p_home_devig — spread-derived, multiplicative de-vig",
            NAIVE: (
                f"constant {home_rate!r}, the home-win rate on training seasons "
                f"{splits.TRAIN_SEASONS[0]}-{splits.TRAIN_SEASONS[-1]} — not on the test "
                "seasons, which would be a season-level aggregate applied inside its own season"
            ),
            ELO: (
                f"1 / (1 + 10 ** (-(elo_diff + hfa) / scale)), hfa {elo.hfa:g} from the frozen "
                f"elo_params.json, scale {elo.scale:.2f} fitted on "
                f"{elo.fitted_on[0]}-{elo.fitted_on[-1]} only"
            ),
        },
        "elo_only_fit": {
            "scale": elo.scale,
            "definitional_scale": 400.0,
            "hfa": elo.hfa,
            "fitted_on": list(elo.fitted_on),
            "train_log_loss": elo.train_log_loss,
        },
        "overall": {
            system: {"n": score.n, "brier": score.brier, "log_loss": score.log_loss}
            for system, score in overall.scores.items()
        },
        "per_season": {
            current.name: {
                "n": current.n,
                "gap_brier": current.gap_brier(),
                "gap_log_loss": current.gap_log_loss(),
                "skill_fraction": current.skill_fraction(),
                "systems": {
                    system: {"brier": score.brier, "log_loss": score.log_loss}
                    for system, score in current.scores.items()
                },
            }
            for current in slices
            if current.name != "overall"
        },
        "reliability": {
            "n_bins": len(next(iter(reliability(evaluation, predictions).values()))),
            "binning": "equal-count bins over the sorted predicted probability",
            "overall": {
                system: [
                    {
                        "lo": current.lo,
                        "hi": current.hi,
                        "n": current.n,
                        "mean_predicted": current.mean_predicted,
                        "empirical": current.empirical,
                    }
                    for current in bins
                ]
                for system, bins in reliability(evaluation, predictions).items()
            },
        },
        "robustness": {
            "moneyline_benchmark": moneyline_robustness(evaluation, predictions),
            "calibrator_comparison": calibrator_comparison(overall, predictions, evaluation),
        },
        "tripwire": tripwire(slices),
        "artifact_witness": {
            "validation_season": splits.VALIDATION_SEASONS[0],
            "recomputed_calibrated_brier": validation_brier,
            "note": (
                "Recomputed from the reloaded booster and calibrator and checked against "
                "train_report.json before any test label was read. The model scored here is "
                "the model Phase 5 fitted."
            ),
        },
        "limitations": list(LIMITATIONS),
        "lightgbm_version": lgb.__version__,
    }


def render(metrics: dict, slices: Sequence[Slice]) -> str:
    """Render the run for a human reading the terminal.

    Args:
        metrics: The results from :func:`build_metrics`.
        slices: Overall slice first, then per season.

    Returns:
        A printable report.
    """
    overall = slices[0]
    game_set = metrics["game_set"]
    out = [
        "",
        f"Evaluation: {game_set['n']} games, seasons "
        f"{game_set['seasons'][0]}-{game_set['seasons'][-1]}, FBS vs FBS, all with a line",
        f"  {game_set['dropped_for_missing_line']} games dropped for having no benchmark",
        "-" * 78,
        f"{'system':<24}{'brier':>10}{'log loss':>11}",
    ]
    for system in HEADLINE_SYSTEMS + (MODEL_RAW, MODEL_PLATT):
        score = overall.scores[system]
        out.append(f"  {SYSTEM_LABELS[system]:<22}{score.brier:>10.5f}{score.log_loss:>11.5f}")

    out += [
        "",
        "Per season",
        "-" * 78,
        f"{'season':<10}{'n':>6}"
        + "".join(f"{SYSTEM_LABELS[system]:>24}" for system in (MODEL, VEGAS))
        + f"{'gap':>10}{'skill':>9}",
    ]
    for current in slices[1:]:
        out.append(
            f"{current.name:<10}{current.n:>6}"
            f"{current.scores[MODEL].brier:>24.5f}{current.scores[VEGAS].brier:>24.5f}"
            f"{current.gap_brier():>+10.5f}{current.skill_fraction():>9.3f}"
        )

    headline = metrics["headline"]
    out += [
        "",
        "Headline",
        "-" * 78,
        f"  {headline['statement']}",
        f"  gap in log loss: {headline['gap_log_loss']:+.5f}",
        f"  the model closes {headline['skill_fraction'] * 100:.1f}% of the distance between "
        "the naive baseline and the line",
    ]

    verdict = metrics["tripwire"]
    out += ["", "Verdict", "-" * 78]
    if verdict["tripped"]:
        out.append(
            "  TRIPWIRE: the model scored better than the de-vigged closing line on "
            f"{verdict['n']} held-out games (gap {verdict['gap_brier']:+.5f}). Per CLAUDE.md "
            "the prior is a bug, not a breakthrough. Investigate the features before "
            "recording this anywhere."
        )
    else:
        out.append(
            f"  Model is short of the line by {verdict['gap_brier']:+.5f} Brier, which is the "
            "expected and intended result."
        )
    beaters = verdict["seasons_beating_vegas"]
    if beaters:
        out.append(
            f"  NOTE: the model scored better than the line in {', '.join(beaters)}. This does "
            "not fail the run (one season is too few games to separate a leak from noise), "
            "but it is worth a look before these numbers are quoted."
        )
    return "\n".join(out)


def results_table_markdown(metrics: dict, slices: Sequence[Slice]) -> str:
    """Render ``results/results_table.md``.

    Args:
        metrics: The results from :func:`build_metrics`.
        slices: Overall slice first, then per season.

    Returns:
        The markdown document.
    """
    overall = slices[0]
    game_set = metrics["game_set"]
    headline = metrics["headline"]
    money = metrics["robustness"]["moneyline_benchmark"]
    elo_fit = metrics["elo_only_fit"]

    lines = [
        f"# Results — held-out seasons {game_set['seasons'][0]}–{game_set['seasons'][-1]}",
        "",
        "> **The headline is a gap.** " + headline["statement"],
        ">",
        "> The model does not beat the market and is not a betting tool. A score better than "
        "the closing line here would be treated as evidence of leakage, not as a result.",
        "",
        f"Generated by `make evaluate`. Machine-readable copy: "
        f"[`metrics.json`]({config.EVAL_METRICS_PATH.name}).",
        "",
        "## The game set",
        "",
        f"- **{game_set['n']} games**: {game_set['rule']}",
        "- Per season: "
        + ", ".join(f"{season} — {count}" for season, count in game_set["n_per_season"].items()),
        f"- **{game_set['dropped_for_missing_line']} games dropped** for having no closing line "
        "from any provider.",
        f"- {game_set['fcs_games_excluded_upstream']}",
        "- Benchmark provider on these games: "
        + ", ".join(f"{name} ({count})" for name, count in game_set["benchmark_providers"].items())
        + ". The Phase 2 ladder switches provider inside the test period; the counts are here "
        "so that is visible rather than buried.",
        "",
        "### Coverage, reported rather than assumed",
        "",
        "The most recent season being incomplete in the source would shrink the test set "
        "toward the easier, earlier part of a year, and would still produce a plausible "
        "number. So what each season actually contains is printed rather than trusted:",
        "",
        "| Season | Regular | Postseason | Regular weeks | Last kickoff |",
        "|---|---|---|---|---|",
    ]
    for season, coverage in game_set["season_coverage"].items():
        weeks = coverage["regular_weeks"]
        lines.append(
            f"| {season} | {coverage['by_season_type'].get('regular', 0)} | "
            f"{coverage['by_season_type'].get('postseason', 0)} | "
            f"{weeks[0]}–{weeks[1]} | {coverage['last_kickoff']} |"
        )
    lines += [
        "",
        "## Overall",
        "",
        "| System | Brier | Log loss |",
        "|---|---|---|",
    ]
    for system in HEADLINE_SYSTEMS:
        score = overall.scores[system]
        lines.append(f"| {SYSTEM_LABELS[system]} | {score.brier:.4f} | {score.log_loss:.4f} |")
    lines += [
        "",
        f"**Gap (model − line): {headline['gap_brier']:+.4f} Brier, "
        f"{headline['gap_log_loss']:+.4f} log loss.**",
        "",
        f"The model closes **{headline['skill_fraction'] * 100:.1f}%** of the distance between "
        "the naive home baseline and the de-vigged closing line "
        f"(`{headline['skill_fraction_definition']}`). That ratio is two small differences "
        "divided by each other, so it moves a lot on little; it is not a percentage of "
        "anything anyone can bet on.",
        "",
        "## Per season",
        "",
        "Reported separately rather than pooled, because a calibrator fitted on one season "
        "(2022) drifting on later ones is a named risk (`RISKS.md` #4), and pooling is exactly "
        "what would hide it.",
        "",
        "| Season | n | Model | Line | Naive | Elo only | Gap | Skill |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for current in slices[1:]:
        lines.append(
            f"| {current.name} | {current.n} | {current.scores[MODEL].brier:.4f} | "
            f"{current.scores[VEGAS].brier:.4f} | {current.scores[NAIVE].brier:.4f} | "
            f"{current.scores[ELO].brier:.4f} | {current.gap_brier():+.4f} | "
            f"{current.skill_fraction() * 100:.1f}% |"
        )

    calibration = metrics["robustness"]["calibrator_comparison"]
    confident = calibration["confident_predictions"]
    isotonic_resolution = calibration["resolution"][MODEL]
    lines += [
        "",
        "## Calibration did not generalise",
        "",
        "Phase 5's isotonic calibrator was fitted on 2022 and its improvement there "
        "(0.2024 → 0.1951 Brier) was measured on the season it was fitted on. `RISKS.md` #25 "
        "said before this was run that the number is a fit and not an estimate. These seasons "
        "are the test, and the calibrator **cost** "
        f"{calibration['isotonic_cost_vs_raw']:+.4f} Brier against leaving the model raw — "
        "in all three seasons, in the same direction.",
        "",
        "| Season | Raw | Isotonic (shipped) | Platt |",
        "|---|---|---|---|",
    ]
    for current in slices:
        lines.append(
            f"| {current.name} | {current.scores[MODEL_RAW].brier:.4f} | "
            f"{current.scores[MODEL].brier:.4f} | {current.scores[MODEL_PLATT].brier:.4f} |"
        )
    lines += [
        "",
        "The mechanism is visible rather than inferred. Isotonic regression is a step "
        f"function, and one fitted on 776 games has few steps: it collapses the booster's "
        f"{calibration['resolution'][MODEL_RAW]['n_distinct']} distinct outputs on these "
        f"games into **{isotonic_resolution['n_distinct']}**, with "
        f"{isotonic_resolution['largest_plateau']} games landing on the single value "
        f"{isotonic_resolution['largest_plateau_value']:.4f}. The damage concentrates where "
        f"the steps are widest — on the {confident['n']} games the raw model called at over "
        f"{confident['threshold']:.2f}, the home team won "
        f"{confident['observed_frequency'] * 100:.1f}% of the time, the raw model said "
        f"{confident['raw_mean_prediction'] * 100:.1f}%, and the calibrated model said "
        f"{confident['isotonic_mean_prediction'] * 100:.1f}%. Platt scaling, with two "
        f"parameters instead of an arbitrary monotone step function, costs "
        f"{calibration['platt_cost_vs_raw']:+.4f} — less, and still worse than raw.",
        "",
        "**This does not change the shipped model.** "
        "The headline above is the calibrated model because that is what Phase 5 shipped, and "
        "picking the raw model now — on the strength of its test-season score — would be "
        "selecting a model on the test set, which is the same mistake as tuning on it, one "
        "phase later. Whether to keep the calibrator is a decision for a later phase, made on "
        "data this one is not allowed to touch.",
        "",
        "## The Elo baseline is close",
        "",
        f"The Elo-only baseline scores {overall.scores[ELO].brier:.4f}, which is **better than "
        f"the shipped calibrated model** ({overall.scores[MODEL].brier:.4f}) and worse than the "
        f"raw booster ({overall.scores[MODEL_RAW].brier:.4f}). Twenty-six features and a "
        "gradient-boosted tree buy very little over one rating difference and a fitted logistic "
        "scale, and after the calibrator they buy less than nothing. Stated here because it is "
        "the most useful thing in this document: it says the rolling box-score features are "
        "carrying almost no independent signal, which is a finding about the features and a "
        "starting point for the next iteration, not a defect in the evaluation.",
        "",
        "For what it is worth as a sanity check, the scale fitted on training seasons came out "
        f"at {elo_fit['scale']:.1f} Elo points per decade of odds, against the definitional "
        f"{elo_fit['definitional_scale']:.0f} the ratings are built under — the rating system "
        "and its probability reading agree.",
    ]

    lines += [
        "",
        "## Reliability",
        "",
        f"![reliability curves]({config.RELIABILITY_PLOT_PATH.name})",
        "",
        f"{metrics['reliability']['n_bins']} equal-count bins — "
        f"{metrics['reliability']['binning']}. Equal-count rather than equal-width because "
        "predictions pile up in the middle: equal-width bins would put a handful of games in "
        "the tail bins and draw them the same size as bins holding hundreds. Bin counts are "
        "annotated on the plot for the same reason.",
        "",
        "## Robustness: the benchmark itself",
        "",
    ]
    if money["n"]:
        lines += [
            f"The benchmark is spread-derived for every season by design (`RISKS.md` #15). "
            f"On the **{money['n']}** test games that also carry a two-sided moneyline "
            f"(excluding {money['games_excluded_as_mismatched']} rows flagged by `RISKS.md` #16 "
            f"and {money['games_without_a_moneyline']} with no moneyline), the de-vigged "
            "moneyline is a second reading of the same market:",
            "",
            "| Benchmark | Brier | Log loss | Gap to model |",
            "|---|---|---|---|",
            f"| spread-derived (shipped) | {money['spread_benchmark_brier']:.5f} | "
            f"{money['spread_benchmark_log_loss']:.5f} | {money['gap_vs_spread_brier']:+.5f} |",
            f"| moneyline-derived | {money['moneyline_brier']:.5f} | "
            f"{money['moneyline_log_loss']:.5f} | {money['gap_vs_moneyline_brier']:+.5f} |",
            "",
            "The two readings of the market differ by "
            f"{abs(money['gap_vs_spread_brier'] - money['gap_vs_moneyline_brier']):.6f} Brier in "
            "the gap they produce, which is four orders of magnitude below the gap itself. The "
            "conclusion does not depend on how the benchmark was built.",
            "",
            money["why_not_the_plan_s_split"],
        ]
    else:
        lines.append(str(money.get("note", "no moneyline comparison available")))

    lines += [
        "",
        "## Limitations",
        "",
    ]
    lines += [f"- {limitation}" for limitation in LIMITATIONS]
    lines += [
        "",
        "See [`model_card.md`](model_card.md) for intended use, and `RISKS.md` for the full "
        "risk register these are drawn from.",
        "",
    ]
    return "\n".join(lines)


def model_card_markdown(metrics: dict, slices: Sequence[Slice]) -> str:
    """Render ``results/model_card.md``.

    Args:
        metrics: The results from :func:`build_metrics`.
        slices: Overall slice first, then per season.

    Returns:
        The markdown document.
    """
    overall = slices[0]
    game_set = metrics["game_set"]
    headline = metrics["headline"]
    lines = [
        "# Model card — cfb-model v0.1.0",
        "",
        "## Intended use",
        "",
        "**Educational and portfolio use only.** This model exists to demonstrate a "
        "leakage-resistant modelling pipeline on a domain where a hard external benchmark "
        "exists. It is **not a betting tool**, it does not beat the market, and it has never "
        "been evaluated for that purpose — no closing-line value, no bet sizing, no "
        "transaction costs.",
        "",
        "The claim, stated in full: **calibration approaching the de-vigged Vegas closing "
        f"line**. On held-out seasons the model is short of that line by "
        f"{headline['gap_brier']:.4f} Brier.",
        "",
        "## Data",
        "",
        f"- Source: [CollegeFootballData.com](https://collegefootballdata.com/), seasons "
        f"{config.FIRST_SEASON}–{config.LAST_SEASON}.",
        "- One row per completed game. FBS-vs-FCS games are excluded from modelling and "
        "scoring; they still feed Elo and the rolling windows.",
        "- Nothing is imputed. Missing lines, missing box scores and the one cancelled game "
        "are excluded explicitly and recorded in `RISKS.md`.",
        "",
        "## Features",
        "",
        f"{len(MODEL_FEATURES)} inputs, all as-of-kickoff: "
        + ", ".join(f"`{name}`" for name in MODEL_FEATURES)
        + ".",
        "",
        "**No market information is a feature.** The betting line is the benchmark, never an "
        "input, and the feature builder is scanned mechanically for market terms as part of "
        "the Phase 4 audit. Every column is documented in `FEATURES.md`, generated from the "
        "code.",
        "",
        "## Split scheme",
        "",
        f"Season-forward and immutable: **train {splits.TRAIN_SEASONS[0]}–"
        f"{splits.TRAIN_SEASONS[-1]}, validation {splits.VALIDATION_SEASONS[0]}, test "
        f"{splits.TEST_SEASONS[0]}–{splits.TEST_SEASONS[-1]}**. Hyperparameters were chosen by "
        "forward-chaining cross-validation; the isotonic calibrator was fitted on the "
        "validation season alone. Random splits are a bug in this project, not a choice.",
        "",
        "Test seasons were untouched until this evaluation: the guards run inside `fit()` "
        "rather than beside it, and the feature store passes a leakage audit that recomputes "
        "a sample of rows from a database truncated at each game's kickoff.",
        "",
        "## Evaluation",
        "",
        f"**{game_set['n']} held-out games**: test seasons, FBS vs FBS, and a closing line "
        f"present — the intersection. {game_set['dropped_for_missing_line']} games were "
        "dropped for having no line from any provider. Every system below is scored on that "
        "one frame, in one order, so the comparison is between systems and not between "
        "game sets.",
        "",
        "| System | Brier | Log loss |",
        "|---|---|---|",
    ]
    for system in HEADLINE_SYSTEMS:
        score = overall.scores[system]
        lines.append(f"| {SYSTEM_LABELS[system]} | {score.brier:.4f} | {score.log_loss:.4f} |")
    lines += [
        "",
        "Per season:",
        "",
        "| Season | n | Model Brier | Line Brier | Gap |",
        "|---|---|---|---|---|",
    ]
    for current in slices[1:]:
        lines.append(
            f"| {current.name} | {current.n} | {current.scores[MODEL].brier:.4f} | "
            f"{current.scores[VEGAS].brier:.4f} | {current.gap_brier():+.4f} |"
        )
    lines += [
        "",
        "Full table and reliability curves: "
        f"[`results_table.md`]({config.RESULTS_TABLE_PATH.name}).",
        "",
        "## Limitations",
        "",
    ]
    lines += [f"- {limitation}" for limitation in LIMITATIONS]
    lines += [
        "",
        "## The statement that matters",
        "",
        "**This model does not beat the market.** It lands between a naive home-field "
        f"baseline and the de-vigged closing line, closer to the line, and short of it by "
        f"{headline['gap_brier']:.4f} Brier over {game_set['n']} games. That is the intended "
        "result. `make evaluate` fails with a non-zero exit code if the model ever scores "
        "better than the line, because on public box-score data the explanation for that "
        "would be a leak.",
        "",
    ]
    return "\n".join(lines)


def write_artifacts(
    metrics: dict,
    slices: Sequence[Slice],
    evaluation: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    results_dir: Path | None = None,
) -> list[Path]:
    """Write the four committed artefacts.

    Args:
        metrics: The results from :func:`build_metrics`.
        slices: Overall slice first, then per season.
        evaluation: The evaluation frame.
        predictions: System key to probabilities.
        results_dir: Destination; defaults to :data:`cfb.config.RESULTS_DIR`.

    Returns:
        The paths written.
    """
    directory = results_dir or config.RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)

    table_path = directory / config.RESULTS_TABLE_PATH.name
    card_path = directory / config.MODEL_CARD_PATH.name
    metrics_path = directory / config.EVAL_METRICS_PATH.name
    plot_path = directory / config.RELIABILITY_PLOT_PATH.name

    table_path.write_text(results_table_markdown(metrics, slices))
    card_path.write_text(model_card_markdown(metrics, slices))
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=False) + "\n")
    plots.reliability_figure(evaluation, predictions, plot_path)
    return [table_path, card_path, metrics_path, plot_path]


def run(frame: pd.DataFrame, conn: sqlite3.Connection, models_dir: Path | None = None):
    """Do the whole phase: load, build the frame, fit the baselines, score, assemble.

    Args:
        frame: The full model frame from :func:`cfb.model.train.load_model_frame`.
        conn: Open connection to the built database.
        models_dir: Where the Phase 5 artefacts live.

    Returns:
        ``(metrics, slices, evaluation, predictions)``.
    """
    booster, calibrator = load_artifacts(models_dir)
    validation_brier = assert_artifacts_match_report(booster, calibrator, frame)

    train = splits.rows_for(frame, splits.TRAIN_SEASONS)
    validation = splits.rows_for(frame, splits.VALIDATION_SEASONS)
    elo_params, _ = elo_pipeline.load_params()
    elo = fit_elo_scale(train, elo_params.hfa)
    home_rate = naive_home_rate(train)
    platt = fit_platt(validation, predict(booster, validation))

    evaluation, dropped = evaluation_frame(frame, load_benchmark_frame(conn))
    predictions = build_predictions(evaluation, booster, calibrator, platt, elo, home_rate)
    slices = score_all(evaluation, predictions)

    metrics = build_metrics(
        slices, evaluation, predictions, elo, home_rate, dropped, validation_brier
    )
    return metrics, slices, evaluation, predictions


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate the model on the held-out seasons and write the artefacts.

    Args:
        argv: Command-line arguments; None reads ``sys.argv``.

    Returns:
        Process exit status. Non-zero if the model beat the benchmark on the pooled test
        set, so ``make evaluate`` fails rather than a leaking model's numbers being written
        into a README.
    """
    parser = argparse.ArgumentParser(description="Evaluate the model on the held-out seasons.")
    parser.add_argument(
        "--dry-run", action="store_true", help="score and report without writing results/"
    )
    args = parser.parse_args(argv)

    if not config.FEATURE_STORE_PATH.exists():
        raise SystemExit(f"no feature store at {config.FEATURE_STORE_PATH}; run `make audit` first")
    if not config.DB_PATH.exists():
        raise SystemExit(f"no database at {config.DB_PATH}; run `make ingest` first")

    frame, excluded_fcs = load_model_frame()
    print(
        f"model frame: {len(frame)} FBS-vs-FBS games, {excluded_fcs} FCS games excluded; "
        f"evaluating seasons {splits.TEST_SEASONS[0]}-{splits.TEST_SEASONS[-1]}"
    )

    conn = connect(config.DB_PATH)
    try:
        metrics, slices, evaluation, predictions = run(frame, conn)
    finally:
        conn.close()

    print(render(metrics, slices))

    if args.dry_run:
        print(f"\ndry run: nothing written to {config.RESULTS_DIR}")
    else:
        written = write_artifacts(metrics, slices, evaluation, predictions)
        print("\nwrote " + "\n      ".join(str(path) for path in written))

    if metrics["tripwire"]["tripped"]:
        print(
            "\nFAILED: the model scored better than the de-vigged closing line on the "
            "held-out seasons. The run is failed deliberately. Investigate the features for "
            "leakage before treating this as a result — do not loosen the tolerance."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
