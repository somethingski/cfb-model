"""Evaluation-side invariants: one game set, no fitted seasons in it, and a gate that fires.

Phase 6 is the only code allowed to read a test label, so its failure modes are the mirror
image of Phase 5's. The three that matter:

* a train or validation season slipping into the frame the headline is computed from,
* the four systems being scored on subtly different games,
* the leakage tripwire being a comment rather than a gate.

Each gets a poisoned-input test, because a check that has only ever passed is not evidence.
The synthetic frame keeps these fast and makes a failure mean the code is wrong rather than
the data having moved; the integration tests at the bottom check the real artefacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfb import config
from cfb.eval import evaluate, systems
from cfb.model import splits, train
from cfb.model.splits import LeakageError
from tests.test_train import synthetic_store


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """The synthetic store, reduced to a model frame."""
    return train.model_frame(synthetic_store())


def benchmark_for(frame: pd.DataFrame, missing: set[int] | None = None) -> pd.DataFrame:
    """Build a synthetic Phase 2 benchmark frame for the given games.

    Args:
        frame: The model frame.
        missing: Game ids to leave without a line, simulating ``RISKS.md`` #10.

    Returns:
        A frame shaped like :func:`cfb.eval.evaluate.load_benchmark_frame`'s output.
    """
    absent = missing or set()
    rows = []
    for index, row in enumerate(frame.itertuples()):
        if row.game_id in absent:
            continue
        devig = 0.5 + 0.2 * np.tanh(row.elo_diff / 200.0)
        rows.append(
            {
                "game_id": row.game_id,
                "p_home_devig": float(devig),
                "p_home_moneyline": float(devig) + 0.01 if index % 3 else None,
                "provider": "Bovada",
                "source_type": "spread",
                "moneyline_mismatch": int(index % 97 == 0),
            }
        )
    return pd.DataFrame(rows)


# --- The evaluation frame -----------------------------------------------------


def test_the_evaluation_frame_holds_test_seasons_only(frame):
    evaluation, _ = evaluate.evaluation_frame(frame, benchmark_for(frame))
    assert splits.seasons_in(evaluation) == splits.TEST_SEASONS
    assert not set(splits.seasons_in(evaluation)) & set(splits.TRAIN_SEASONS)
    assert not set(splits.seasons_in(evaluation)) & set(splits.VALIDATION_SEASONS)


def test_a_training_season_in_the_evaluation_frame_is_a_hard_failure(frame):
    """The poisoned case for the split guard.

    Built by handing ``assert_test_only`` a frame that ``evaluation_frame`` would never
    produce, which is the point: the guard has to catch a caller that assembles the rows
    some other way, not just the one path through this module.
    """
    poisoned = pd.concat(
        [
            splits.rows_for(frame, splits.TEST_SEASONS),
            splits.rows_for(frame, [splits.TRAIN_SEASONS[-1]]),
        ]
    )
    with pytest.raises(LeakageError, match="non-test rows"):
        splits.assert_test_only(poisoned, "the evaluation frame")


def test_a_validation_season_in_the_evaluation_frame_is_a_hard_failure(frame):
    poisoned = pd.concat(
        [
            splits.rows_for(frame, splits.TEST_SEASONS),
            splits.rows_for(frame, splits.VALIDATION_SEASONS),
        ]
    )
    with pytest.raises(LeakageError, match="non-test rows"):
        splits.assert_test_only(poisoned, "the evaluation frame")


def test_an_empty_evaluation_frame_raises_rather_than_scoring_nothing(frame):
    with pytest.raises(LeakageError, match="no rows at all"):
        splits.assert_test_only(frame.iloc[0:0], "the evaluation frame")


def test_games_without_a_line_are_dropped_and_counted(frame):
    test_rows = splits.rows_for(frame, splits.TEST_SEASONS)
    absent = set(test_rows["game_id"].tolist()[:7])
    evaluation, dropped = evaluate.evaluation_frame(frame, benchmark_for(frame, missing=absent))
    assert dropped == 7
    assert not set(evaluation["game_id"]) & absent
    assert evaluation["p_home_devig"].notna().all()


def test_the_evaluation_frame_is_ordered_by_kickoff(frame):
    evaluation, _ = evaluate.evaluation_frame(frame, benchmark_for(frame))
    ordered = evaluation.sort_values(["start_date", "game_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(evaluation, ordered)


# --- One game set, every system -----------------------------------------------


def predictions_for(evaluation: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fake predictions for every system, aligned to the frame."""
    n = len(evaluation)
    rng = np.random.default_rng(5)
    return {
        evaluate.MODEL: np.clip(rng.uniform(0.05, 0.95, n), 0.02, 0.98),
        evaluate.MODEL_RAW: np.clip(rng.uniform(0.05, 0.95, n), 0.02, 0.98),
        evaluate.MODEL_PLATT: np.clip(rng.uniform(0.05, 0.95, n), 0.02, 0.98),
        evaluate.VEGAS: evaluation["p_home_devig"].to_numpy(dtype=float),
        evaluate.NAIVE: np.full(n, 0.575),
        evaluate.ELO: np.clip(rng.uniform(0.05, 0.95, n), 0.02, 0.98),
    }


def test_every_system_is_scored_on_the_same_games_in_the_same_order(frame):
    """Exit criterion 2, made mechanical.

    The systems cannot disagree about the game set because there is only one frame and one
    order; this asserts that the structure actually delivers that rather than assuming it.
    """
    evaluation, _ = evaluate.evaluation_frame(frame, benchmark_for(frame))
    predictions = predictions_for(evaluation)

    assert {len(values) for values in predictions.values()} == {len(evaluation)}

    slices = evaluate.score_all(evaluation, predictions)
    for current in slices:
        counts = {score.n for score in current.scores.values()}
        assert counts == {current.n}, f"{current.name}: systems scored different game counts"

    per_season = sum(current.n for current in slices[1:])
    assert per_season == slices[0].n, "the season slices must partition the pooled slice"


def test_every_headline_system_is_actually_scored(frame):
    evaluation, _ = evaluate.evaluation_frame(frame, benchmark_for(frame))
    slices = evaluate.score_all(evaluation, predictions_for(evaluation))
    assert set(evaluate.HEADLINE_SYSTEMS) <= set(slices[0].scores)


def test_a_slice_scores_only_its_own_rows(frame):
    """A season slice must not quietly widen to the whole frame."""
    evaluation, _ = evaluate.evaluation_frame(frame, benchmark_for(frame))
    predictions = predictions_for(evaluation)
    season = splits.TEST_SEASONS[0]
    mask = (evaluation["season"] == season).to_numpy()
    current = evaluate.score_slice(str(season), evaluation, predictions, mask)
    assert current.n == int(mask.sum())
    assert current.n < len(evaluation)

    expected = evaluate.brier_score(
        train.outcomes(evaluation)[mask], predictions[evaluate.MODEL][mask]
    )
    assert current.scores[evaluate.MODEL].brier == pytest.approx(expected)


# --- The leakage tripwire -----------------------------------------------------


def slice_with(model_brier: float, vegas_brier: float, name: str = "overall") -> evaluate.Slice:
    """Build a slice with the two Brier scores the tripwire looks at."""
    return evaluate.Slice(
        name=name,
        n=2398,
        scores={
            evaluate.MODEL: evaluate.Scored(evaluate.MODEL, 2398, model_brier, 0.5),
            evaluate.VEGAS: evaluate.Scored(evaluate.VEGAS, 2398, vegas_brier, 0.5),
        },
    )


def test_the_tripwire_stays_quiet_when_the_model_is_worse_than_the_line():
    verdict = evaluate.tripwire([slice_with(0.191, 0.176)])
    assert verdict["tripped"] is False
    assert verdict["gap_brier"] == pytest.approx(0.015)


def test_the_tripwire_fires_when_the_model_beats_the_line():
    """The poisoned case. Without this, the gate is a comment."""
    verdict = evaluate.tripwire([slice_with(0.170, 0.176)])
    assert verdict["tripped"] is True
    assert verdict["gap_brier"] < 0


def test_the_tripwire_tolerates_a_dead_heat_but_not_a_real_win():
    """The tolerance is slop around a tie, not a licence to be slightly better."""
    tolerance = evaluate.TRIPWIRE_TOLERANCE
    assert evaluate.tripwire([slice_with(0.176 - tolerance / 2, 0.176)])["tripped"] is False
    assert evaluate.tripwire([slice_with(0.176 - tolerance * 2, 0.176)])["tripped"] is True


def test_a_single_season_beating_the_line_is_reported_and_does_not_fail_the_run():
    """Confirmed scope: ~800 games cannot separate a leak from noise."""
    verdict = evaluate.tripwire(
        [
            slice_with(0.191, 0.176),
            slice_with(0.170, 0.176, name="2024"),
            slice_with(0.199, 0.182, name="2025"),
        ]
    )
    assert verdict["tripped"] is False
    assert verdict["seasons_beating_vegas"] == ["2024"]


# --- What Phase 6 is allowed to fit -------------------------------------------


def test_the_elo_scale_fit_refuses_a_test_season(frame):
    poisoned = splits.rows_for(frame, [*splits.TRAIN_SEASONS, splits.TEST_SEASONS[0]])
    with pytest.raises(LeakageError, match="Elo-only scale fit"):
        systems.fit_elo_scale(poisoned, hfa=50.0)


def test_the_naive_baseline_refuses_a_test_season(frame):
    poisoned = splits.rows_for(frame, [*splits.TRAIN_SEASONS, splits.TEST_SEASONS[0]])
    with pytest.raises(LeakageError, match="naive home-rate baseline"):
        systems.naive_home_rate(poisoned)


def test_the_platt_calibrator_refuses_anything_but_the_validation_season(frame):
    train_rows = splits.rows_for(frame, splits.TRAIN_SEASONS)
    with pytest.raises(LeakageError, match="Platt calibrator fit"):
        systems.fit_platt(train_rows, np.full(len(train_rows), 0.5))


def test_the_platt_calibrator_refuses_misaligned_predictions(frame):
    validation = splits.rows_for(frame, splits.VALIDATION_SEASONS)
    with pytest.raises(ValueError, match="predictions were given"):
        systems.fit_platt(validation, np.full(len(validation) - 1, 0.5))


def test_the_naive_baseline_is_the_training_rate_not_the_test_rate(frame):
    """A baseline that knew the test seasons' own home rate would flatter itself."""
    train_rows = splits.rows_for(frame, splits.TRAIN_SEASONS)
    test_rows = splits.rows_for(frame, splits.TEST_SEASONS)
    rate = systems.naive_home_rate(train_rows)
    assert rate == pytest.approx(float(train.outcomes(train_rows).mean()))
    assert rate != pytest.approx(float(train.outcomes(test_rows).mean()))


def test_the_elo_scale_fit_recovers_a_scale_it_was_given(frame):
    """Generate outcomes from a known scale and check the fit finds it.

    Fits on training seasons of the synthetic frame with labels redrawn from an Elo logistic
    at scale 250, which is far from both the search midpoint and the definitional 400, so
    landing near it cannot be an accident of where the grid starts.
    """
    train_rows = splits.rows_for(frame, splits.TRAIN_SEASONS).copy()
    truth = systems.elo_probabilities(train_rows, scale=250.0, hfa=50.0)
    rng = np.random.default_rng(3)
    train_rows["label_home_win"] = (rng.uniform(size=len(train_rows)) < truth).astype(int)

    fitted = systems.fit_elo_scale(train_rows, hfa=50.0)
    assert fitted.scale == pytest.approx(250.0, rel=0.15)


def test_elo_probabilities_drop_home_field_at_a_neutral_site():
    frame = pd.DataFrame({"elo_diff": [0.0, 0.0], "neutral_site": [0, 1]})
    hosted, neutral = systems.elo_probabilities(frame, scale=400.0, hfa=50.0)
    assert neutral == pytest.approx(0.5), "a neutral-site coin flip gets no home boost"
    assert hosted > 0.5


def test_elo_probabilities_refuse_a_non_positive_scale():
    frame = pd.DataFrame({"elo_diff": [100.0], "neutral_site": [0]})
    with pytest.raises(ValueError, match="scale must be positive"):
        systems.elo_probabilities(frame, scale=0.0, hfa=50.0)


# --- The built artefacts ------------------------------------------------------


@pytest.mark.integration
def test_the_saved_model_still_reproduces_the_phase_5_validation_brier():
    """The artefact witness, run as a test as well as inside the phase.

    If this fails, ``models/`` and ``models/train_report.json`` disagree and one of them is
    stale — which would mean the published headline came from a model nobody reviewed.
    """
    if not config.GBT_PATH.exists() or not config.FEATURE_STORE_PATH.exists():
        pytest.skip("no trained model or feature store; run `make train` first")
    booster, calibrator = evaluate.load_artifacts()
    model_rows, _ = train.load_model_frame()
    recomputed = evaluate.assert_artifacts_match_report(booster, calibrator, model_rows)
    assert 0.15 < recomputed < 0.25


@pytest.mark.integration
def test_the_written_results_cover_every_test_season_on_one_game_set():
    if not config.EVAL_METRICS_PATH.exists():
        pytest.skip("no results; run `make evaluate` first")
    import json

    metrics = json.loads(config.EVAL_METRICS_PATH.read_text())
    assert metrics["game_set"]["seasons"] == list(splits.TEST_SEASONS)
    assert set(metrics["per_season"]) == {str(season) for season in splits.TEST_SEASONS}
    assert sum(metrics["game_set"]["n_per_season"].values()) == metrics["game_set"]["n"]
    for system in evaluate.HEADLINE_SYSTEMS:
        assert metrics["overall"][system]["n"] == metrics["game_set"]["n"]


@pytest.mark.integration
def test_every_test_season_landed_complete_including_its_postseason():
    """The plan's assumption about 2025, turned into a check.

    A season missing its bowls and playoff would shrink the test set toward the easier part
    of a year and still produce a plausible Brier. The postseason is small and its size is
    stable across seasons, so a season carrying far fewer than its neighbours is the shape a
    partial download would take.
    """
    if not config.EVAL_METRICS_PATH.exists():
        pytest.skip("no results; run `make evaluate` first")
    import json

    coverage = json.loads(config.EVAL_METRICS_PATH.read_text())["game_set"]["season_coverage"]
    assert set(coverage) == {str(season) for season in splits.TEST_SEASONS}
    postseason = {
        season: block["by_season_type"].get("postseason", 0) for season, block in coverage.items()
    }
    assert all(count >= 40 for count in postseason.values()), (
        f"a test season is short of postseason games: {postseason}. Report the actual "
        "coverage rather than scoring a silently shrunken test set."
    )
    for season, block in coverage.items():
        last_week = block["regular_weeks"][1]
        assert last_week >= 15, f"{season} regular season ends at week {last_week}"
        assert block["n"] > 700, f"{season} holds only {block['n']} games"


@pytest.mark.integration
def test_the_published_result_does_not_beat_the_line():
    """The gate, asserted on the committed artefact rather than only inside the run.

    If this ever goes red, the fix is in the features. Not in the tolerance.
    """
    if not config.EVAL_METRICS_PATH.exists():
        pytest.skip("no results; run `make evaluate` first")
    import json

    verdict = json.loads(config.EVAL_METRICS_PATH.read_text())["tripwire"]
    assert verdict["tripped"] is False, (
        "the published results show the model beating the de-vigged closing line. Per "
        "CLAUDE.md the prior is a bug, not a breakthrough — investigate the features."
    )
    assert verdict["gap_brier"] > 0
