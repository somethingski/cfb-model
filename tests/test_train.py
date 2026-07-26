"""Training-side invariants: no test season reaches a fit, and two runs agree.

Every test here runs on a synthetic frame rather than the built store, so they are fast
and so a failure means the code is wrong rather than the data having changed. The one
integration test at the bottom checks that the real store still has the shape this module
assumes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfb import config
from cfb.features.build import FEATURE_COLUMNS, KEY_SPECS
from cfb.model import splits, train

GAMES_PER_SEASON = 160


def synthetic_store(seasons=tuple(config.SEASONS), fcs_every: int = 20) -> pd.DataFrame:
    """Build a frame shaped like the Phase 4 feature store.

    The label carries real signal from ``elo_diff`` so a fitted model has something to
    learn; everything else is noise of a plausible scale. Seeded, so the frame is the same
    on every run and a determinism test is testing LightGBM rather than the fixture.

    Args:
        seasons: Seasons to generate.
        fcs_every: Every nth game is flagged as an FCS matchup.

    Returns:
        A frame with the store's columns, keys and label.
    """
    rng = np.random.default_rng(11)
    rows = []
    game_id = 1
    for season in seasons:
        for index in range(GAMES_PER_SEASON):
            elo_diff = float(rng.normal(0, 150))
            probability = 1.0 / (1.0 + np.exp(-(elo_diff + 50) / 100.0))
            week = 1 + index % 15
            rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "start_date": f"{season}-09-{1 + index % 28:02d}T18:00:00+00:00",
                    "home_team_id": 100 + index % 60,
                    "away_team_id": 200 + index % 60,
                    "home_elo_pre": 1500 + elo_diff / 2,
                    "away_elo_pre": 1500 - elo_diff / 2,
                    "elo_diff": elo_diff,
                    "week": week,
                    "season_type": "postseason" if week > 14 else "regular",
                    "neutral_site": int(index % 17 == 0),
                    "conference_game": int(index % 3 == 0),
                    "fcs_opponent": int(index % fcs_every == 0),
                    "rest_days_home": 7 + index % 4,
                    "rest_days_away": 7 + index % 5,
                    "rest_diff": (index % 4) - (index % 5),
                    "off_ppg_roll_home": None if week == 1 else float(rng.normal(28, 7)),
                    "def_ppg_roll_home": None if week == 1 else float(rng.normal(25, 7)),
                    "off_ypp_roll_home": None if week == 1 else float(rng.normal(5.8, 0.8)),
                    "def_ypp_roll_home": None if week == 1 else float(rng.normal(5.6, 0.8)),
                    "pace_roll_home": None if week == 1 else float(rng.normal(70, 6)),
                    "off_ppg_roll_away": None if week == 1 else float(rng.normal(27, 7)),
                    "def_ppg_roll_away": None if week == 1 else float(rng.normal(26, 7)),
                    "off_ypp_roll_away": None if week == 1 else float(rng.normal(5.7, 0.8)),
                    "def_ypp_roll_away": None if week == 1 else float(rng.normal(5.7, 0.8)),
                    "pace_roll_away": None if week == 1 else float(rng.normal(69, 6)),
                    "prev_season_win_pct_home": float(rng.uniform(0, 1)),
                    "prev_season_win_pct_away": float(rng.uniform(0, 1)),
                    "prior_games_home": week - 1,
                    "prior_games_away": week - 1,
                    "fcs_games_in_window_home": int(week > 3),
                    "fcs_games_in_window_away": 0,
                    "as_of": f"{season}-09-{1 + index % 28:02d}T00:00:00+00:00",
                    "label_home_win": int(rng.uniform() < probability),
                }
            )
            game_id += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """The synthetic store, reduced to a model frame."""
    return train.model_frame(synthetic_store())


def small_params() -> dict:
    """Fast parameters for tests that only need *a* fitted model, not a good one."""
    return {**train.FIXED_PARAMS, "num_leaves": 7, "learning_rate": 0.1, "min_data_in_leaf": 40}


# --- The model frame ---------------------------------------------------------


def test_model_frame_drops_every_fcs_game(frame):
    assert len(frame) > 0
    assert (frame["fcs_opponent"] == 0).all()


def test_model_frame_encoding_does_not_depend_on_which_rows_were_loaded(frame):
    """A frame with no postseason games must still encode 'postseason' the same way.

    Category codes are assigned by order of appearance unless the categories are declared,
    so a differently-filtered frame would silently give the saved model a different meaning
    for the same string.
    """
    regular_only = train.model_frame(
        synthetic_store()[lambda f: f["season_type"] == "regular"].copy()
    )
    assert list(regular_only["season_type"].cat.categories) == list(
        frame["season_type"].cat.categories
    )


def test_model_frame_refuses_an_unknown_season_type():
    store = synthetic_store()
    store.loc[1, "season_type"] = "exhibition"  # row 1: row 0 is an FCS game and is dropped first
    with pytest.raises(ValueError, match="unknown season_type"):
        train.model_frame(store)


def test_model_features_are_the_phase_4_features_minus_the_declared_exclusions():
    assert set(train.MODEL_FEATURES) == set(FEATURE_COLUMNS) - set(train.EXCLUDED_FROM_MODEL)


def test_no_key_column_and_no_label_is_a_model_feature():
    keys = {spec.name for spec in KEY_SPECS}
    assert not keys & set(train.MODEL_FEATURES)
    assert "label_home_win" not in train.MODEL_FEATURES


@pytest.mark.parametrize(
    "term", ["spread", "moneyline", "over_under", "vegas", "line", "devig", "p_home"]
)
def test_no_market_term_appears_in_the_model_matrix(term):
    """A line-derived feature is the stop-everything error of this project.

    Phase 4's audit scans the feature builder's source; this scans the column list the
    model is actually fitted on, which is the last place it could enter.
    """
    assert not [name for name in train.MODEL_FEATURES if term in name]


# --- The leakage guards ------------------------------------------------------


def test_fit_refuses_a_frame_containing_a_test_season_row(frame):
    poisoned = pd.concat(
        [splits.rows_for(frame, splits.TRAIN_SEASONS), splits.rows_for(frame, [2024]).head(1)]
    )
    with pytest.raises(splits.LeakageError, match=r"test-season rows reached.*2024"):
        train.fit_booster(poisoned, small_params(), num_boost_round=5)


def test_fit_refuses_a_test_season_row_in_the_early_stopping_set(frame):
    train_rows = splits.rows_for(frame, splits.TRAIN_SEASONS)
    with pytest.raises(splits.LeakageError, match="early-stopping set"):
        train.fit_booster(
            train_rows,
            small_params(),
            num_boost_round=5,
            valid=splits.rows_for(frame, [2023]),
            early_stopping_rounds=2,
        )


def test_the_forward_cv_search_never_fits_on_a_season_it_scores(frame, monkeypatch):
    """The guard proves test seasons stay out; this proves the folds are forward.

    Recorded at the fitting boundary rather than argued from the fold definition, because
    the fold definition is exactly the thing that could be wrong.
    """
    seen = []
    original = train.fit_booster

    def recording_fit(train_rows, params, num_boost_round, valid=None, early_stopping_rounds=None):
        seen.append((splits.seasons_in(train_rows), splits.seasons_in(valid)))
        return original(train_rows, params, num_boost_round, valid, early_stopping_rounds)

    monkeypatch.setattr(train, "fit_booster", recording_fit)
    train.run_folds(splits.rows_for(frame, splits.TRAIN_SEASONS), train.Candidate(7, 0.1, 40, 0.9))

    assert len(seen) == len(splits.forward_folds())
    for fit_seasons, scored_seasons in seen:
        assert max(fit_seasons) < min(scored_seasons)
        assert not set(fit_seasons) & set(scored_seasons)


def test_the_calibrator_refuses_a_non_validation_row(frame):
    validation = splits.rows_for(frame, splits.VALIDATION_SEASONS)
    poisoned = pd.concat([validation, splits.rows_for(frame, [2024]).head(1)])
    raw = np.linspace(0.1, 0.9, len(poisoned))
    with pytest.raises(splits.LeakageError, match="non-validation rows"):
        train.fit_calibrator(poisoned, raw)


def test_the_calibrator_refuses_training_rows_too(frame):
    rows = splits.rows_for(frame, [2021])
    with pytest.raises(splits.LeakageError, match="non-validation rows"):
        train.fit_calibrator(rows, np.linspace(0.1, 0.9, len(rows)))


def test_the_calibrator_refuses_misaligned_predictions(frame):
    validation = splits.rows_for(frame, splits.VALIDATION_SEASONS)
    with pytest.raises(ValueError, match="predictions were given"):
        train.fit_calibrator(validation, np.array([0.5, 0.5]))


# --- Determinism -------------------------------------------------------------


def test_two_fits_with_the_same_seed_produce_identical_predictions(frame):
    train_rows = splits.rows_for(frame, splits.TRAIN_SEASONS)
    validation = splits.rows_for(frame, splits.VALIDATION_SEASONS)
    predictions = [
        train.predict(train.fit_booster(train_rows, small_params(), num_boost_round=40), validation)
        for _ in range(2)
    ]
    np.testing.assert_array_equal(predictions[0], predictions[1])


def test_two_searches_with_the_same_seed_produce_identical_fold_log_losses(frame):
    grid = {
        "num_leaves": (7,),
        "learning_rate": (0.1,),
        "min_data_in_leaf": (40,),
        "feature_fraction": (0.9,),
    }
    train_rows = splits.rows_for(frame, splits.TRAIN_SEASONS)
    first, second = (train.search(train_rows, grid) for _ in range(2))
    assert [f.log_loss for f in first[0].folds] == [f.log_loss for f in second[0].folds]
    assert first[0].mean_log_loss == second[0].mean_log_loss


# --- Calibration -------------------------------------------------------------


def test_the_calibrator_is_monotone_on_a_grid(frame):
    validation = splits.rows_for(frame, splits.VALIDATION_SEASONS)
    rng = np.random.default_rng(3)
    raw = np.clip(rng.uniform(0.05, 0.95, len(validation)), 0, 1)
    calibrator = train.fit_calibrator(validation, raw)

    grid = np.linspace(0.0, 1.0, 101)
    calibrated = train.apply_calibrator(calibrator, grid)
    assert np.all(np.diff(calibrated) >= 0), "isotonic output must be non-decreasing"


def test_calibrated_output_is_clipped_into_the_supportable_range(frame):
    validation = splits.rows_for(frame, splits.VALIDATION_SEASONS)
    raw = np.linspace(0.01, 0.99, len(validation))
    calibrator = train.fit_calibrator(validation, raw)
    calibrated = train.apply_calibrator(calibrator, np.array([0.0, 0.5, 1.0]))
    assert calibrated.min() >= train.CLIP_LO
    assert calibrated.max() <= train.CLIP_HI


# --- Metrics -----------------------------------------------------------------


def test_a_perfect_forecast_scores_zero_brier():
    outcomes = np.array([1.0, 0.0, 1.0])
    assert train.brier_score(outcomes, outcomes) == 0.0


def test_a_confident_miss_is_finite_but_large():
    loss = train.log_loss_score(np.array([1.0]), np.array([0.0]))
    assert np.isfinite(loss) and loss > 30


def test_evaluate_reports_the_benchmark_only_on_games_that_have_a_line(frame):
    validation = splits.rows_for(frame, splits.VALIDATION_SEASONS).head(10)
    raw = np.full(len(validation), 0.6)
    benchmark = {int(game_id): 0.55 for game_id in validation["game_id"].head(6)}
    metrics = train.evaluate(validation, raw, raw, benchmark, home_rate=0.57)
    assert metrics.n == 10
    assert metrics.n_with_line == 6
    assert metrics.n_without_line == 4


def test_beating_the_benchmark_is_reported_as_an_alarm(frame):
    """The alarm is the point of the comparison, so it gets a test of its own."""
    validation = splits.rows_for(frame, splits.VALIDATION_SEASONS).head(20)
    actual = validation["label_home_win"].to_numpy(dtype=float)
    benchmark = {int(game_id): 0.5 for game_id in validation["game_id"]}
    metrics = train.evaluate(validation, actual, actual, benchmark, home_rate=0.57)
    assert metrics.beats_vegas is True


# --- The built store ---------------------------------------------------------


@pytest.mark.integration
def test_the_real_store_still_has_the_shape_this_module_assumes():
    if not config.FEATURE_STORE_PATH.exists():
        pytest.skip("no feature store; run `make features` first")
    model_rows, excluded = train.load_model_frame()
    assert excluded > 0, "the store should contain FCS games for this phase to exclude"
    assert (model_rows["fcs_opponent"] == 0).all()
    assert splits.seasons_in(model_rows) == tuple(config.SEASONS)
    assert set(train.MODEL_FEATURES) <= set(model_rows.columns)
    assert len(splits.rows_for(model_rows, splits.VALIDATION_SEASONS)) > 500
