"""The split scheme partitions the seasons, and the guards fire on poisoned input."""

from __future__ import annotations

import pandas as pd
import pytest

from cfb import config
from cfb.model import splits


def frame_with(seasons: list[int]) -> pd.DataFrame:
    """A minimal frame carrying only what the guards read."""
    return pd.DataFrame({"season": seasons, "label_home_win": [1] * len(seasons)})


def test_splits_partition_the_season_range_with_no_overlap_and_no_gaps():
    covered = (
        list(splits.TRAIN_SEASONS) + list(splits.VALIDATION_SEASONS) + list(splits.TEST_SEASONS)
    )
    assert sorted(covered) == list(config.SEASONS)
    assert len(set(covered)) == len(covered), "a season belongs to two splits"


def test_the_splits_are_season_forward():
    assert max(splits.TRAIN_SEASONS) < min(splits.VALIDATION_SEASONS)
    assert max(splits.VALIDATION_SEASONS) < min(splits.TEST_SEASONS)


def test_the_training_boundary_is_the_one_in_config():
    assert max(splits.TRAIN_SEASONS) == config.TRAIN_LAST_SEASON


@pytest.mark.parametrize(
    ("season", "expected"),
    [(2014, "train"), (2021, "train"), (2022, "validation"), (2023, "test"), (2025, "test")],
)
def test_split_of_names_the_right_split(season, expected):
    assert splits.split_of(season) == expected


def test_split_of_refuses_a_season_outside_the_project_range():
    with pytest.raises(ValueError, match="belongs to no split"):
        splits.split_of(2013)


def test_forward_folds_are_the_plans_three_folds():
    assert splits.forward_folds() == (
        ((2014, 2015, 2016, 2017, 2018), 2019),
        ((2014, 2015, 2016, 2017, 2018, 2019), 2020),
        ((2014, 2015, 2016, 2017, 2018, 2019, 2020), 2021),
    )


def test_every_fold_fits_only_on_seasons_before_the_one_it_scores():
    for fit_seasons, score_season in splits.forward_folds():
        assert max(fit_seasons) < score_season


def test_no_fold_scores_outside_the_training_seasons():
    for _, score_season in splits.forward_folds():
        assert splits.split_of(score_season) == "train"


def test_forward_folds_rejects_unordered_seasons():
    with pytest.raises(ValueError, match="strictly ascending"):
        splits.forward_folds([2016, 2014, 2015, 2017, 2018, 2019])


def test_forward_folds_rejects_too_few_seasons_to_make_a_fold():
    with pytest.raises(ValueError, match="cannot build a forward fold"):
        splits.forward_folds([2014, 2015, 2016])


def test_assert_no_test_rows_passes_on_clean_input():
    splits.assert_no_test_rows(frame_with([2014, 2019, 2021]))


def test_assert_no_test_rows_fires_on_a_poisoned_test_season_row():
    poisoned = frame_with([2014, 2019, 2024])
    with pytest.raises(splits.LeakageError, match=r"test-season rows reached.*2024"):
        splits.assert_no_test_rows(poisoned, "the GBT training set")


def test_assert_validation_only_passes_on_the_validation_season():
    splits.assert_validation_only(frame_with([2022, 2022]))


@pytest.mark.parametrize("intruder", [2021, 2024])
def test_assert_validation_only_fires_on_any_other_season(intruder):
    with pytest.raises(splits.LeakageError, match="non-validation rows"):
        splits.assert_validation_only(frame_with([2022, intruder]))


def test_assert_validation_only_refuses_an_empty_frame():
    with pytest.raises(splits.LeakageError, match="no rows at all"):
        splits.assert_validation_only(frame_with([]))


def test_the_guards_refuse_a_frame_with_no_season_column():
    with pytest.raises(KeyError, match="no 'season' column"):
        splits.assert_no_test_rows(pd.DataFrame({"elo_diff": [1.0]}))


def test_rows_for_selects_only_the_requested_seasons():
    frame = frame_with([2014, 2015, 2016])
    assert splits.seasons_in(splits.rows_for(frame, [2014, 2016])) == (2014, 2016)
