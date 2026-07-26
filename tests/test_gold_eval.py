"""Gold regression for the evaluation arithmetic — exit criterion 4.

The unit tests in ``tests/test_metrics.py`` prove the Brier formula is right on four games
written into the test file. They cannot prove the pipeline applied it to the right games,
with the right labels, and the right probabilities attached. That is the aggregation rather
than the arithmetic, and it is where a shift-by-one in a join or an off-by-one in a mask
survives a green suite.

So a human scores one week of the test period in a spreadsheet and this compares the two.
The check is only evidence because the human's route to the number does not share any code
with the pipeline's — which is why ``hand_computed`` is emitted null and why the
instructions say not to compute it by running this project.

Red until a person has done it. A skipped gate is a forgotten gate.
"""

from __future__ import annotations

import json

import pytest

from cfb import config

FIXTURE_PATH = config.GOLD_DIR / "eval_fixture.json"
TOLERANCE = 1e-6
"""Six decimal places.

Tighter than the Phase 2 fixture's 1e-4, because that one needed a Z-table and this one is
a spreadsheet averaging a column — there is no reason for the two to differ at all beyond
float printing.
"""


def load_fixture() -> dict:
    """Read the worksheet, or an empty stand-in when it has not been generated."""
    if not FIXTURE_PATH.exists():
        return {"human_verified": False, "games": [], "hand_computed": {}, "pipeline": {}}
    return json.loads(FIXTURE_PATH.read_text())


FIXTURE = load_fixture()
GAMES = FIXTURE.get("games", [])

pytestmark = pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason=f"no fixture at {FIXTURE_PATH}; run scripts/make_eval_fixture.py",
)


def test_human_has_scored_the_slice_by_hand() -> None:
    """The gate for exit criterion 4. Red until a person has scored the week themselves."""
    hand = FIXTURE.get("hand_computed", {})
    missing = [field for field, value in hand.items() if value is None]
    if missing:
        pytest.fail(
            "hand_computed values are still blank: "
            + ", ".join(missing)
            + f"\n\nOpen {FIXTURE_PATH}, paste the p_model / p_vegas / home_win columns into a "
            "spreadsheet, compute the mean of (p - home_win)^2 for each system, and fill them "
            "in. Do not run this project's code to get them — the pipeline's answers are "
            "already in the file, and the point is that yours arrive by a different route."
        )
    if not FIXTURE.get("human_verified"):
        pytest.fail(
            f"{FIXTURE_PATH} has hand_computed values but human_verified is still false. Set "
            "it to true, with verified_by and verified_on, once you have checked them."
        )


@pytest.mark.parametrize("system", ["model_brier", "vegas_brier"])
def test_the_pipeline_matches_the_hand_computed_brier(system: str) -> None:
    """The regression this fixture exists to be.

    If a later change to the evaluation frame, the ordering, the label join or the metric
    alters either number, this is what notices.
    """
    hand = FIXTURE.get("hand_computed", {}).get(system)
    if hand is None:
        pytest.skip("not hand-computed yet; test_human_has_scored_the_slice_by_hand covers that")
    assert hand == pytest.approx(FIXTURE["pipeline"][system], abs=TOLERANCE)


def test_the_hand_counted_home_wins_match() -> None:
    """A separate check on the labels, independent of any probability.

    If the pipeline's Brier and the human's agree but the outcome column was misjoined, both
    numbers would be wrong together and nothing above would notice. Counting home wins is one
    spreadsheet cell and it isolates the label.
    """
    hand = FIXTURE.get("hand_computed", {}).get("home_wins")
    if hand is None:
        pytest.skip("not hand-counted yet")
    assert hand == FIXTURE["pipeline"]["home_wins"]


# --- The worksheet itself -----------------------------------------------------


def test_the_slice_is_a_test_season() -> None:
    """The fixture must not quietly be scoring a season the model was fitted on."""
    from cfb.model import splits

    assert FIXTURE["slice"]["season"] in splits.TEST_SEASONS


def test_the_worksheet_holds_the_whole_slice() -> None:
    assert len(GAMES) == FIXTURE["slice"]["n"]
    assert len(GAMES) > 40, "a slice this small would not catch much"
    assert len({game["game_id"] for game in GAMES}) == len(GAMES), "duplicate game in the slice"


def test_every_row_carries_what_a_human_needs_to_score_it() -> None:
    for game in GAMES:
        assert 0.0 <= game["p_model"] <= 1.0
        assert 0.0 <= game["p_vegas"] <= 1.0
        assert game["home_win"] in (0, 1)
        assert game["home_team"] and game["away_team"]


def test_the_pipeline_block_is_the_mean_squared_error_of_the_rows_shown() -> None:
    """The worksheet has to be self-consistent, or the human is scoring different games.

    This one *does* recompute from the fixture's own rows, which makes it a check that the
    file is internally coherent — not a substitute for the human, who is the only one whose
    arithmetic arrives independently.
    """
    for system, column in (("model_brier", "p_model"), ("vegas_brier", "p_vegas")):
        recomputed = sum((game[column] - game["home_win"]) ** 2 for game in GAMES) / len(GAMES)
        assert recomputed == pytest.approx(FIXTURE["pipeline"][system], abs=TOLERANCE)
