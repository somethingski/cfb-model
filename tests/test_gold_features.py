"""Gold regression for the rolling-window arithmetic.

The leakage audit proves the features can be rebuilt from a database that ends at kickoff.
It does **not** prove the arithmetic inside them is what anyone intended — a consistently
wrong mean recomputes to the same consistently wrong mean and sails through. That is what
this fixture is for, and it is only evidence if a human did the arithmetic independently:
``hand_computed`` is filled in with a calculator, ``pipeline`` is what this project
produced, and the test asserts they agree.

Following ``gold/vegas_fixture.json``, the generator emits every answer as null and this
test fails until a person has filled them in. A fixture whose answers came out of the code
it checks would pass its own test while proving nothing.
"""

from __future__ import annotations

import json

import pytest

from cfb import config

FIXTURE_PATH = config.GOLD_DIR / "features_fixture.json"
TOLERANCE = 1e-6
"""These are means and ratios of small integers; a calculator gets them exactly."""


def load_fixture() -> dict:
    """Read the worksheet, or an empty stand-in when it has not been generated."""
    if not FIXTURE_PATH.exists():
        return {"human_verified": False, "games": []}
    return json.loads(FIXTURE_PATH.read_text())


FIXTURE = load_fixture()
GAMES = FIXTURE.get("games", [])

pytestmark = pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason=f"no fixture at {FIXTURE_PATH}; run scripts/make_features_fixture.py",
)


def test_human_has_done_the_arithmetic() -> None:
    """Red until a person has worked these windows by hand."""
    missing = [
        f"{game['case']} ({game['game_id']}): {field}"
        for game in GAMES
        for field, value in game.get("hand_computed", {}).items()
        if value is None and game["pipeline"].get(field) is not None
    ]
    if missing:
        pytest.fail(
            "hand_computed values are still blank:\n  "
            + "\n  ".join(missing)
            + f"\n\nWork them out from the worksheets in {FIXTURE_PATH} with a calculator — "
            "not by running this project's code — then fill them in."
        )
    if not FIXTURE.get("human_verified"):
        pytest.fail(
            f"{FIXTURE_PATH} has human_verified=false. Set human_verified/verified_by/"
            "verified_on once you have checked every row."
        )


@pytest.mark.parametrize("game", GAMES, ids=[game["case"] for game in GAMES])
def test_pipeline_matches_the_hand_computed_window(game: dict) -> None:
    """The regression itself, one game per case."""
    if not FIXTURE.get("human_verified"):
        pytest.skip("fixture not yet hand-computed; test_human_has_done_the_arithmetic is the gate")

    for field, expected in game["hand_computed"].items():
        if expected is None:
            continue
        produced = game["pipeline"][field]
        assert produced is not None, f"{field}: hand-computed {expected}, pipeline produced null"
        assert produced == pytest.approx(expected, abs=TOLERANCE), (
            f"{game['case']} ({game['game_id']}) {field}: "
            f"hand-computed {expected}, pipeline {produced}"
        )


def test_the_worksheet_covers_the_cases_that_can_go_wrong() -> None:
    """A fixture of three identical situations would not catch much.

    The three cases are chosen so that between them they exercise a clean window, an FCS
    game inside a window (which the confirmed policy keeps in the rolling means), and
    windows of different lengths with different rest on the two sides.
    """
    assert len(GAMES) >= 3, "the plan asks for 3 hand-computed games"

    windows = [len(game["home_worksheet"]["prior_games_this_season"]) for game in GAMES]
    assert max(windows) <= 4, "a window this long is not something a human will check by hand"
    assert min(windows) >= 2, "a one-game window does not exercise the averaging"

    fcs = any(
        prior["opponent_is_fcs"]
        for game in GAMES
        for side in ("home_worksheet", "away_worksheet")
        for prior in game[side]["prior_games_this_season"]
    )
    assert fcs, "no FCS game inside any window; the confirmed inclusion policy is unpinned"

    uneven = any(
        game["pipeline"]["rest_days_home"] != game["pipeline"]["rest_days_away"] for game in GAMES
    )
    assert uneven, "both teams rest equally in every case; rest_diff is unpinned"


def test_the_pipeline_block_was_not_copied_into_the_answers() -> None:
    """Guards the one way this fixture could be faked without anybody noticing.

    Pasting ``pipeline`` into ``hand_computed`` would turn the regression green while
    proving nothing. A human with a calculator writes rounded decimals; the pipeline writes
    full float repr. Identical float64 values across every field of every game means the
    block was copied.
    """
    if not FIXTURE.get("human_verified"):
        pytest.skip("nothing to check until the fixture is filled in")

    identical = [
        field
        for game in GAMES
        for field, value in game["hand_computed"].items()
        if value is not None and repr(value) == repr(game["pipeline"][field])
    ]
    inexact = [
        field
        for game in GAMES
        for field, value in game["pipeline"].items()
        if value is not None and value != round(value, 6)
    ]
    assert not (inexact and len(identical) == sum(len(g["pipeline"]) for g in GAMES)), (
        "every hand_computed value is bit-identical to the pipeline's, including the "
        "non-terminating ones. That is a copy, not a computation."
    )
