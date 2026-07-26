"""Propose the Phase 6 gold worksheet: one week of predictions for a human to score by hand.

Exit criterion 4 of the phase asks a person to recompute Brier for a small slice and match
the pipeline. This emits the slice — every game of one test week with the model's
probability, the line's probability, and the outcome — and leaves ``hand_computed`` null,
following the Phase 2 and Phase 4 fixtures. ``tests/test_gold_eval.py`` fails until it is
filled in.

What this checks that nothing else does. The unit tests prove the Brier *formula* is right
on four games written into the test file. They cannot prove that the pipeline applied it to
the right games, in the right order, against the right labels, with the right probabilities
attached — the aggregation, not the arithmetic. Those are exactly the mistakes that survive
a green test suite, and a spreadsheet catches them because a spreadsheet is not built out of
the same parts.

The probabilities handed over are the pipeline's own, so this is not a check on the model.
It is a check that the number in ``results_table.md`` is the mean squared error of the
numbers the model actually produced on the games it actually scored.

Usage::

    python scripts/make_eval_fixture.py
"""

from __future__ import annotations

import json
import sqlite3

from cfb import config
from cfb.eval import evaluate
from cfb.ingest.schema import connect
from cfb.model.train import load_model_frame, outcomes

FIXTURE_PATH = config.GOLD_DIR / "eval_fixture.json"

SLICE_SEASON: int = 2023
SLICE_WEEK: int = 1
SLICE_SEASON_TYPE: str = "regular"
"""The slice a human scores by hand: the opening week of the test period.

Confirmed with Sean at ~50 games — small enough to paste into a spreadsheet, large enough
that a shift-by-one in the join would show up. Week 1 also happens to be where every rolling
feature is null, so the model's predictions there lean on Elo; that has no bearing on the
arithmetic being checked, but it is worth knowing when reading the numbers.
"""


def team_names(conn: sqlite3.Connection) -> dict[int, str]:
    """Map team id to school, so the worksheet is readable without a second lookup."""
    return {int(row[0]): str(row[1]) for row in conn.execute("SELECT team_id, school FROM teams")}


def build(conn: sqlite3.Connection) -> dict:
    """Assemble the worksheet from a live evaluation run.

    Args:
        conn: Open connection to the built database.

    Returns:
        The fixture dict, with every ``hand_computed`` slot left null.

    Raises:
        SystemExit: If the chosen slice holds no games, which would mean the season, week or
            season type named above no longer matches the data.
    """
    frame, _ = load_model_frame()
    _, _, evaluation, predictions = evaluate.run(frame, conn)

    mask = (
        (evaluation["season"] == SLICE_SEASON)
        & (evaluation["week"] == SLICE_WEEK)
        & (evaluation["season_type"] == SLICE_SEASON_TYPE)
    ).to_numpy()
    if not mask.any():
        raise SystemExit(
            f"no games in {SLICE_SEASON} week {SLICE_WEEK} ({SLICE_SEASON_TYPE}); the slice "
            "named in this script no longer matches the data"
        )

    names = team_names(conn)
    rows = evaluation[mask]
    model = predictions[evaluate.MODEL][mask]
    vegas = predictions[evaluate.VEGAS][mask]
    actual = outcomes(evaluation)[mask]

    games = [
        {
            "game_id": int(row.game_id),
            "date": str(row.start_date)[:10],
            "home_team": names.get(int(row.home_team_id), str(row.home_team_id)),
            "away_team": names.get(int(row.away_team_id), str(row.away_team_id)),
            "p_model": float(model[index]),
            "p_vegas": float(vegas[index]),
            "home_win": int(actual[index]),
        }
        for index, row in enumerate(rows.itertuples())
    ]

    return {
        "human_verified": False,
        "verified_by": None,
        "verified_on": None,
        "slice": {
            "season": SLICE_SEASON,
            "week": SLICE_WEEK,
            "season_type": SLICE_SEASON_TYPE,
            "n": len(games),
        },
        "method": (
            "Brier = mean over the games below of (p - home_win)^2, where p is that system's "
            "probability and home_win is 1 if the home team won and 0 otherwise. No weighting, "
            "no clipping, every listed game counted once."
        ),
        "instructions": (
            "Paste the p_model and home_win columns into a spreadsheet and compute the mean of "
            "(p_model - home_win)^2; do the same for p_vegas. Fill both into hand_computed, "
            "then set human_verified, verified_by and verified_on. Do not compute these by "
            "running this project's code — the pipeline's answers are already in the pipeline "
            "block, and the point is that yours arrive by a different route. The test compares "
            "the two to 6 decimal places."
        ),
        "pipeline": {
            "model_brier": evaluate.brier_score(actual, model),
            "vegas_brier": evaluate.brier_score(actual, vegas),
            "home_wins": int(actual.sum()),
        },
        "hand_computed": {"model_brier": None, "vegas_brier": None, "home_wins": None},
        "games": games,
    }


def main() -> int:
    """Write the worksheet, preserving any hand-computed values already filled in."""
    if not config.DB_PATH.exists():
        raise SystemExit(f"no database at {config.DB_PATH}; run `make ingest` first")
    if not config.GBT_PATH.exists():
        raise SystemExit(f"no model at {config.GBT_PATH}; run `make train` first")

    conn = connect(config.DB_PATH)
    try:
        fixture = build(conn)
    finally:
        conn.close()

    # Never clobber a human's arithmetic on a re-run: it is the one thing in this repository
    # that cannot be regenerated.
    if FIXTURE_PATH.exists():
        existing = json.loads(FIXTURE_PATH.read_text())
        kept = existing.get("hand_computed", {})
        if any(value is not None for value in kept.values()):
            fixture["hand_computed"] = kept
        fixture["human_verified"] = existing.get("human_verified", False)
        fixture["verified_by"] = existing.get("verified_by")
        fixture["verified_on"] = existing.get("verified_on")

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")

    print(f"wrote {FIXTURE_PATH}")
    print(
        f"  {fixture['slice']['n']} games: {SLICE_SEASON} week {SLICE_WEEK} "
        f"({SLICE_SEASON_TYPE}), {fixture['pipeline']['home_wins']} home wins"
    )
    print("\nScore the two columns in a spreadsheet, fill in hand_computed, then set")
    print("human_verified to true. tests/test_gold_eval.py is red until you do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
