"""Propose the Phase 2 gold worksheet: line inputs for a human to de-vig by hand.

This script deliberately does **not** fill in the answers. The Phase 1 fixture could be
proposed complete because a human verifies it against an external source
(collegefootballdata.com), so the pipeline is not grading its own homework. Here the
check is arithmetic, and there is no external source to look it up in — so the only way
the fixture can be evidence is if the human computes the probabilities themselves and the
pipeline's numbers are compared against *those*.

So the emitted worksheet carries the inputs and leaves ``hand_computed`` null.
``tests/test_gold_vegas.py`` fails while any of them is null.

Usage::

    python scripts/make_gold_vegas_fixture.py
"""

from __future__ import annotations

import json
import sqlite3

from cfb import config
from cfb.ingest.schema import connect
from cfb.vegas.benchmark import estimate_sigma

FIXTURE_PATH = config.GOLD_DIR / "vegas_fixture.json"

CASES: dict[int, str] = {
    401520148: "home favourite, spread and moneyline",
    401634300: "away favourite, spread and moneyline",
    401525819: "pick em",
    400869421: "pre-moneyline era, spread only",
    400934513: "extreme favourite, widest spread in the database",
}
"""Chosen to pin both signs, both source paths, and the tail.

Cases 1 and 2 are near mirror images (-7.0 with -300/+250 against +8.0 with +250/-300),
so an inverted sign convention cannot satisfy both. That pairing is the point of them.
"""


def build(conn: sqlite3.Connection) -> dict:
    """Assemble the worksheet from the built benchmark table.

    Args:
        conn: Open connection with ``vegas_benchmark`` populated.

    Returns:
        The fixture dict, with every ``hand_computed`` slot left null.

    Raises:
        SystemExit: If a chosen game is missing from the benchmark.
    """
    train_seasons = [s for s in config.SEASONS if s <= config.TRAIN_LAST_SEASON]
    sigma = estimate_sigma(conn, train_seasons)

    games = []
    for game_id, case in CASES.items():
        row = conn.execute(
            """
            SELECT g.season, h.school AS home_team, a.school AS away_team,
                   b.provider, b.spread, b.ml_provider, b.home_moneyline, b.away_moneyline,
                   b.p_home_devig, b.p_home_moneyline, b.source_type
            FROM vegas_benchmark b
            JOIN games g ON g.game_id = b.game_id
            JOIN teams h ON h.team_id = g.home_team_id
            JOIN teams a ON a.team_id = g.away_team_id
            WHERE b.game_id = ?
            """,
            (game_id,),
        ).fetchone()
        if row is None:
            raise SystemExit(f"game {game_id} is not in vegas_benchmark; run `make benchmark`")

        games.append(
            {
                "case": case,
                "game_id": game_id,
                "season": row["season"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "provider": row["provider"],
                "spread": row["spread"],
                "ml_provider": row["ml_provider"],
                "home_moneyline": row["home_moneyline"],
                "away_moneyline": row["away_moneyline"],
                "source_type": row["source_type"],
                "pipeline": {
                    "p_home_devig": row["p_home_devig"],
                    "p_home_moneyline": row["p_home_moneyline"],
                },
                "hand_computed": {
                    "p_home_devig": None,
                    "p_home_moneyline": None,
                },
            }
        )

    return {
        "human_verified": False,
        "verified_by": None,
        "verified_on": None,
        "sigma": sigma,
        "sigma_fitted_on": f"{train_seasons[0]}-{train_seasons[-1]}",
        "method": (
            "p_home_devig = Phi(-spread / sigma). "
            "p_home_moneyline = devig(american_to_implied(home), american_to_implied(away)), "
            "multiplicative. Spreads are home-relative: negative favours the home team."
        ),
        "instructions": (
            "Compute each hand_computed value yourself from the inputs on the row - with a "
            "calculator and a Z-table, not by running this project's code. Then fill them in "
            "and set human_verified/verified_by/verified_on. The test compares hand_computed "
            "against pipeline to 4 decimal places."
        ),
        "games": games,
    }


def main() -> int:
    """Write the worksheet, preserving any hand-computed values already filled in."""
    if not config.DB_PATH.exists():
        raise SystemExit(f"no database at {config.DB_PATH}; run `make ingest` first")

    conn = connect(config.DB_PATH)
    try:
        fixture = build(conn)
    finally:
        conn.close()

    # Never clobber a human's arithmetic on a re-run: that is the one thing here that
    # cannot be regenerated.
    if FIXTURE_PATH.exists():
        existing = json.loads(FIXTURE_PATH.read_text())
        previous = {game["game_id"]: game.get("hand_computed", {}) for game in existing["games"]}
        for game in fixture["games"]:
            kept = previous.get(game["game_id"], {})
            if any(value is not None for value in kept.values()):
                game["hand_computed"] = kept
        fixture["human_verified"] = existing.get("human_verified", False)
        fixture["verified_by"] = existing.get("verified_by")
        fixture["verified_on"] = existing.get("verified_on")

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {FIXTURE_PATH} with {len(fixture['games'])} games")
    print("sigma =", fixture["sigma"])
    print("\nFill in every hand_computed value by hand, then set human_verified to true.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
