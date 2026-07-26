"""Emits the Phase 4 gold worksheet: three games with their rolling windows spelled out.

Following the Phase 2 and Phase 3 fixtures, this generator writes the **inputs** and leaves
every expected value null. A fixture whose answers came out of this project's own code
would pass its own test while proving nothing; the human's independent arithmetic is the
only thing that makes it evidence.

The worksheet deliberately gives raw box-score fields rather than derived ones — rushing
attempts and the ``"completions-attempts"`` string, not a plays count — so that the human
also re-derives the plays decomposition, which was a Phase 4 decision rather than a fact.

Run: ``python scripts/make_features_fixture.py``
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pandas as pd

from cfb import config
from cfb.features import build
from cfb.ingest.schema import connect

FIXTURE_PATH = config.GOLD_DIR / "features_fixture.json"

HAND_COMPUTED_FIELDS: tuple[str, ...] = (
    "rest_days_home",
    "rest_days_away",
    "off_ppg_roll_home",
    "def_ppg_roll_home",
    "off_ypp_roll_home",
    "def_ypp_roll_home",
    "pace_roll_home",
    "off_ppg_roll_away",
    "def_ppg_roll_away",
    "off_ypp_roll_away",
    "def_ypp_roll_away",
    "pace_roll_away",
    "prev_season_win_pct_home",
    "prev_season_win_pct_away",
)


def team_names(conn: sqlite3.Connection) -> dict[int, str]:
    """Team id to school name."""
    return dict(conn.execute("SELECT team_id, school FROM teams"))


def raw_box_score(conn: sqlite3.Connection, game_id: int, team_id: int) -> dict[str, Any]:
    """The three box-score fields a plays count is built from, as stored.

    Args:
        conn: Open connection.
        game_id: The game.
        team_id: The team.

    Returns:
        ``total_yards``, ``rushing_attempts`` and the raw ``completion_attempts`` string.
    """
    rows = dict(
        conn.execute(
            "SELECT stat_name, stat_raw FROM game_team_stats WHERE game_id = ? AND team_id = ? "
            "AND stat_name IN ('totalYards', 'rushingAttempts', 'completionAttempts')",
            (game_id, team_id),
        )
    )
    return {
        "total_yards": rows.get("totalYards"),
        "rushing_attempts": rows.get("rushingAttempts"),
        "completion_attempts": rows.get("completionAttempts"),
    }


def worksheet(
    conn: sqlite3.Connection,
    context: build.FeatureContext,
    game: build.ScheduledGame,
    team_id: int,
    names: dict[int, str],
) -> dict[str, Any]:
    """One team's side of the worksheet: every prior game it played, with raw numbers.

    Args:
        conn: Open connection.
        context: Loaded feature context.
        game: The target game.
        team_id: The team whose window to spell out.
        names: Team id to school name.

    Returns:
        The prior games this season, the previous season's record, and the previous game
        before this one whatever season it fell in.
    """
    priors = build.priors_before(context.team_games.get(team_id, []), game.start_date)
    in_season = [prior for prior in priors if prior.season == game.season]
    last_season = [prior for prior in priors if prior.season == game.season - 1]

    return {
        "team": names.get(team_id, str(team_id)),
        "team_id": team_id,
        "previous_game_kickoff": priors[-1].start_date if priors else None,
        "prior_games_this_season": [
            {
                "kickoff": prior.start_date,
                "opponent": names.get(prior.opponent_id, str(prior.opponent_id)),
                "opponent_is_fcs": prior.opponent_is_fcs,
                "points_for": prior.points_for,
                "points_against": prior.points_against,
                "own_box_score": raw_box_score(conn, prior.game_id, team_id),
                "opponent_box_score": raw_box_score(conn, prior.game_id, prior.opponent_id),
            }
            for prior in in_season
        ],
        "previous_season": {
            "season": game.season - 1,
            "wins": sum(1 for prior in last_season if prior.won == 1.0),
            "losses": sum(1 for prior in last_season if prior.won == 0.0),
            "ties": sum(1 for prior in last_season if prior.won == 0.5),
        },
    }


def choose_games(frame) -> list[tuple[int, str]]:
    """Pick three games with windows small enough to work by hand.

    Args:
        frame: The built feature store.

    Returns:
        ``(game_id, case)`` for each chosen game.

    Raises:
        RuntimeError: If the store has no game matching a case, which would mean the store
            is not what this generator was written against.
    """
    regular = frame[(frame["season_type"] == "regular") & frame["off_ypp_roll_home"].notna()]
    cases = [
        (
            "three prior games each, no FCS in either window",
            (regular["prior_games_home"] == 3)
            & (regular["prior_games_away"] == 3)
            & (regular["fcs_games_in_window_home"] == 0)
            & (regular["fcs_games_in_window_away"] == 0)
            & (regular["season"] == 2018),
        ),
        (
            "an FCS opponent inside the home team's window",
            (regular["prior_games_home"] == 3)
            & (regular["prior_games_away"] == 3)
            & (regular["fcs_games_in_window_home"] >= 1)
            & (regular["season"] == 2022),
        ),
        (
            "uneven windows and uneven rest",
            (regular["prior_games_home"] == 2)
            & (regular["prior_games_away"] == 3)
            & (regular["rest_days_home"] != regular["rest_days_away"])
            & (regular["season"] == 2015),
        ),
    ]
    chosen: list[tuple[int, str]] = []
    for case, mask in cases:
        matches = regular[mask].sort_values(["start_date", "game_id"])
        if matches.empty:
            raise RuntimeError(f"no game in the store matches the case {case!r}")
        chosen.append((int(matches.iloc[0]["game_id"]), case))
    return chosen


def main() -> int:
    """Write the worksheet.

    Returns:
        Process exit status.
    """
    if not config.DB_PATH.exists():
        raise SystemExit(f"no database at {config.DB_PATH}; run `make ingest` first")

    frame = build.read_frame()
    conn = connect(config.DB_PATH)
    try:
        names = team_names(conn)
        context = build.load_context(conn)
        schedule = {game.game_id: game for game in context.schedule}
        indexed = frame.set_index("game_id")

        games = []
        for game_id, case in choose_games(frame):
            game = schedule[game_id]
            stored = indexed.loc[game_id]
            games.append(
                {
                    "case": case,
                    "game_id": game_id,
                    "season": game.season,
                    "week": game.week,
                    "kickoff": game.start_date,
                    "home": names.get(game.home_team_id, str(game.home_team_id)),
                    "away": names.get(game.away_team_id, str(game.away_team_id)),
                    "home_worksheet": worksheet(conn, context, game, game.home_team_id, names),
                    "away_worksheet": worksheet(conn, context, game, game.away_team_id, names),
                    "pipeline": {
                        field: None if pd.isna(stored[field]) else float(stored[field])
                        for field in HAND_COMPUTED_FIELDS
                    },
                    "hand_computed": dict.fromkeys(HAND_COMPUTED_FIELDS),
                }
            )
    finally:
        conn.close()

    fixture = {
        "human_verified": False,
        "verified_by": None,
        "verified_on": None,
        "method": (
            "Rolling stats average the team's PREVIOUS games this season only — the game "
            "on the row is never in its own window. ppg and pace are plain means over "
            "those games. ypp is a ratio of sums: total yards divided by total plays "
            "across the window, not the mean of the per-game ratios. plays = "
            "rushingAttempts + pass attempts, where pass attempts is the second number in "
            "the 'completions-attempts' string (NCAA charges sack yardage to rushing, so "
            "sacks are already inside rushingAttempts). def_* use the opponent's numbers "
            "from those same games. rest_days is UTC calendar days since the team's "
            "previous completed game, capped at 30."
        ),
        "instructions": (
            "Work every hand_computed value out yourself from the worksheets below — with "
            "a calculator, not by running this project's code. Then fill them in and set "
            "human_verified/verified_by/verified_on. The test compares hand_computed "
            "against pipeline to 1e-6, and fails while any value is still null."
        ),
        "games": games,
    }
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {FIXTURE_PATH} with {len(games)} games and every answer blank")
    for game in games:
        print(
            f"  {game['game_id']}  {game['season']} wk {game['week']}  "
            f"{game['away']} at {game['home']}  ({game['case']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
