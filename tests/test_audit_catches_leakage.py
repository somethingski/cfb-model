"""Proof that the leakage audit can fail.

``CLAUDE.md``: a test that cannot fail is not a test. The same applies, harder, to a gate.
An audit that has only ever been seen to pass is indistinguishable from an audit that
always passes, and the whole of Phase 4 rests on it.

So: build a small league, confirm the honest feature store survives the audit, then poison
the store two ways and demand the audit fails *and names the poisoned column*. Naming
matters — an audit that fails for an unrelated reason would satisfy a weaker assertion
while telling you nothing.

The two poisons are the two shapes leakage actually takes here:

1. **The shift is dropped.** A rolling stat averages the games up to *and including* the
   one being predicted. This is the canonical bug — the entire reason ``priors_before``
   exists — and it is invisible in every summary statistic.
2. **A post-game rating is stored.** The Elo column holds what the team was rated *after*
   the game rather than before it.

The poisons live here and never in ``src/``, following ``tests/test_elo_chronology.py``.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from cfb.features import audit, build
from cfb.ingest.schema import connect
from tests import toy_league

SAMPLE = 12
"""Random games on top of the pinned edge cases. The toy league is small, so this plus the
pinned set covers most of it."""


@pytest.fixture
def toy(tmp_path) -> sqlite3.Connection:
    """A built toy database: games, box scores and pre-game Elo."""
    conn = connect(tmp_path / "toy.sqlite")
    toy_league.populate(conn)
    yield conn
    conn.close()


def audit_toy(conn: sqlite3.Connection, frame: pd.DataFrame) -> audit.AuditResult:
    """Run the real audit against a toy store.

    Args:
        conn: The toy database.
        frame: The store to audit, honest or poisoned.

    Returns:
        The audit result.
    """
    return audit.run_audit(conn, frame, sample_size=SAMPLE, seed=audit.AUDIT_SEED)


def failures_mentioning(result: audit.AuditResult, column: str) -> list[str]:
    """Failures that name a given column.

    Args:
        result: A completed audit.
        column: The column the poison touched.

    Returns:
        The matching failure messages.
    """
    return [failure for failure in result.failures if column in failure]


# --- the honest baseline ------------------------------------------------------


def test_an_honest_store_passes_the_audit(toy: sqlite3.Connection) -> None:
    """Without this, every test below could be passing for the wrong reason."""
    result = audit_toy(toy, build.build_frame(toy))
    assert result.passed, "the honest store failed:\n" + "\n".join(result.failures)
    assert len(result.sampled) > 10, "too few games audited for the poisons below to mean much"


# --- poison 1: the shift is dropped -------------------------------------------


def rolling_ppg_including_this_game(conn: sqlite3.Connection) -> dict[tuple[int, int], float]:
    """Mean points scored per team, counting the game being predicted.

    The canonical leakage bug, computed deliberately.

    Args:
        conn: The toy database.

    Returns:
        ``(game_id, team_id)`` to the leaky rolling mean.
    """
    totals: dict[tuple[int, int], list[int]] = {}
    leaked: dict[tuple[int, int], float] = {}
    rows = conn.execute(
        "SELECT game_id, season, home_team_id, away_team_id, home_points, away_points "
        "FROM games ORDER BY start_date, game_id"
    ).fetchall()
    for row in rows:
        for team_id, points in (
            (row["home_team_id"], row["home_points"]),
            (row["away_team_id"], row["away_points"]),
        ):
            history = totals.setdefault((team_id, row["season"]), [])
            history.append(points)
            leaked[(row["game_id"], team_id)] = sum(history) / len(history)
    return leaked


def test_the_audit_catches_a_rolling_stat_with_the_shift_dropped(toy: sqlite3.Connection) -> None:
    """The bug the whole phase is built to prevent must not survive the gate."""
    frame = build.build_frame(toy)
    leaky = rolling_ppg_including_this_game(toy)
    frame["off_ppg_roll_home"] = [
        leaky[(row.game_id, row.home_team_id)] for row in frame.itertuples()
    ]
    frame["off_ppg_roll_away"] = [
        leaky[(row.game_id, row.away_team_id)] for row in frame.itertuples()
    ]

    result = audit_toy(toy, frame)
    assert not result.passed, "a rolling stat that includes its own game passed the audit"
    named = failures_mentioning(result, "off_ppg_roll_home")
    assert named, (
        "the audit failed, but not about the poisoned column; it would not have told you "
        f"where to look. Failures: {result.failures}"
    )


# --- poison 2: a post-game rating is stored -----------------------------------


def post_game_elo(conn: sqlite3.Connection) -> dict[tuple[int, int], float]:
    """What each team was rated *after* each game.

    Read off the stored table rather than recomputed: a team's post-game rating for game
    *g* is the rating it carried into its next game.

    Args:
        conn: The toy database.

    Returns:
        ``(game_id, team_id)`` to the post-game rating, for every game the team followed
        with another one.
    """
    rows = conn.execute(
        """
        SELECT g.game_id, g.start_date, g.home_team_id, g.away_team_id,
               e.home_elo_pre, e.away_elo_pre
        FROM games g JOIN elo_pregame e ON e.game_id = g.game_id
        ORDER BY g.start_date, g.game_id
        """
    ).fetchall()
    previous: dict[int, int] = {}
    post: dict[tuple[int, int], float] = {}
    for row in rows:
        for team_id, rating in (
            (row["home_team_id"], row["home_elo_pre"]),
            (row["away_team_id"], row["away_elo_pre"]),
        ):
            if team_id in previous:
                post[(previous[team_id], team_id)] = rating
            previous[team_id] = row["game_id"]
    return post


def test_the_audit_catches_a_post_game_rating(toy: sqlite3.Connection) -> None:
    """An Elo column that knows the result of its own game must not survive the gate."""
    frame = build.build_frame(toy)
    post = post_game_elo(toy)
    frame["home_elo_pre"] = [
        post.get((row.game_id, row.home_team_id), row.home_elo_pre) for row in frame.itertuples()
    ]
    frame["elo_diff"] = frame["home_elo_pre"] - frame["away_elo_pre"]

    result = audit_toy(toy, frame)
    assert not result.passed, "a post-game Elo rating passed the audit"
    assert failures_mentioning(result, "home_elo_pre"), (
        f"the audit failed, but not about the poisoned column. Failures: {result.failures}"
    )


# --- poison 3: a null quietly filled in ---------------------------------------


def test_the_audit_catches_a_back_filled_null(toy: sqlite3.Connection) -> None:
    """Back-filling week 1 from later games is leakage that looks like tidying up.

    Nulls are compared as strictly as numbers precisely so this cannot slip through.
    """
    frame = build.build_frame(toy)
    filled = frame["off_ppg_roll_home"].isna()
    assert filled.any(), "the toy league has no null rolling stats to back-fill"
    frame.loc[filled, "off_ppg_roll_home"] = frame["off_ppg_roll_home"].mean()

    result = audit_toy(toy, frame)
    assert not result.passed, "a back-filled null passed the audit"
    assert failures_mentioning(result, "off_ppg_roll_home"), (
        f"the audit failed, but not about the poisoned column. Failures: {result.failures}"
    )


# --- poison 4: a market column reaches the builder ----------------------------


def test_the_market_check_fires_on_a_builder_that_reads_the_lines_table() -> None:
    """The stop-everything error, proved detectable.

    ``assert_no_market_source`` scans executable text and exempts docstrings, so this feeds
    it both: a module whose prose mentions the market must pass, and one whose code queries
    it must not.
    """
    innocent = (
        '"""Reads the lines table? Never — spread and moneyline are the benchmark."""\nX = 1\n'
    )
    guilty = '"""A blameless docstring."""\nQUERY = "SELECT spread FROM lines"\n'

    assert not [term for term in audit.MARKET_TERMS if term in audit.executable_text(innocent)], (
        "a docstring mentioning the market tripped the check; it will cry wolf"
    )
    found = [term for term in audit.MARKET_TERMS if term in audit.executable_text(guilty)]
    assert "lines" in found and "spread" in found, (
        "a builder querying the lines table went undetected; the check cannot fail"
    )
