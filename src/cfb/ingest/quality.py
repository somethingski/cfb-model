"""Data-quality assertions and the per-season summary table.

These checks fail loudly and never repair anything. A missing line, a missing box score,
or an anomalous season count is a fact about the source data: it gets reported here,
excluded downstream, and written into ``RISKS.md`` — never interpolated away.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

# Bounds on FBS-involved games per season. Observed: 868 (2014) to 910 (2023). The band
# is wide enough to survive schedule expansion and narrow enough that a broken filter or
# a half-finished backfill trips it.
MIN_GAMES_PER_SEASON = 700
MAX_GAMES_PER_SEASON = 1000

# 2020 was COVID-shortened with conference-only schedules (RISKS.md #6). It gets its own
# documented lower bound rather than a relaxed global one, so the anomaly stays visible.
COVID_SEASON = 2020
MIN_GAMES_COVID_SEASON = 450


class DataQualityError(AssertionError):
    """Raised when ingested data violates an invariant the later phases depend on."""


@dataclass(frozen=True)
class SeasonSummary:
    """Per-season coverage counts used for the human-eyeballed summary table."""

    season: int
    games: int
    completed: int
    with_lines: int
    with_team_stats: int

    @property
    def pct_lines(self) -> float:
        """Percentage of games having at least one posted line from any provider."""
        return 100.0 * self.with_lines / self.games if self.games else 0.0

    @property
    def pct_team_stats(self) -> float:
        """Percentage of games having at least one team box-score row."""
        return 100.0 * self.with_team_stats / self.games if self.games else 0.0


def summarize(conn: sqlite3.Connection) -> list[SeasonSummary]:
    """Compute per-season coverage counts.

    Args:
        conn: Connection to the ingested database.

    Returns:
        One summary per season present, ordered by season.
    """
    rows = conn.execute(
        """
        SELECT g.season                                        AS season,
               COUNT(*)                                        AS games,
               SUM(g.completed)                                AS completed,
               SUM(EXISTS(SELECT 1 FROM lines l
                          WHERE l.game_id = g.game_id))        AS with_lines,
               SUM(EXISTS(SELECT 1 FROM game_team_stats s
                          WHERE s.game_id = g.game_id))        AS with_team_stats
        FROM games g
        GROUP BY g.season
        ORDER BY g.season
        """
    ).fetchall()
    return [
        SeasonSummary(
            season=row["season"],
            games=row["games"],
            completed=row["completed"] or 0,
            with_lines=row["with_lines"] or 0,
            with_team_stats=row["with_team_stats"] or 0,
        )
        for row in rows
    ]


def format_summary(summaries: list[SeasonSummary]) -> str:
    """Render the per-season summary as a fixed-width table.

    Args:
        summaries: Output of :func:`summarize`.

    Returns:
        A printable table. Line and box-score coverage are reported, never asserted:
        gaps are excluded by later phases, not filled.
    """
    header = f"{'season':>6}  {'games':>6}  {'completed':>9}  {'% lines':>8}  {'% stats':>8}"
    divider = "-" * len(header)
    lines = [header, divider]
    for summary in summaries:
        lines.append(
            f"{summary.season:>6}  {summary.games:>6}  {summary.completed:>9}  "
            f"{summary.pct_lines:>7.1f}%  {summary.pct_team_stats:>7.1f}%"
        )
    total_games = sum(s.games for s in summaries)
    lines.append(divider)
    lines.append(f"{'total':>6}  {total_games:>6}")
    return "\n".join(lines)


def check_no_null_start_dates(conn: sqlite3.Connection) -> None:
    """Assert the leakage clock has no holes.

    Args:
        conn: Connection to the ingested database.

    Raises:
        DataQualityError: If any game has a null or empty ``start_date``.
    """
    bad = conn.execute(
        "SELECT game_id FROM games WHERE start_date IS NULL OR TRIM(start_date) = ''"
    ).fetchall()
    if bad:
        ids = ", ".join(str(row["game_id"]) for row in bad[:5])
        raise DataQualityError(
            f"{len(bad)} game(s) have no start_date, e.g. {ids}. start_date is the "
            "leakage clock; it cannot have holes."
        )


def check_completed_games_have_scores(conn: sqlite3.Connection) -> None:
    """Assert every completed game has both final scores.

    Args:
        conn: Connection to the ingested database.

    Raises:
        DataQualityError: If a completed game is missing a score.
    """
    bad = conn.execute(
        """
        SELECT game_id FROM games
        WHERE completed = 1 AND (home_points IS NULL OR away_points IS NULL)
        """
    ).fetchall()
    if bad:
        ids = ", ".join(str(row["game_id"]) for row in bad[:5])
        raise DataQualityError(
            f"{len(bad)} completed game(s) have a null score, e.g. {ids}. "
            "Exclude them explicitly and record the gap in RISKS.md; do not impute."
        )


def check_no_duplicate_games(conn: sqlite3.Connection) -> None:
    """Assert ``game_id`` is unique.

    The primary key already enforces this; the check exists so that a future schema
    change cannot quietly remove the guarantee.

    Args:
        conn: Connection to the ingested database.

    Raises:
        DataQualityError: If any ``game_id`` appears more than once.
    """
    duplicates = conn.execute(
        "SELECT game_id, COUNT(*) AS n FROM games GROUP BY game_id HAVING n > 1"
    ).fetchall()
    if duplicates:
        raise DataQualityError(f"{len(duplicates)} duplicate game_id(s) in games")


def check_season_game_counts(conn: sqlite3.Connection, summaries: list[SeasonSummary]) -> None:
    """Assert each season's game count is within documented bounds.

    Args:
        conn: Connection to the ingested database (unused; kept for a uniform signature).
        summaries: Output of :func:`summarize`.

    Raises:
        DataQualityError: If a season falls outside its bound.
    """
    del conn
    for summary in summaries:
        lower = MIN_GAMES_COVID_SEASON if summary.season == COVID_SEASON else MIN_GAMES_PER_SEASON
        if not lower <= summary.games <= MAX_GAMES_PER_SEASON:
            raise DataQualityError(
                f"season {summary.season} has {summary.games} FBS-involved games, "
                f"outside the documented bound [{lower}, {MAX_GAMES_PER_SEASON}]"
            )


def run_all_checks(conn: sqlite3.Connection) -> list[SeasonSummary]:
    """Run every data-quality assertion and return the per-season summary.

    Args:
        conn: Connection to the ingested database.

    Returns:
        The per-season summary, for printing by the caller.

    Raises:
        DataQualityError: On the first violated invariant.
    """
    summaries = summarize(conn)
    if not summaries:
        raise DataQualityError("no games in the database; nothing to check")
    check_no_duplicate_games(conn)
    check_no_null_start_dates(conn)
    check_completed_games_have_scores(conn)
    check_season_game_counts(conn, summaries)
    return summaries
