"""Backfill CFBD seasons into SQLite.

Idempotent by construction: every response is cached to disk on first fetch, and every
write is an upsert keyed on the natural primary key. Running twice produces identical row
counts and, on the second run, zero network calls — which the CLI reports so the claim is
checkable rather than assumed.

Safe to interrupt: caching is per-request and commits are per-season, so a resumed run
picks up where it stopped without re-hitting the API for what it already has.

Run with ``make ingest`` or ``python -m cfb.ingest.backfill --help``.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cfb import config
from cfb.ingest import quality, transform
from cfb.ingest.client import CFBDClient, CFBDError
from cfb.ingest.schema import connect, init_db

log = logging.getLogger("cfb.ingest")


def upsert_teams(conn: sqlite3.Connection, teams: Iterable[dict[str, Any]]) -> None:
    """Insert or update team identities.

    Args:
        conn: Open connection.
        teams: Dicts with ``team_id``, ``school`` and optionally ``mascot``,
            ``abbreviation``.
    """
    conn.executemany(
        """
        INSERT INTO teams (team_id, school, mascot, abbreviation)
        VALUES (:team_id, :school, :mascot, :abbreviation)
        ON CONFLICT(team_id) DO UPDATE SET
            school       = excluded.school,
            mascot       = COALESCE(excluded.mascot, teams.mascot),
            abbreviation = COALESCE(excluded.abbreviation, teams.abbreviation)
        """,
        [
            {
                "team_id": team["team_id"],
                "school": team["school"],
                "mascot": team.get("mascot"),
                "abbreviation": team.get("abbreviation"),
            }
            for team in teams
        ],
    )


def upsert_team_seasons(
    conn: sqlite3.Connection, rows: Iterable[dict[str, Any]], *, overwrite: bool
) -> None:
    """Insert per-season team affiliations.

    Args:
        conn: Open connection.
        rows: Dicts with ``team_id``, ``season`` and optionally ``conference``,
            ``division``, ``classification``.
        overwrite: True for authoritative ``/teams/fbs`` rows, which carry conference
            division. False for rows recovered from the game payload (FCS opponents),
            which must not overwrite the richer record.
    """
    conflict = (
        """
        ON CONFLICT(team_id, season) DO UPDATE SET
            conference     = excluded.conference,
            division       = excluded.division,
            classification = excluded.classification
        """
        if overwrite
        else "ON CONFLICT(team_id, season) DO NOTHING"
    )
    conn.executemany(
        f"""
        INSERT INTO team_seasons (team_id, season, conference, division, classification)
        VALUES (:team_id, :season, :conference, :division, :classification)
        {conflict}
        """,
        [
            {
                "team_id": row["team_id"],
                "season": row["season"],
                "conference": row.get("conference"),
                "division": row.get("division"),
                "classification": row.get("classification"),
            }
            for row in rows
        ],
    )


def upsert_games(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> None:
    """Insert or replace game spine rows.

    Args:
        conn: Open connection.
        rows: Output of :func:`cfb.ingest.transform.game_row`.
    """
    conn.executemany(
        """
        INSERT OR REPLACE INTO games (
            game_id, season, week, season_type, start_date, start_time_tbd,
            neutral_site, conference_game, home_team_id, away_team_id,
            home_points, away_points, completed
        ) VALUES (
            :game_id, :season, :week, :season_type, :start_date, :start_time_tbd,
            :neutral_site, :conference_game, :home_team_id, :away_team_id,
            :home_points, :away_points, :completed
        )
        """,
        list(rows),
    )


def upsert_lines(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> None:
    """Insert or replace betting lines, one row per game per provider.

    Args:
        conn: Open connection.
        rows: Output of :func:`cfb.ingest.transform.line_rows`.
    """
    conn.executemany(
        """
        INSERT OR REPLACE INTO lines (
            game_id, provider, spread, spread_open, over_under, over_under_open,
            home_moneyline, away_moneyline
        ) VALUES (
            :game_id, :provider, :spread, :spread_open, :over_under, :over_under_open,
            :home_moneyline, :away_moneyline
        )
        """,
        list(rows),
    )


def upsert_team_stats(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> None:
    """Insert or replace tidy box-score stat rows.

    Args:
        conn: Open connection.
        rows: Output of :func:`cfb.ingest.transform.stat_rows`.
    """
    conn.executemany(
        """
        INSERT OR REPLACE INTO game_team_stats (
            game_id, team_id, is_home, stat_name, stat_value, stat_raw
        ) VALUES (:game_id, :team_id, :is_home, :stat_name, :stat_value, :stat_raw)
        """,
        list(rows),
    )


def ingest_season(conn: sqlite3.Connection, client: CFBDClient, season: int) -> dict[str, int]:
    """Fetch and store one season.

    Order matters: teams are written before games so the foreign keys resolve, and games
    before lines and box scores so orphan rows are dropped rather than silently creating
    a game the spine does not know about.

    Args:
        conn: Open connection.
        client: Cached API client.
        season: Season year.

    Returns:
        Counts of rows written, keyed by table.
    """
    fbs_teams = client.get("/teams/fbs", year=season)
    upsert_teams(
        conn,
        [
            {
                "team_id": team["id"],
                "school": team["school"],
                "mascot": team.get("mascot"),
                "abbreviation": team.get("abbreviation"),
            }
            for team in fbs_teams
        ],
    )
    upsert_team_seasons(
        conn,
        [
            {
                "team_id": team["id"],
                "season": season,
                "conference": team.get("conference"),
                "division": team.get("division"),
                "classification": team.get("classification"),
            }
            for team in fbs_teams
        ],
        overwrite=True,
    )

    # The spine is fetched unfiltered and narrowed here; see transform.is_fbs_involved
    # for why the API's own classification filter is not used.
    all_games = client.get("/games", year=season)
    games = [game for game in all_games if transform.is_fbs_involved(game)]

    team_rows = [row for game in games for row in transform.teams_from_game(game)]
    upsert_teams(conn, team_rows)
    upsert_team_seasons(conn, team_rows, overwrite=False)
    upsert_games(conn, [transform.game_row(game) for game in games])

    game_ids = {game["id"] for game in games}

    line_records = client.get("/lines", year=season)
    line_rows = [
        row
        for record in line_records
        if record.get("id") in game_ids
        for row in transform.line_rows(record)
    ]
    upsert_lines(conn, line_rows)

    # /games/teams requires a week, so the calendar drives the iteration; it also carries
    # the postseason week, which a naive range(1, 16) would miss.
    stat_row_count = 0
    for entry in client.get("/calendar", year=season):
        records = client.get(
            "/games/teams",
            year=season,
            week=entry["week"],
            seasonType=entry["seasonType"],
        )
        stat_rows = [
            row
            for record in records
            if record.get("id") in game_ids
            for row in transform.stat_rows(record)
        ]
        upsert_team_stats(conn, stat_rows)
        stat_row_count += len(stat_rows)

    conn.commit()
    return {"games": len(games), "lines": len(line_rows), "team_stats": stat_row_count}


def backfill(
    seasons: Iterable[int], db_path: Path | None = None, client: CFBDClient | None = None
) -> None:
    """Backfill the given seasons into SQLite and run the quality gate.

    Args:
        seasons: Season years, ingested in ascending order.
        db_path: Target database. Defaults to ``config.DB_PATH``.
        client: Injected for testing; a default cached client otherwise.

    Raises:
        quality.DataQualityError: If the ingested data violates an invariant.
    """
    config.ensure_dirs()
    db_path = db_path if db_path is not None else config.DB_PATH
    client = client if client is not None else CFBDClient()

    conn = connect(db_path)
    try:
        init_db(conn)
        for season in sorted(seasons):
            counts = ingest_season(conn, client, season)
            log.info(
                "%d: %d games, %d line rows, %d stat rows",
                season,
                counts["games"],
                counts["lines"],
                counts["team_stats"],
            )

        log.info(
            "\nrequests: %d network, %d from cache",
            client.network_calls,
            client.cache_hits,
        )
        summaries = quality.run_all_checks(conn)
        log.info("\n%s", quality.format_summary(summaries))
        log.info(
            "\nLine and box-score coverage is reported, not enforced: gaps are excluded "
            "by later phases and recorded in RISKS.md, never filled in."
        )
    finally:
        conn.close()


def parse_seasons(spec: str) -> list[int]:
    """Parse a season specification.

    Args:
        spec: ``"2014-2025"``, ``"2019,2021"``, or ``"2020"``.

    Returns:
        Sorted, de-duplicated season years.

    Raises:
        ValueError: If the spec is malformed.
    """
    seasons: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, _, end = part.partition("-")
            seasons.update(range(int(start), int(end) + 1))
        else:
            seasons.add(int(part))
    if not seasons:
        raise ValueError(f"no seasons parsed from {spec!r}")
    return sorted(seasons)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list; ``sys.argv[1:]`` when None.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Backfill CFBD data into SQLite.")
    parser.add_argument(
        "--seasons",
        default=f"{config.FIRST_SEASON}-{config.LAST_SEASON}",
        help="seasons to ingest, e.g. '2014-2025' or '2019,2021' (default: all)",
    )
    parser.add_argument("--db", type=Path, default=config.DB_PATH, help="target SQLite file")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the API key with one live request and exit without ingesting",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    try:
        if args.check:
            client = CFBDClient()
            teams = client.get("/teams/fbs", year=config.LAST_SEASON)
            log.info("API key works: %d FBS teams returned for %d", len(teams), config.LAST_SEASON)
            return 0
        backfill(parse_seasons(args.seasons), db_path=args.db)
    except (CFBDError, quality.DataQualityError, ValueError) as exc:
        log.error("ingest failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
