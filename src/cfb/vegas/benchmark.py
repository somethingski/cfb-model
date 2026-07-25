"""Builds ``vegas_benchmark`` — the yardstick every later result is measured against.

The project's claim is *calibration approaching the de-vigged Vegas closing line*, so this
table is the thing that claim is stated against. It is derived, not ingested: it can be
dropped and rebuilt from ``lines`` at any time.

Three decisions are baked in here and recorded in ``DECISIONS.md``:

1. **No single provider covers 2014-2025.** ``consensus`` runs 2014-2022 and stops during
   2023; Bovada runs 2019-2025. Since the seam falls exactly at the Phase 5 train/test
   boundary, providers are tried in a fixed ladder per game and the winner is recorded on
   the row, so the switch is visible in the data rather than hidden in a season rule.
2. **The primary probability is always spread-derived**, even for the 2021+ games that
   also have moneylines. A benchmark whose construction changes at 2021 would measure the
   test seasons against a different object than the training seasons. The moneyline
   probability is stored alongside, where it exists, as a free Phase 6 sensitivity check.
3. **Sigma is fitted on training seasons only** (2014-2021). :func:`estimate_sigma` raises
   if handed a later season; that guard is the leakage boundary for this phase.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from cfb import config
from cfb.ingest.schema import connect
from cfb.vegas.odds import moneyline_to_prob, spread_to_prob

PROVIDER_LADDER: tuple[str, ...] = (
    "consensus",
    "Bovada",
    "ESPN Bet",
    "DraftKings",
    "teamrankings",
    "William Hill (New Jersey)",
    "Caesars",
    "numberfire",
    "Caesars Sportsbook (Colorado)",
    "Caesars (Pennsylvania)",
    "SugarHouse",
    "Draft Kings",
)
"""Provider preference, most preferred first, applied per game.

Ordered from measured coverage, not from reputation. ``consensus`` leads because it is a
market aggregate rather than one book's opinion and it covers 2014-2022 almost completely.
Bovada follows as the only book spanning every test season (2023-2025) and back to 2019.
The tail rungs are near-empty CFBD spellings that between them are the sole source for two
2023 games; listing them explicitly costs nothing and keeps the selection deterministic.

The same ladder serves the moneyline column under a different predicate. ``consensus``
never quotes moneylines, so it is skipped there and Bovada leads by construction.
"""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS vegas_benchmark (
    game_id          INTEGER PRIMARY KEY,
    provider         TEXT NOT NULL,
    spread           REAL,
    p_home_devig     REAL NOT NULL,
    source_type      TEXT NOT NULL,
    ml_provider      TEXT,
    home_moneyline   INTEGER,
    away_moneyline   INTEGER,
    p_home_moneyline REAL,
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);
"""

SOURCE_SPREAD = "spread"
SOURCE_MONEYLINE = "moneyline"

MISMATCH_THRESHOLD = 0.25
"""How far apart the spread and moneyline views must be to count as a real disagreement.

Below this, "disagreement" is just two near-pick'em prices landing either side of 0.5,
which says nothing. Above it, the two sources name different favourites.
"""

MONEYLINE_MISMATCH_SQL = f"""
    b.p_home_moneyline IS NOT NULL
    AND b.spread != 0
    AND ABS(b.p_home_devig - b.p_home_moneyline) > {MISMATCH_THRESHOLD}
    AND (b.spread < 0) != (b.p_home_moneyline > 0.5)
"""
"""Rows where the two price sources name different favourites (RISKS #16).

Shared between the report and its test so the definition cannot drift between them.
"""


# --- selection (pure) ---------------------------------------------------------


def has_spread(row: Mapping[str, Any]) -> bool:
    """True if the line row carries a closing spread."""
    return row.get("spread") is not None


def has_moneyline_pair(row: Mapping[str, Any]) -> bool:
    """True if the line row quotes *both* sides.

    One-sided moneylines cannot be de-vigged — there is no second price to normalise
    against — so they are treated as absent rather than paired with a guess. 58 such rows
    exist in the database.
    """
    return row.get("home_moneyline") is not None and row.get("away_moneyline") is not None


def pick_line(
    rows: Iterable[Mapping[str, Any]],
    predicate: Callable[[Mapping[str, Any]], bool],
    ladder: Sequence[str] = PROVIDER_LADDER,
) -> Mapping[str, Any] | None:
    """Choose one line row for a game by provider preference.

    Args:
        rows: All ``lines`` rows for a single game.
        predicate: What makes a row usable, e.g. :func:`has_spread`.
        ladder: Provider preference, most preferred first.

    Returns:
        The usable row from the most preferred provider, or None if no row is usable.
        A provider absent from the ladder ranks last, and ties among such providers are
        broken by provider name so the choice never depends on row order.
    """
    rank = {provider: index for index, provider in enumerate(ladder)}
    usable = [row for row in rows if predicate(row)]
    if not usable:
        return None
    return min(usable, key=lambda row: (rank.get(row["provider"], len(ladder)), row["provider"]))


def benchmark_row(
    game_id: int,
    rows: Iterable[Mapping[str, Any]],
    sigma: float,
    ladder: Sequence[str] = PROVIDER_LADDER,
) -> dict[str, Any] | None:
    """Build one ``vegas_benchmark`` row from a game's line rows.

    The primary probability is spread-derived whenever a spread exists. The moneyline
    path is a fallback for a game quoted on the moneyline but not the spread — no such
    game is in the database today, but the branch exists and is unit-tested rather than
    left as an untested hole that a future CFBD change would fall through.

    Args:
        game_id: The game.
        rows: All ``lines`` rows for that game.
        sigma: Margin-residual standard deviation from :func:`estimate_sigma`.
        ladder: Provider preference.

    Returns:
        A row dict, or None when the game has no usable line data at all. None means
        *exclude and report*, never impute.
    """
    rows = list(rows)
    spread_line = pick_line(rows, has_spread, ladder)
    ml_line = pick_line(rows, has_moneyline_pair, ladder)

    p_moneyline = None
    if ml_line is not None:
        p_moneyline = moneyline_to_prob(ml_line["home_moneyline"], ml_line["away_moneyline"])

    if spread_line is not None:
        provider = spread_line["provider"]
        spread = spread_line["spread"]
        p_home = spread_to_prob(spread, sigma)
        source_type = SOURCE_SPREAD
    elif ml_line is not None:
        provider = ml_line["provider"]
        spread = None
        p_home = p_moneyline
        source_type = SOURCE_MONEYLINE
    else:
        return None

    return {
        "game_id": game_id,
        "provider": provider,
        "spread": spread,
        "p_home_devig": p_home,
        "source_type": source_type,
        "ml_provider": None if ml_line is None else ml_line["provider"],
        "home_moneyline": None if ml_line is None else ml_line["home_moneyline"],
        "away_moneyline": None if ml_line is None else ml_line["away_moneyline"],
        "p_home_moneyline": p_moneyline,
    }


# --- sigma (the leakage boundary) --------------------------------------------


def estimate_sigma(conn: sqlite3.Connection, seasons: Sequence[int]) -> float:
    """Fit the margin-residual standard deviation used by the spread conversion.

    The residual of a game is ``actual_margin - (-spread)``: how far the final margin
    landed from what the market expected. Sigma is their root-mean-square **about zero**,
    not their sample standard deviation about their own mean, because the conversion in
    :func:`~cfb.vegas.odds.spread_to_prob` assumes a zero-mean residual. Fitting an
    intercept as well was considered and rejected: the mean residual is -0.37 points on a
    sigma near 16, which moves any probability by less than 0.01 and is more likely to be
    training-set noise than signal.

    Args:
        conn: Open connection to the built database.
        seasons: Seasons to fit on. Must all be at or before
            :data:`cfb.config.TRAIN_LAST_SEASON`.

    Returns:
        Sigma in points.

    Raises:
        ValueError: If any requested season is past the training boundary, or if no
            usable games are found. The season guard is the leakage gate for this phase:
            sigma is a fitted parameter, so letting it see a validation or test season
            would leak those outcomes into the yardstick they are scored against.
    """
    illegal = sorted({season for season in seasons if season > config.TRAIN_LAST_SEASON})
    if illegal:
        raise ValueError(
            f"sigma may only be fitted on seasons through {config.TRAIN_LAST_SEASON}; "
            f"refusing to fit on {illegal}. Sigma is a fitted parameter and the benchmark "
            "is what later seasons are scored against."
        )

    by_game = line_rows_by_game(conn, seasons=seasons, completed_only=True)
    margins = dict(
        conn.execute(
            f"""
            SELECT game_id, home_points - away_points
            FROM games
            WHERE completed = 1
              AND home_points IS NOT NULL
              AND away_points IS NOT NULL
              AND season IN ({",".join("?" * len(seasons))})
            """,
            tuple(seasons),
        ).fetchall()
    )

    residuals = []
    for game_id, margin in margins.items():
        line = pick_line(by_game.get(game_id, []), has_spread)
        if line is None:
            continue
        residuals.append(margin + line["spread"])

    if not residuals:
        raise ValueError(f"no completed games with a spread in seasons {list(seasons)}")
    return math.sqrt(sum(r * r for r in residuals) / len(residuals))


# --- build --------------------------------------------------------------------


def line_rows_by_game(
    conn: sqlite3.Connection,
    seasons: Sequence[int] | None = None,
    completed_only: bool = False,
) -> dict[int, list[dict[str, Any]]]:
    """Load ``lines`` rows grouped by game.

    Args:
        conn: Open connection.
        seasons: Restrict to these seasons; None means all.
        completed_only: Restrict to games with a final score.

    Returns:
        Mapping of ``game_id`` to its line rows.
    """
    clauses = []
    params: list[Any] = []
    if seasons is not None:
        clauses.append(f"g.season IN ({','.join('?' * len(seasons))})")
        params.extend(seasons)
    if completed_only:
        clauses.append("g.completed = 1")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in conn.execute(
        f"""
        SELECT l.game_id, l.provider, l.spread, l.home_moneyline, l.away_moneyline
        FROM lines l JOIN games g ON g.game_id = l.game_id
        {where}
        """,
        params,
    ):
        grouped.setdefault(row["game_id"], []).append(dict(row))
    return grouped


def build_benchmark(conn: sqlite3.Connection, sigma: float) -> dict[str, int]:
    """Create and populate ``vegas_benchmark`` for every game with usable line data.

    Rebuilt from scratch on every run so a re-run can never leave a stale row behind.

    Args:
        conn: Open connection to the built database.
        sigma: From :func:`estimate_sigma`.

    Returns:
        Counts: ``games``, ``included``, ``no_line_rows``, ``unusable_line_rows``.
    """
    conn.executescript(SCHEMA_SQL)
    conn.execute("DELETE FROM vegas_benchmark")

    by_game = line_rows_by_game(conn)
    game_ids = [row[0] for row in conn.execute("SELECT game_id FROM games ORDER BY game_id")]

    rows = []
    no_line_rows = 0
    unusable = 0
    for game_id in game_ids:
        lines = by_game.get(game_id, [])
        if not lines:
            no_line_rows += 1
            continue
        row = benchmark_row(game_id, lines, sigma)
        if row is None:
            unusable += 1
            continue
        rows.append(row)

    conn.executemany(
        """
        INSERT INTO vegas_benchmark (
            game_id, provider, spread, p_home_devig, source_type,
            ml_provider, home_moneyline, away_moneyline, p_home_moneyline
        ) VALUES (
            :game_id, :provider, :spread, :p_home_devig, :source_type,
            :ml_provider, :home_moneyline, :away_moneyline, :p_home_moneyline
        )
        """,
        rows,
    )
    conn.commit()
    return {
        "games": len(game_ids),
        "included": len(rows),
        "no_line_rows": no_line_rows,
        "unusable_line_rows": unusable,
    }


# --- reporting ----------------------------------------------------------------


def coverage_report(conn: sqlite3.Connection) -> str:
    """Render the per-season coverage and sanity report for exit criteria 3 and 4.

    Args:
        conn: Open connection with ``vegas_benchmark`` already built.

    Returns:
        A printable report. Excluded games are counted per season and listed by game_id,
        never imputed.
    """
    lines = ["", "Coverage by season", "-" * 78]
    lines.append(
        f"{'season':>6}  {'games':>6}  {'in bench':>8}  {'excluded':>8}  "
        f"{'with ML':>8}  {'mean p_home':>11}  providers"
    )
    for season, games, included, with_ml, mean_p in conn.execute(
        """
        SELECT g.season,
               COUNT(*),
               COUNT(b.game_id),
               COUNT(b.p_home_moneyline),
               AVG(b.p_home_devig)
        FROM games g LEFT JOIN vegas_benchmark b ON b.game_id = g.game_id
        GROUP BY g.season ORDER BY g.season
        """
    ):
        mix = ", ".join(
            f"{provider} {count}"
            for provider, count in conn.execute(
                """
                SELECT b.provider, COUNT(*) FROM vegas_benchmark b
                JOIN games g ON g.game_id = b.game_id
                WHERE g.season = ? GROUP BY 1 ORDER BY 2 DESC
                """,
                (season,),
            )
        )
        mean_text = "-" if mean_p is None else f"{mean_p:.4f}"
        lines.append(
            f"{season:>6}  {games:>6}  {included:>8}  {games - included:>8}  "
            f"{with_ml:>8}  {mean_text:>11}  {mix}"
        )

    excluded = conn.execute(
        """
        SELECT g.game_id, g.season,
               (SELECT COUNT(*) FROM lines l WHERE l.game_id = g.game_id)
        FROM games g
        WHERE g.game_id NOT IN (SELECT game_id FROM vegas_benchmark)
        ORDER BY g.season, g.game_id
        """
    ).fetchall()
    no_rows = [row for row in excluded if row[2] == 0]
    unusable = [row for row in excluded if row[2] > 0]
    lines += [
        "",
        f"Excluded: {len(excluded)} games, never imputed (RISKS #10)",
        f"  no line row from any provider : {len(no_rows)}",
        f"  line rows but no usable price : {len(unusable)}"
        + (f" -> {[row[0] for row in unusable]}" if unusable else ""),
    ]

    lines += ["", "Sanity distribution (exit criterion 3)", "-" * 78]
    total, mean_p, min_p, max_p = conn.execute(
        "SELECT COUNT(*), AVG(p_home_devig), MIN(p_home_devig), MAX(p_home_devig) "
        "FROM vegas_benchmark"
    ).fetchone()
    fbs_mean, fbs_n = conn.execute(
        """
        SELECT AVG(b.p_home_devig), COUNT(*) FROM vegas_benchmark b
        JOIN games g ON g.game_id = b.game_id
        JOIN team_seasons th ON th.team_id = g.home_team_id AND th.season = g.season
        JOIN team_seasons ta ON ta.team_id = g.away_team_id AND ta.season = g.season
        WHERE th.classification = 'fbs' AND ta.classification = 'fbs'
        """
    ).fetchone()
    actual_all, actual_fbs = conn.execute(
        """
        SELECT AVG(CASE WHEN g.home_points > g.away_points THEN 1.0 ELSE 0 END),
               AVG(CASE WHEN th.classification = 'fbs' AND ta.classification = 'fbs'
                        THEN (CASE WHEN g.home_points > g.away_points THEN 1.0 ELSE 0 END)
                   END)
        FROM games g
        JOIN vegas_benchmark b ON b.game_id = g.game_id
        LEFT JOIN team_seasons th ON th.team_id = g.home_team_id AND th.season = g.season
        LEFT JOIN team_seasons ta ON ta.team_id = g.away_team_id AND ta.season = g.season
        WHERE g.completed = 1 AND g.home_points IS NOT NULL
        """
    ).fetchone()
    outside = conn.execute(
        "SELECT COUNT(*) FROM vegas_benchmark WHERE p_home_devig <= 0.01 OR p_home_devig >= 0.99"
    ).fetchone()[0]

    lines += [
        f"  rows                        : {total}",
        f"  mean p_home, all games      : {mean_p:.4f}   (actual home win rate {actual_all:.4f})",
        f"  mean p_home, both FBS       : {fbs_mean:.4f}   (actual {actual_fbs:.4f}, n={fbs_n})",
        f"  min / max p_home            : {min_p:.6f} / {max_p:.6f}",
        f"  outside (0.01, 0.99)        : {outside} ({100 * outside / total:.1f}%)",
    ]

    for source_type, count in conn.execute(
        "SELECT source_type, COUNT(*) FROM vegas_benchmark GROUP BY 1 ORDER BY 2 DESC"
    ):
        lines.append(f"  source_type={source_type:<12}    : {count}")

    lines += ["", "Moneyline cross-check (RISKS #16)", "-" * 78]
    with_ml, mean_gap = conn.execute(
        "SELECT COUNT(p_home_moneyline), AVG(ABS(p_home_devig - p_home_moneyline)) "
        "FROM vegas_benchmark WHERE p_home_moneyline IS NOT NULL"
    ).fetchone()
    mismatched, neutral, postseason = conn.execute(
        f"""
        SELECT COUNT(*), SUM(g.neutral_site), SUM(g.season_type = 'postseason')
        FROM vegas_benchmark b JOIN games g ON g.game_id = b.game_id
        WHERE {MONEYLINE_MISMATCH_SQL}
        """
    ).fetchone()
    lines += [
        f"  games with both prices      : {with_ml}",
        f"  mean |spread - moneyline|   : {mean_gap:.4f}",
        f"  disagree on the favourite   : {mismatched} by more than "
        f"{MISMATCH_THRESHOLD} ({neutral} neutral-site, {postseason} postseason)",
        "  -> these are a CFBD home/away assignment defect on neutral-site games, not a",
        "     de-vig error. p_home_devig is spread-derived and unaffected. Phase 6 must",
        "     exclude them from the moneyline sensitivity check rather than average over them.",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the benchmark table and print the coverage report.

    Args:
        argv: Command-line arguments; None reads ``sys.argv``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description="Build the de-vigged Vegas benchmark table.")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="print the coverage report for an already-built table without rebuilding",
    )
    args = parser.parse_args(argv)

    if not config.DB_PATH.exists():
        raise SystemExit(f"no database at {config.DB_PATH}; run `make ingest` first")

    conn = connect(config.DB_PATH)
    try:
        if not args.report_only:
            train_seasons = [s for s in config.SEASONS if s <= config.TRAIN_LAST_SEASON]
            sigma = estimate_sigma(conn, train_seasons)
            print(
                f"sigma = {sigma:.4f} points, fitted on {train_seasons[0]}-{train_seasons[-1]} "
                "(training seasons only)"
            )
            counts = build_benchmark(conn, sigma)
            print(
                f"built vegas_benchmark: {counts['included']} of {counts['games']} games "
                f"({counts['no_line_rows']} with no line row, "
                f"{counts['unusable_line_rows']} with no usable price)"
            )
        print(coverage_report(conn))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
