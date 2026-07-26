"""Walks the schedule in chronological order and writes ``elo_pregame``.

**This module is the phase's leakage boundary.** One rule decides whether Phase 3 is
sound, and it is enforced in exactly one place, :func:`run_elo`:

    The ratings written for a game are read **before** that game's result is applied,
    and games are visited in ``(start_date, game_id)`` order.

Everything else here is bookkeeping around that rule. Two consequences worth stating up
front, because both are easy to get wrong quietly:

* **Only pre-game ratings are stored.** Post-game ratings are a join away from becoming a
  feature that knows the result of its own game, so they never enter the database — the
  same reasoning that kept CFBD's ``homePostgameElo`` out of Phase 1's schema.
* **Ordering is by kickoff, never by week.** Two 2020 New Mexico State games are stamped
  ``season = 2020`` with kickoffs in Feb/Mar 2021 (``RISKS.md`` #13), so ``(season, week)``
  ordering would place them before games they actually followed.

Subdivision policy, confirmed by Sean before this was written and logged in
``DECISIONS.md``: an FCS opponent is a fixed 1200 that never updates, while the FBS team
it plays does update — losing to an FCS team has to hurt. A team's subdivision is read per
season from ``team_seasons``, so a team that moves up (nine of them, 2015-2025) starts at
the newcomer prior in its first FBS season, and a team that moves down (Idaho, 2018)
becomes a fixed 1200 opponent and has its rating discarded.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cfb import config
from cfb.elo.engine import (
    EloParams,
    brier_score,
    expected,
    log_loss,
    run_season_regression,
    update,
)
from cfb.ingest.schema import connect

FBS = "fbs"
"""The only classification whose ratings this system tracks. The database holds exactly
two values, ``fbs`` and ``fcs``."""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS elo_pregame (
    game_id      INTEGER PRIMARY KEY,
    home_elo_pre REAL NOT NULL,
    away_elo_pre REAL NOT NULL,
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);
"""

UTC_SUFFIX = "+00:00"
"""Every kickoff is stored with this offset, which is what makes lexicographic ordering of
``start_date`` identical to chronological ordering. Asserted in :func:`load_games` rather
than assumed: a single row in another offset would reorder the walk silently."""


@dataclass(frozen=True)
class Game:
    """One scheduled game, as the Elo walk needs it."""

    game_id: int
    season: int
    start_date: str
    neutral_site: bool
    home_team_id: int
    away_team_id: int
    home_points: int | None
    away_points: int | None
    completed: bool

    @property
    def has_result(self) -> bool:
        """True if the game was played and both scores are present.

        One game in the database was cancelled and never played (``RISKS.md`` #12). It
        still gets pre-game ratings — the matchup was scheduled, and Phase 4 may want to
        know what the teams were rated — but it moves nobody's rating.
        """
        return self.completed and self.home_points is not None and self.away_points is not None


@dataclass(frozen=True)
class PregameRow:
    """A game with the ratings both teams carried into it.

    Carries more than the three columns that reach the database: the extras (subdivision
    flags, neutral-site flag, result) are what the tuning objective and the sanity report
    score against, and recomputing them from SQL for every one of the several hundred
    grid-search runs would be the slow way to get the same numbers.
    """

    game_id: int
    season: int
    home_team_id: int
    away_team_id: int
    home_elo_pre: float
    away_elo_pre: float
    home_is_fbs: bool
    away_is_fbs: bool
    neutral_site: bool
    home_points: int | None
    away_points: int | None

    @property
    def both_fbs(self) -> bool:
        """True if both teams were FBS that season."""
        return self.home_is_fbs and self.away_is_fbs

    @property
    def home_won(self) -> float | None:
        """1.0 if the home team won, 0.0 if it lost, 0.5 for a tie, None if unplayed."""
        if self.home_points is None or self.away_points is None:
            return None
        if self.home_points > self.away_points:
            return 1.0
        return 0.0 if self.home_points < self.away_points else 0.5


@dataclass
class EloRun:
    """The output of one pass over the schedule.

    Attributes:
        rows: Pre-game ratings for every game, in the order they were visited.
        ratings: Final ratings of every tracked team.
        season_end_ratings: Ratings at the end of each season, before that season's
            regression was applied. What the sanity report ranks.
        promotions: ``(season, team_id, rating)`` for each team initialised into the
            system, including the first-season cohort.
        demotions: ``(season, team_id)`` for each tracked team dropped on leaving FBS.
    """

    rows: list[PregameRow]
    ratings: dict[int, float]
    season_end_ratings: dict[int, dict[int, float]]
    promotions: list[tuple[int, int, float]]
    demotions: list[tuple[int, int]]


class RatingBook:
    """Current ratings, plus the subdivision rules that decide whose ratings move.

    Kept as a class rather than a bare dict because "what is this team rated right now"
    and "does this team's rating update" are the same question asked two ways, and both
    answers depend on the season. Splitting them across call sites is how an FCS team
    quietly acquires a rating.
    """

    def __init__(self, classification: Mapping[tuple[int, int], str], params: EloParams) -> None:
        """Initialise an empty book.

        Args:
            classification: ``(team_id, season)`` to ``"fbs"`` or ``"fcs"``, from
                ``team_seasons``. Every game's two teams have an entry for that game's
                season, so a lookup during the walk never falls through to a guess.
            params: Rating parameters.
        """
        self._classification = classification
        self._params = params
        self._ratings: dict[int, float] = {}
        self.promotions: list[tuple[int, int, float]] = []
        self.demotions: list[tuple[int, int]] = []

    @property
    def ratings(self) -> dict[int, float]:
        """The tracked ratings. Mutating the copy does not affect the book."""
        return dict(self._ratings)

    def is_fbs(self, team_id: int, season: int) -> bool:
        """True if the team played that season as FBS."""
        return self._classification.get((team_id, season)) == FBS

    def rating_for(self, team_id: int, season: int, first_season: int) -> float:
        """Return a team's current rating, initialising it on first FBS appearance.

        Args:
            team_id: The team.
            season: The season the game belongs to, which is what decides the team's
                subdivision.
            first_season: The first season in the data. A team already FBS then is an
                incumbent and starts at ``params.initial``; anyone appearing later is a
                newcomer and starts lower.

        Returns:
            The rating to use for this game. Always ``params.fcs`` for an FCS team.
        """
        if not self.is_fbs(team_id, season):
            return self._params.fcs
        if team_id not in self._ratings:
            start = self._params.initial if season == first_season else self._params.newcomer
            self._ratings[team_id] = start
            self.promotions.append((season, team_id, start))
        return self._ratings[team_id]

    def apply(self, team_id: int, season: int, rating: float) -> None:
        """Store a post-game rating, ignoring the write for an FCS opponent.

        The asymmetry is the FCS policy: 1200 is a fixed reference point, so the FBS side
        of the game updates and the FCS side does not.

        Args:
            team_id: The team.
            season: The game's season.
            rating: The post-game rating from :func:`cfb.elo.engine.update`.
        """
        if self.is_fbs(team_id, season):
            self._ratings[team_id] = rating

    def start_season(self, season: int) -> None:
        """Regress every tracked rating toward the mean, and drop teams that left FBS.

        Args:
            season: The season now beginning.
        """
        for team_id in sorted(self._ratings):
            if self._classification.get((team_id, season), FBS) != FBS:
                # Left FBS. Its rating is discarded rather than frozen: if it ever came
                # back, a rating years out of date would be worse than the newcomer prior.
                del self._ratings[team_id]
                self.demotions.append((season, team_id))
            else:
                self._ratings[team_id] = run_season_regression(self._ratings[team_id], self._params)


def run_elo(
    games: Iterable[Game],
    classification: Mapping[tuple[int, int], str],
    params: EloParams,
) -> EloRun:
    """Walk the schedule in chronological order, snapshotting ratings before each game.

    **Read this loop slowly — it is one of the four silent-error sites in the operating
    guide.** The order of the three steps per game is the whole phase:

    1. read both ratings (this is what gets stored),
    2. write the snapshot,
    3. only then apply the result.

    Swap 2 and 3 and every rating in the table knows the score of its own game, with no
    test failing unless one is written to look for it. ``tests/test_elo_chronology.py``
    is written to look for it, from both directions.

    Games are re-sorted here rather than trusted from the caller. The sort is the
    invariant; a caller that happens to pass rows in the right order is a coincidence, and
    ``tune.py`` calls this a few hundred times.

    Args:
        games: Games to process. Order is irrelevant — they are sorted internally.
        classification: ``(team_id, season)`` to subdivision.
        params: Rating parameters.

    Returns:
        The completed run.
    """
    ordered = sorted(games, key=lambda game: (game.start_date, game.game_id))
    book = RatingBook(classification, params)
    rows: list[PregameRow] = []
    season_end: dict[int, dict[int, float]] = {}

    if not ordered:
        return EloRun([], {}, {}, [], [])

    first_season = min(game.season for game in ordered)
    current_season = ordered[0].season

    for game in ordered:
        if game.season != current_season:
            # Seasons never run backwards in kickoff order (bowl games of season S kick
            # off before week 1 of S+1), so a change of season is a season boundary.
            season_end[current_season] = book.ratings
            book.start_season(game.season)
            current_season = game.season

        home_pre = book.rating_for(game.home_team_id, game.season, first_season)
        away_pre = book.rating_for(game.away_team_id, game.season, first_season)

        rows.append(
            PregameRow(
                game_id=game.game_id,
                season=game.season,
                home_team_id=game.home_team_id,
                away_team_id=game.away_team_id,
                home_elo_pre=home_pre,
                away_elo_pre=away_pre,
                home_is_fbs=book.is_fbs(game.home_team_id, game.season),
                away_is_fbs=book.is_fbs(game.away_team_id, game.season),
                neutral_site=game.neutral_site,
                home_points=game.home_points,
                away_points=game.away_points,
            )
        )

        # --- nothing above this line may depend on the result of this game -------------
        if not game.has_result:
            continue
        home_post, away_post = update(
            home_pre,
            away_pre,
            game.home_points,
            game.away_points,
            params,
            neutral_site=game.neutral_site,
        )
        book.apply(game.home_team_id, game.season, home_post)
        book.apply(game.away_team_id, game.season, away_post)

    season_end[current_season] = book.ratings
    return EloRun(rows, book.ratings, season_end, book.promotions, book.demotions)


# --- database -----------------------------------------------------------------


def load_games(conn: sqlite3.Connection, through_season: int | None = None) -> list[Game]:
    """Read the game spine in kickoff order.

    Args:
        conn: Open connection to the built database.
        through_season: Stop after this season. Used by the tuning grid, which has no
            business reading past the training boundary, and by the chronology test.

    Returns:
        Games ordered by ``(start_date, game_id)``.

    Raises:
        ValueError: If any kickoff is stored in an offset other than UTC, which would
            break the ordering the whole phase rests on.
    """
    where = "" if through_season is None else "WHERE season <= ?"
    params: tuple[Any, ...] = () if through_season is None else (through_season,)
    rows = conn.execute(
        f"""
        SELECT game_id, season, start_date, neutral_site,
               home_team_id, away_team_id, home_points, away_points, completed
        FROM games {where}
        ORDER BY start_date, game_id
        """,
        params,
    ).fetchall()

    offenders = [row["game_id"] for row in rows if not row["start_date"].endswith(UTC_SUFFIX)]
    if offenders:
        raise ValueError(
            f"{len(offenders)} games have a non-UTC start_date, e.g. {offenders[:5]}. "
            "Chronological ordering is lexicographic on this column; mixed offsets would "
            "reorder the walk without failing anything."
        )

    return [
        Game(
            game_id=row["game_id"],
            season=row["season"],
            start_date=row["start_date"],
            neutral_site=bool(row["neutral_site"]),
            home_team_id=row["home_team_id"],
            away_team_id=row["away_team_id"],
            home_points=row["home_points"],
            away_points=row["away_points"],
            completed=bool(row["completed"]),
        )
        for row in rows
    ]


def load_classifications(conn: sqlite3.Connection) -> dict[tuple[int, int], str]:
    """Read each team's subdivision per season.

    Args:
        conn: Open connection.

    Returns:
        ``(team_id, season)`` to ``"fbs"`` or ``"fcs"``.
    """
    return {
        (row["team_id"], row["season"]): row["classification"]
        for row in conn.execute(
            "SELECT team_id, season, classification FROM team_seasons "
            "WHERE classification IS NOT NULL"
        )
    }


def write_elo_pregame(conn: sqlite3.Connection, rows: Sequence[PregameRow]) -> int:
    """Replace ``elo_pregame`` with the given rows.

    Rebuilt from empty every time. A partial refresh could leave a row computed under a
    previous parameter set sitting next to rows computed under the current one.

    Args:
        conn: Open connection.
        rows: Rows from :func:`run_elo`.

    Returns:
        The number of rows written.
    """
    conn.executescript(SCHEMA_SQL)
    conn.execute("DELETE FROM elo_pregame")
    conn.executemany(
        "INSERT INTO elo_pregame (game_id, home_elo_pre, away_elo_pre) VALUES (?, ?, ?)",
        [(row.game_id, row.home_elo_pre, row.away_elo_pre) for row in rows],
    )
    conn.commit()
    return len(rows)


def build(conn: sqlite3.Connection, params: EloParams) -> EloRun:
    """Run Elo over the whole database and write the table.

    Args:
        conn: Open connection to the built database.
        params: Rating parameters, normally loaded from ``elo_params.json``.

    Returns:
        The run, for the caller to report on.
    """
    run = run_elo(load_games(conn), load_classifications(conn), params)
    write_elo_pregame(conn, run.rows)
    return run


# --- scoring and reporting ----------------------------------------------------


def elo_probability(row: PregameRow, params: EloParams) -> float:
    """Elo-only home win probability: a logistic of the rating gap plus home advantage.

    This is the Elo-only baseline of exit criterion 3 and the objective the grid search
    minimises. It is not a model — no features, no fitting beyond three parameters — and
    it is expected to land well short of the Vegas benchmark.

    Args:
        row: A pre-game row.
        params: Rating parameters; uses ``hfa``.

    Returns:
        The probability the home team wins.
    """
    return expected(
        row.home_elo_pre,
        row.away_elo_pre,
        0.0 if row.neutral_site else params.hfa,
    )


def scoreable(
    rows: Iterable[PregameRow],
    seasons: Iterable[int] | None = None,
    both_fbs: bool = True,
) -> list[PregameRow]:
    """Filter rows down to games that can be scored.

    Args:
        rows: Rows from a run.
        seasons: Restrict to these seasons; None means all.
        both_fbs: Keep only FBS-vs-FBS games. True for every objective and headline
            number in this project: FCS games are foregone conclusions that would flatter
            any metric they are averaged into, and ``RISKS.md`` #3 keeps them out of
            training as well.

    Returns:
        Games with a result, matching the filters.
    """
    allowed = None if seasons is None else set(seasons)
    return [
        row
        for row in rows
        if row.home_won is not None
        and (allowed is None or row.season in allowed)
        and (row.both_fbs or not both_fbs)
    ]


def score(rows: Sequence[PregameRow], params: EloParams) -> dict[str, float]:
    """Brier score and log loss for the Elo-only predictions on the given rows.

    Args:
        rows: Games to score, already filtered by :func:`scoreable`.
        params: Rating parameters.

    Returns:
        ``{"n", "brier", "log_loss"}``.

    Raises:
        ValueError: If ``rows`` is empty — an empty sample scores perfectly, which is the
            kind of vacuous pass this project's tests exist to prevent.
    """
    if not rows:
        raise ValueError("no scoreable games")
    probabilities = [elo_probability(row, params) for row in rows]
    outcomes = [row.home_won for row in rows]
    return {
        "n": float(len(rows)),
        "brier": brier_score(probabilities, outcomes),
        "log_loss": log_loss(probabilities, outcomes),
    }


def home_baseline(train_rows: Sequence[PregameRow], rows: Sequence[PregameRow]) -> dict[str, float]:
    """Score the naive baseline: predict the training home-win rate for every game.

    The floor of the results table. Elo means nothing as a number on its own — it means
    something as a point between "home teams win about 57% of the time" and the closing
    line, and the operating guide asks for both ends beside every result.

    The constant comes from the training seasons even when scoring later ones. Using each
    season's own home-win rate would be a season-level aggregate applied to games inside
    that season, which is the exact leakage pattern ``CLAUDE.md`` names.

    Args:
        train_rows: Scoreable training-season games, which supply the constant.
        rows: Games to score.

    Returns:
        ``{"rate", "brier", "log_loss"}``.
    """
    rate = sum(row.home_won for row in train_rows) / len(train_rows)
    outcomes = [row.home_won for row in rows]
    return {
        "rate": rate,
        "brier": brier_score([rate] * len(rows), outcomes),
        "log_loss": log_loss([rate] * len(rows), outcomes),
    }


def vegas_comparison(
    conn: sqlite3.Connection, rows: Sequence[PregameRow], params: EloParams
) -> dict[str, float] | None:
    """Score Elo and the Vegas benchmark on the identical set of games.

    Exit criterion 3 says Elo-only should be clearly *worse* than Vegas, and that only
    means anything if both are measured on the same games — the benchmark is missing 96
    games with no line (``RISKS.md`` #10).

    Args:
        conn: Open connection with ``vegas_benchmark`` built.
        rows: Scoreable rows.
        params: Rating parameters.

    Returns:
        Metrics for both, or None if the benchmark table is absent.
    """
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vegas_benchmark'"
    ).fetchone()
    if not exists:
        return None

    benchmark = dict(conn.execute("SELECT game_id, p_home_devig FROM vegas_benchmark"))
    shared = [row for row in rows if row.game_id in benchmark]
    if not shared:
        return None

    outcomes = [row.home_won for row in shared]
    elo_probabilities = [elo_probability(row, params) for row in shared]
    vegas_probabilities = [benchmark[row.game_id] for row in shared]
    return {
        "n": float(len(shared)),
        "elo_brier": brier_score(elo_probabilities, outcomes),
        "elo_log_loss": log_loss(elo_probabilities, outcomes),
        "vegas_brier": brier_score(vegas_probabilities, outcomes),
        "vegas_log_loss": log_loss(vegas_probabilities, outcomes),
    }


def sanity_report(conn: sqlite3.Connection, run: EloRun, params: EloParams, top: int = 5) -> str:
    """Render the eyeball checks for exit criterion 3.

    Args:
        conn: Open connection, for team names and the Vegas comparison.
        run: A completed run.
        params: The parameters it used.
        top: How many teams to list per season.

    Returns:
        A printable report.
    """
    names = dict(conn.execute("SELECT team_id, school FROM teams"))
    out = ["", f"Top {top} by end-of-season Elo (before the next season's regression)", "-" * 78]
    for season in sorted(run.season_end_ratings):
        ranked = sorted(run.season_end_ratings[season].items(), key=lambda kv: -kv[1])[:top]
        listed = ", ".join(f"{names.get(tid, tid)} {rating:.0f}" for tid, rating in ranked)
        out.append(f"  {season}  {listed}")

    out += ["", "Elo-only accuracy, FBS vs FBS only (exit criterion 3)", "-" * 78]
    out.append(f"{'season':>6}  {'games':>6}  {'brier':>7}  {'log loss':>9}")
    for season in sorted({row.season for row in run.rows}):
        rows = scoreable(run.rows, seasons=[season])
        metrics = score(rows, params)
        out.append(
            f"{season:>6}  {int(metrics['n']):>6}  {metrics['brier']:>7.4f}  "
            f"{metrics['log_loss']:>9.4f}"
        )

    train = [s for s in config.SEASONS if s <= config.TRAIN_LAST_SEASON]
    test = [s for s in config.SEASONS if s > config.TRAIN_LAST_SEASON]
    aggregates = (
        (f"train {train[0]}-{train[-1]}", train),
        (f"held out {test[0]}-{test[-1]}", test),
    )
    for label, seasons in aggregates:
        rows = scoreable(run.rows, seasons=seasons)
        metrics = score(rows, params)
        out.append(
            f"  {label:<22} n={int(metrics['n']):<6} brier={metrics['brier']:.4f}  "
            f"log loss={metrics['log_loss']:.4f}"
        )

    out += ["", "Where Elo sits: naive baseline -> Elo -> the closing line, same games", "-" * 78]
    train_rows = scoreable(run.rows, seasons=train)
    for label, seasons in (("train", train), ("held out", test)):
        rows = scoreable(run.rows, seasons=seasons)
        comparison = vegas_comparison(conn, rows, params)
        if comparison is None:
            out.append("  vegas_benchmark not built; run `make benchmark`")
            break
        naive = home_baseline(train_rows, rows)
        gap = naive["brier"] - comparison["vegas_brier"]
        closed = (naive["brier"] - comparison["elo_brier"]) / gap
        out += [
            f"  {label} (n={int(comparison['n'])})",
            f"    naive home {naive['rate']:.4f}   brier={naive['brier']:.4f}  "
            f"log loss={naive['log_loss']:.4f}",
            f"    elo only                brier={comparison['elo_brier']:.4f}  "
            f"log loss={comparison['elo_log_loss']:.4f}",
            f"    vegas closing line      brier={comparison['vegas_brier']:.4f}  "
            f"log loss={comparison['vegas_log_loss']:.4f}",
            f"    Elo closes {100 * closed:.0f}% of the naive-to-Vegas Brier gap",
        ]
    out.append(
        "  Elo-only is expected to sit clearly short of the line. If it ever matches, "
        "the prior is leakage, not a result."
    )

    out += ["", "Subdivision bookkeeping", "-" * 78]
    incumbents = [p for p in run.promotions if p[0] == min(run.season_end_ratings)]
    newcomers = [p for p in run.promotions if p[0] != min(run.season_end_ratings)]
    out.append(
        f"  initialised at {params.initial:.0f} in the first season : {len(incumbents)} teams"
    )
    for season, team_id, rating in newcomers:
        out.append(f"  entered FBS in {season} at {rating:.0f} : {names.get(team_id, team_id)}")
    for season, team_id in run.demotions:
        out.append(f"  left FBS in {season}, rating discarded  : {names.get(team_id, team_id)}")
    fcs_games = sum(1 for row in run.rows if not row.both_fbs)
    out.append(f"  games against a fixed-{params.fcs:.0f} FCS opponent : {fcs_games}")
    return "\n".join(out)


def load_params() -> tuple[EloParams, str]:
    """Load the frozen parameters, falling back to the plan's starting values.

    Returns:
        ``(params, source)`` where ``source`` names where they came from, so a run can
        never leave you guessing whether it used tuned or default values.
    """
    if config.ELO_PARAMS_PATH.exists():
        values = json.loads(config.ELO_PARAMS_PATH.read_text())
        return EloParams.from_dict(values.get("params", values)), str(config.ELO_PARAMS_PATH)
    return EloParams(), "engine defaults (elo_params.json not found; run `make elo-tune`)"


def main(argv: Sequence[str] | None = None) -> int:
    """Build ``elo_pregame`` and print the sanity report.

    Args:
        argv: Command-line arguments; None reads ``sys.argv``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description="Build the pre-game Elo table.")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="recompute and report without writing the table",
    )
    args = parser.parse_args(argv)

    if not config.DB_PATH.exists():
        raise SystemExit(f"no database at {config.DB_PATH}; run `make ingest` first")

    params, source = load_params()
    conn = connect(config.DB_PATH)
    try:
        print(f"parameters from {source}")
        print(
            f"  K={params.k:g}  HFA={params.hfa:g}  regression={params.regression:.4f}  "
            f"init={params.initial:g}  newcomer={params.newcomer:g}  fcs={params.fcs:g}"
        )
        games = load_games(conn)
        run = run_elo(games, load_classifications(conn), params)
        if args.report_only:
            print(f"computed {len(run.rows)} pre-game rows (not written)")
        else:
            written = write_elo_pregame(conn, run.rows)
            print(f"wrote elo_pregame: {written} of {len(games)} games")
        print(sanity_report(conn, run, params))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
