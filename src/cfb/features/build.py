"""Feature computation for Phase 4, built around one cutoff rule.

**Every feature for a game uses only information available strictly before that game's
kickoff.** This module is arranged so that the rule lives in exactly one function,
:func:`priors_before`, rather than being re-implemented inside fifteen queries:

1. **Load** — SQL runs once, up front, and produces plain dataclasses. The relation names
   are a parameter (:class:`Relations`), which is what lets ``audit.py`` point the same
   loaders at a truncated SQL view instead of the base tables.
2. **Select** — :func:`priors_before` filters a team's games to those that kicked off
   before the target game. This is the shift-by-one, spelled once.
3. **Compute** — the feature functions are pure and never touch the database. Given the
   same prior list they return the same numbers, so what the leakage audit is really
   testing is step 2.

The consequence worth stating: a leak has to get through :func:`priors_before` to exist.
``tests/test_shift_by_one.py`` attacks that function directly and
``src/cfb/features/audit.py`` attacks it end to end by rebuilding the prior set from a
truncated view of the database and demanding the same answer.

Two things this module deliberately does **not** read:

* ``lines`` and ``vegas_benchmark``. The closing line is the benchmark, never an input
  (``CLAUDE.md``). :func:`cfb.features.audit.assert_no_market_source` checks this
  mechanically rather than trusting anyone to remember it.
* Any column of the target game's own box score. ``game_team_stats`` is read only for
  games that are already in the prior set.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from cfb import config
from cfb.elo.pipeline import FBS, load_classifications
from cfb.ingest.schema import connect

REST_DAYS_CAP: int = 30
"""Rest days are capped here, which collapses two cases into one.

A season opener follows an offseason of roughly 240 days, and the very first game in the
data has no previous game at all. Physically both mean the same thing — fully rested — and
an uncapped 240 would tower over every in-season value. Confirmed by Sean; logged in
``DECISIONS.md``.
"""

MIN_PRIOR_GAMES: int = 1
"""Rolling means need at least this many prior games in the season, else null.

Week 1 therefore has null rolling stats for both teams, by design. The nulls are kept and
documented — back-filling them from later games is the leakage bug this phase exists to
prevent.
"""


# --- what the features are, in one table --------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    """One column of the feature store, and everything said about it.

    This table is the single source of truth for three things that must not drift apart:
    the columns :func:`build_frame` writes, the comparison rule ``audit.py`` applies to
    each column, and the documentation ``docs.py`` renders into ``FEATURES.md``.

    Attributes:
        name: Column name in the parquet store.
        kind: ``"exact"`` for ints, flags and strings; ``"float"`` for anything compared
            under a tolerance. The audit reads this to decide how to compare.
        nullable: Whether the column may legitimately be null.
        definition: What the number is.
        depends_on: The latest information the value can depend on. Every entry here must
            resolve to something strictly before kickoff — that is exit criterion 3.
        null_policy: When the value is null and why it is not filled.
    """

    name: str
    kind: str
    nullable: bool
    definition: str
    depends_on: str
    null_policy: str


KEY_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "game_id", "exact", False, "CFBD game identifier.", "scheduling time", "never null"
    ),
    FeatureSpec(
        "season", "exact", False, "Season the game belongs to.", "scheduling time", "never null"
    ),
    FeatureSpec(
        "start_date",
        "exact",
        False,
        "Kickoff, UTC. The leakage clock every cutoff in this project is measured against.",
        "scheduling time",
        "never null",
    ),
    FeatureSpec("home_team_id", "exact", False, "Home team.", "scheduling time", "never null"),
    FeatureSpec("away_team_id", "exact", False, "Away team.", "scheduling time", "never null"),
)

FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "home_elo_pre",
        "float",
        False,
        "Home team's Elo rating carried into this game.",
        "the last game either team played before kickoff (Phase 3 walks in kickoff order "
        "and snapshots before applying a result)",
        "never null — every team has a rating on first appearance",
    ),
    FeatureSpec(
        "away_elo_pre",
        "float",
        False,
        "Away team's Elo rating carried into this game.",
        "as home_elo_pre",
        "never null",
    ),
    FeatureSpec(
        "elo_diff", "float", False, "home_elo_pre − away_elo_pre.", "as home_elo_pre", "never null"
    ),
    FeatureSpec("week", "exact", False, "Week of the season.", "scheduling time", "never null"),
    FeatureSpec(
        "season_type",
        "exact",
        False,
        "'regular', 'postseason' or 'spring_regular'.",
        "scheduling time",
        "never null",
    ),
    FeatureSpec(
        "neutral_site",
        "exact",
        False,
        "1 if neither team is hosting.",
        "scheduling time",
        "never null",
    ),
    FeatureSpec(
        "conference_game",
        "exact",
        False,
        "1 if both teams share a conference.",
        "scheduling time",
        "never null in the built database",
    ),
    FeatureSpec(
        "fcs_opponent",
        "exact",
        False,
        "1 if one of the two teams played that season as FCS.",
        "scheduling time plus the season's subdivision table",
        "never null",
    ),
    FeatureSpec(
        "rest_days_home",
        "exact",
        False,
        f"UTC calendar days since the home team's previous completed game, capped at "
        f"{REST_DAYS_CAP}.",
        "the home team's previous completed game",
        f"never null — no previous game yields the {REST_DAYS_CAP}-day cap",
    ),
    FeatureSpec(
        "rest_days_away",
        "exact",
        False,
        "As rest_days_home, for the away team.",
        "the away team's previous completed game",
        f"never null — cap is {REST_DAYS_CAP}",
    ),
    FeatureSpec(
        "rest_diff",
        "exact",
        False,
        "rest_days_home − rest_days_away.",
        "both teams' previous completed games",
        "never null",
    ),
    FeatureSpec(
        "off_ppg_roll_home",
        "float",
        True,
        "Mean points scored by the home team over its previous games this season.",
        "the home team's last completed game before kickoff",
        f"null with fewer than {MIN_PRIOR_GAMES} prior game this season (all of week 1)",
    ),
    FeatureSpec(
        "def_ppg_roll_home",
        "float",
        True,
        "Mean points allowed by the home team over its previous games this season.",
        "the home team's last completed game before kickoff",
        f"null with fewer than {MIN_PRIOR_GAMES} prior game this season",
    ),
    FeatureSpec(
        "off_ypp_roll_home",
        "float",
        True,
        "Home team's yards per play this season: total yards gained ÷ total plays run, "
        "summed over previous games.",
        "the home team's last completed game before kickoff that has a box score",
        "null with no prior game carrying a box score (RISKS #11)",
    ),
    FeatureSpec(
        "def_ypp_roll_home",
        "float",
        True,
        "Yards per play allowed by the home team: opponents' total yards ÷ opponents' "
        "total plays, summed over previous games.",
        "the home team's last completed game before kickoff that has a box score",
        "null with no prior game carrying a box score",
    ),
    FeatureSpec(
        "pace_roll_home",
        "float",
        True,
        "Mean plays run per game by the home team over its previous games this season.",
        "the home team's last completed game before kickoff that has a box score",
        "null with no prior game carrying a box score",
    ),
    FeatureSpec(
        "off_ppg_roll_away",
        "float",
        True,
        "As off_ppg_roll_home, for the away team.",
        "the away team's last completed game before kickoff",
        "as the home column",
    ),
    FeatureSpec(
        "def_ppg_roll_away",
        "float",
        True,
        "As def_ppg_roll_home, for the away team.",
        "the away team's last completed game before kickoff",
        "as the home column",
    ),
    FeatureSpec(
        "off_ypp_roll_away",
        "float",
        True,
        "As off_ypp_roll_home, for the away team.",
        "the away team's last completed game before kickoff that has a box score",
        "as the home column",
    ),
    FeatureSpec(
        "def_ypp_roll_away",
        "float",
        True,
        "As def_ypp_roll_home, for the away team.",
        "the away team's last completed game before kickoff that has a box score",
        "as the home column",
    ),
    FeatureSpec(
        "pace_roll_away",
        "float",
        True,
        "As pace_roll_home, for the away team.",
        "the away team's last completed game before kickoff that has a box score",
        "as the home column",
    ),
    FeatureSpec(
        "prev_season_win_pct_home",
        "float",
        True,
        "Home team's win percentage over the previous season's completed games.",
        "the home team's last game of the previous season",
        "null for a team with no games in the previous season (2014, and teams entering FBS)",
    ),
    FeatureSpec(
        "prev_season_win_pct_away",
        "float",
        True,
        "As prev_season_win_pct_home, for the away team.",
        "the away team's last game of the previous season",
        "as the home column",
    ),
    FeatureSpec(
        "prior_games_home",
        "exact",
        False,
        "Number of the home team's completed games this season before kickoff. The witness "
        "for every rolling null in the row.",
        "the home team's last completed game before kickoff",
        "never null; 0 in week 1",
    ),
    FeatureSpec(
        "prior_games_away",
        "exact",
        False,
        "As prior_games_home, for the away team.",
        "the away team's last completed game before kickoff",
        "never null; 0 in week 1",
    ),
    FeatureSpec(
        "fcs_games_in_window_home",
        "exact",
        False,
        "How many of the home team's prior games this season were against an FCS opponent. "
        "Those games are included in the rolling means, so this column exposes how much of "
        "a rolling value came from them.",
        "the home team's last completed game before kickoff",
        "never null; 0 when there were none",
    ),
    FeatureSpec(
        "fcs_games_in_window_away",
        "exact",
        False,
        "As fcs_games_in_window_home, for the away team.",
        "the away team's last completed game before kickoff",
        "never null",
    ),
    FeatureSpec(
        "as_of",
        "exact",
        True,
        "Kickoff of the latest game any rolling or rest feature in this row read. Asserted "
        "strictly less than start_date for every row.",
        "itself — this is the measured answer to 'how late is the latest input?'",
        "null only when neither team had played before (the first games of 2014)",
    ),
)

LABEL_SPEC = FeatureSpec(
    "label_home_win",
    "exact",
    False,
    "1 if the home team won. Present for convenience; used only by Phase 5's training "
    "code and never by feature computation.",
    "the game's own result — this is the label, not a feature",
    "never null; the store holds completed games only",
)

ALL_SPECS: tuple[FeatureSpec, ...] = KEY_SPECS + FEATURE_SPECS + (LABEL_SPEC,)
COLUMNS: tuple[str, ...] = tuple(spec.name for spec in ALL_SPECS)
FEATURE_COLUMNS: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)


# --- loading ------------------------------------------------------------------


@dataclass(frozen=True)
class Relations:
    """Which relations the loaders read from.

    Production reads the base tables. The leakage audit swaps in SQL views that expose
    only the games that had kicked off before the game under test, so the identical
    loading and feature code runs against a truncated database.
    """

    games: str = "games"
    stats: str = "game_team_stats"


DEFAULT_RELATIONS = Relations()


@dataclass(frozen=True)
class ScheduledGame:
    """A game as the schedule knows it, before anybody looks at a box score."""

    game_id: int
    season: int
    week: int
    season_type: str
    start_date: str
    neutral_site: bool
    conference_game: bool
    home_team_id: int
    away_team_id: int
    home_points: int | None
    away_points: int | None
    completed: bool

    @property
    def has_result(self) -> bool:
        """True if the game was played and both scores are present."""
        return self.completed and self.home_points is not None and self.away_points is not None

    @property
    def home_win(self) -> int:
        """1 if the home team won, else 0.

        Raises:
            ValueError: If the game has no result, or was a tie. The database holds zero
                ties in 10,373 completed games, so a tie means something upstream changed
                and a silent 0 would be the wrong way to find out.
        """
        if not self.has_result:
            raise ValueError(f"game {self.game_id} has no result")
        if self.home_points == self.away_points:
            raise ValueError(f"game {self.game_id} is a tie; the binary label is undefined")
        return int(self.home_points > self.away_points)


@dataclass(frozen=True)
class TeamGame:
    """One team's side of one completed game — the unit the rolling features average.

    Both the team's own production and what it conceded are on the same record, so a
    defensive feature never has to go looking for the opponent's row and risk picking up
    the wrong game.
    """

    game_id: int
    season: int
    start_date: str
    team_id: int
    opponent_id: int
    opponent_is_fcs: bool
    points_for: int
    points_against: int
    yards_for: float | None
    plays_for: float | None
    yards_against: float | None
    plays_against: float | None

    @property
    def won(self) -> float:
        """1.0 for a win, 0.0 for a loss, 0.5 for a tie."""
        if self.points_for > self.points_against:
            return 1.0
        return 0.0 if self.points_for < self.points_against else 0.5

    @property
    def has_box_score(self) -> bool:
        """True if both sides' yards and plays are known for this game."""
        return None not in (self.yards_for, self.plays_for, self.yards_against, self.plays_against)


def load_schedule(
    conn: sqlite3.Connection, relations: Relations = DEFAULT_RELATIONS
) -> list[ScheduledGame]:
    """Read the game spine in kickoff order.

    Args:
        conn: Open connection.
        relations: Which relations to read. The audit passes truncated views.

    Returns:
        Games ordered by ``(start_date, game_id)``.
    """
    rows = conn.execute(
        f"""
        SELECT game_id, season, week, season_type, start_date, neutral_site,
               conference_game, home_team_id, away_team_id, home_points, away_points, completed
        FROM {relations.games}
        ORDER BY start_date, game_id
        """  # noqa: S608 - relation names are module constants, never user input
    ).fetchall()
    return [
        ScheduledGame(
            game_id=row["game_id"],
            season=row["season"],
            week=row["week"],
            season_type=row["season_type"],
            start_date=row["start_date"],
            neutral_site=bool(row["neutral_site"]),
            conference_game=bool(row["conference_game"]),
            home_team_id=row["home_team_id"],
            away_team_id=row["away_team_id"],
            home_points=row["home_points"],
            away_points=row["away_points"],
            completed=bool(row["completed"]),
        )
        for row in rows
    ]


def load_box_scores(
    conn: sqlite3.Connection, relations: Relations = DEFAULT_RELATIONS
) -> dict[tuple[int, int], tuple[float | None, float | None]]:
    """Read total yards and plays per team per game.

    Plays are not a stored statistic — they are reconstructed here, which is the Phase 4
    decision ``RISKS.md`` #14 deferred. See :func:`plays_from`.

    Args:
        conn: Open connection.
        relations: Which relations to read.

    Returns:
        ``(game_id, team_id)`` to ``(total_yards, plays)``. Either element may be None
        when the underlying statistic is absent.
    """
    rows = conn.execute(
        f"""
        SELECT game_id, team_id,
               MAX(CASE WHEN stat_name = 'totalYards'          THEN stat_value END) AS total_yards,
               MAX(CASE WHEN stat_name = 'rushingAttempts'     THEN stat_value END) AS rush_att,
               MAX(CASE WHEN stat_name = 'completionAttempts'  THEN stat_raw   END) AS comp_att
        FROM {relations.stats}
        GROUP BY game_id, team_id
        """  # noqa: S608 - relation names are module constants, never user input
    ).fetchall()
    return {
        (row["game_id"], row["team_id"]): (
            row["total_yards"],
            plays_from(row["rush_att"], row["comp_att"]),
        )
        for row in rows
    }


def load_elo(conn: sqlite3.Connection) -> dict[int, tuple[float, float]]:
    """Read the pre-game ratings Phase 3 wrote.

    Not parameterised by :class:`Relations`: the audit does not read this table under
    truncation, it re-runs the Phase 3 walk over the truncated schedule and compares. A
    stored table that merely agreed with itself would prove nothing.

    Args:
        conn: Open connection with ``elo_pregame`` built.

    Returns:
        ``game_id`` to ``(home_elo_pre, away_elo_pre)``.

    Raises:
        RuntimeError: If the table is missing.
    """
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='elo_pregame'"
    ).fetchone()
    if not exists:
        raise RuntimeError("elo_pregame is missing; run `make elo` before building features")
    return {
        row["game_id"]: (row["home_elo_pre"], row["away_elo_pre"])
        for row in conn.execute("SELECT game_id, home_elo_pre, away_elo_pre FROM elo_pregame")
    }


# --- pure feature functions ---------------------------------------------------


def parse_pass_attempts(raw: str) -> int:
    """Pull the attempts out of a ``"completions-attempts"`` string.

    ``completionAttempts`` is one of five composite box-score fields stored as text with a
    null ``stat_value`` (``RISKS.md`` #14). Decomposing it is the price of a plays count.

    Args:
        raw: A string like ``"21-31"``.

    Returns:
        The attempts.

    Raises:
        ValueError: If the string is not two integers separated by a hyphen. Ingestion
            stored these verbatim, so a new shape means the upstream format moved and
            guessing at it would silently corrupt every pace and yards-per-play value.
    """
    parts = raw.split("-")
    if len(parts) != 2:
        raise ValueError(f"completionAttempts is not 'completions-attempts': {raw!r}")
    try:
        return int(parts[1])
    except ValueError as error:
        raise ValueError(f"completionAttempts has a non-integer attempts field: {raw!r}") from error


def plays_from(rush_attempts: float | None, completion_attempts: str | None) -> float | None:
    """Reconstruct offensive plays from rushing attempts and pass attempts.

    Plays are ``rushingAttempts + passAttempts`` and nothing else. NCAA scoring charges
    sack yardage to rushing, so a sack is already counted as a rushing attempt — adding
    sacks separately would double-count them. Logged in ``DECISIONS.md``.

    Args:
        rush_attempts: ``rushingAttempts`` for the team in that game.
        completion_attempts: The raw ``"C-A"`` string, or None if absent.

    Returns:
        The play count, or None if either input is missing. Never a partial sum: a plays
        figure missing its passing half would look plausible and be wrong.
    """
    if rush_attempts is None or completion_attempts is None:
        return None
    return float(rush_attempts) + parse_pass_attempts(completion_attempts)


def mean(values: Sequence[float]) -> float | None:
    """Arithmetic mean, or None for an empty sequence.

    Args:
        values: The values.

    Returns:
        The mean, or None.
    """
    return sum(values) / len(values) if values else None


def rate(numerators: Sequence[float], denominators: Sequence[float]) -> float | None:
    """Ratio of sums — the volume-weighted rate over a window.

    Deliberately not the mean of per-game ratios. A 45-play game and a 90-play game say
    different amounts about a team's yards per play, and averaging the two ratios would
    weight them equally. Logged in ``DECISIONS.md``.

    Args:
        numerators: Per-game numerators, e.g. yards.
        denominators: Per-game denominators, e.g. plays. Same length as ``numerators``.

    Returns:
        ``sum(numerators) / sum(denominators)``, or None if empty or the denominator sums
        to zero.
    """
    total = sum(denominators)
    return sum(numerators) / total if denominators and total else None


def calendar_days_between(earlier: str, later: str) -> int:
    """UTC calendar days from one kickoff to another.

    Calendar days rather than elapsed 24-hour periods, so a Saturday-to-Saturday gap reads
    as 7 whatever time the two games kicked off. Both timestamps are UTC (asserted in
    Phase 3's loader), so the two are measured on the same clock.

    Args:
        earlier: The earlier kickoff, ISO-8601 with a UTC offset.
        later: The later kickoff.

    Returns:
        The number of days.

    Raises:
        ValueError: If ``later`` precedes ``earlier``. Callers pass a prior game and its
            successor, so a negative gap means the chronology broke upstream.
    """
    start: date = datetime.fromisoformat(earlier).date()
    end: date = datetime.fromisoformat(later).date()
    days = (end - start).days
    if days < 0:
        raise ValueError(f"{later} precedes {earlier}; the game order is wrong")
    return days


def rest_days(kickoff: str, previous_kickoff: str | None) -> int:
    """Days of rest before a game, capped.

    Args:
        kickoff: The game's kickoff.
        previous_kickoff: The team's previous completed game, or None if it has none.

    Returns:
        Calendar days since the previous game, capped at :data:`REST_DAYS_CAP`. A team
        with no previous game gets the cap: an offseason and a first-ever game both mean
        "fully rested", and the cap is what stops a 240-day opener from dwarfing every
        in-season value.
    """
    if previous_kickoff is None:
        return REST_DAYS_CAP
    return min(calendar_days_between(previous_kickoff, kickoff), REST_DAYS_CAP)


def priors_before(team_games: Sequence[TeamGame], kickoff: str) -> list[TeamGame]:
    """A team's completed games that kicked off strictly before ``kickoff``.

    **This is the leakage boundary of Phase 4 and it is written once.** The comparison is
    strict: a game starting at the same instant is not prior. ``start_date`` is stored in a
    single UTC offset, so the string comparison is a chronological one.

    Args:
        team_games: That team's completed games, in any order.
        kickoff: The target game's kickoff.

    Returns:
        The prior games, in kickoff order.
    """
    return sorted(
        (game for game in team_games if game.start_date < kickoff),
        key=lambda game: (game.start_date, game.game_id),
    )


@dataclass(frozen=True)
class TeamFeatures:
    """Everything one side of a matchup contributes to a feature row."""

    rest_days: int
    off_ppg_roll: float | None
    def_ppg_roll: float | None
    off_ypp_roll: float | None
    def_ypp_roll: float | None
    pace_roll: float | None
    prev_season_win_pct: float | None
    prior_games: int
    fcs_games_in_window: int
    as_of: str | None


def team_features(game: ScheduledGame, priors: Sequence[TeamGame]) -> TeamFeatures:
    """Compute one team's features for one game from its prior games.

    Pure: no database, no clock, no globals. Given the same priors it returns the same
    numbers, which is what makes the audit's recomputation a fair comparison.

    Args:
        game: The target game.
        priors: That team's completed games before kickoff, from :func:`priors_before`.

    Returns:
        The team's side of the feature row.

    Raises:
        ValueError: If any prior game starts at or after kickoff. The caller has already
            filtered, so this is the same rule asserted a second time at the point of use —
            cheap, and the one place a future game would have to slip through.
    """
    late = [prior.game_id for prior in priors if prior.start_date >= game.start_date]
    if late:
        raise ValueError(
            f"game {game.game_id}: {len(late)} 'prior' games kick off at or after it, "
            f"e.g. {late[:3]} — the shift has been dropped"
        )

    in_season = [prior for prior in priors if prior.season == game.season]
    last_season = [prior for prior in priors if prior.season == game.season - 1]
    boxed = [prior for prior in in_season if prior.has_box_score]

    enough = len(in_season) >= MIN_PRIOR_GAMES
    return TeamFeatures(
        rest_days=rest_days(game.start_date, priors[-1].start_date if priors else None),
        off_ppg_roll=mean([p.points_for for p in in_season]) if enough else None,
        def_ppg_roll=mean([p.points_against for p in in_season]) if enough else None,
        off_ypp_roll=rate([p.yards_for for p in boxed], [p.plays_for for p in boxed]),
        def_ypp_roll=rate([p.yards_against for p in boxed], [p.plays_against for p in boxed]),
        pace_roll=mean([p.plays_for for p in boxed]),
        prev_season_win_pct=mean([p.won for p in last_season]),
        prior_games=len(in_season),
        fcs_games_in_window=sum(1 for p in in_season if p.opponent_is_fcs),
        as_of=priors[-1].start_date if priors else None,
    )


# --- assembly -----------------------------------------------------------------


@dataclass(frozen=True)
class FeatureContext:
    """Everything loaded from the database, ready for the pure functions.

    Attributes:
        schedule: Every game in the source relation, in kickoff order.
        team_games: ``team_id`` to that team's completed games, in kickoff order.
        classification: ``(team_id, season)`` to subdivision.
    """

    schedule: list[ScheduledGame]
    team_games: dict[int, list[TeamGame]]
    classification: Mapping[tuple[int, int], str]


def build_team_games(
    schedule: Iterable[ScheduledGame],
    box_scores: Mapping[tuple[int, int], tuple[float | None, float | None]],
    classification: Mapping[tuple[int, int], str],
) -> dict[int, list[TeamGame]]:
    """Turn the game spine into two per-team records per game.

    Only completed games become :class:`TeamGame` records. A cancelled game
    (``RISKS.md`` #12) was never played, so it contributes no points, no rest and no
    rolling sample — but it stays in the schedule, because the teams still had a week off.

    Args:
        schedule: Games to expand.
        box_scores: Yards and plays per ``(game_id, team_id)``.
        classification: ``(team_id, season)`` to subdivision.

    Returns:
        ``team_id`` to that team's completed games, in kickoff order.
    """
    by_team: dict[int, list[TeamGame]] = {}
    for game in schedule:
        if not game.has_result:
            continue
        home_yards, home_plays = box_scores.get((game.game_id, game.home_team_id), (None, None))
        away_yards, away_plays = box_scores.get((game.game_id, game.away_team_id), (None, None))
        sides = (
            (
                game.home_team_id,
                game.away_team_id,
                game.home_points,
                game.away_points,
                home_yards,
                home_plays,
                away_yards,
                away_plays,
            ),
            (
                game.away_team_id,
                game.home_team_id,
                game.away_points,
                game.home_points,
                away_yards,
                away_plays,
                home_yards,
                home_plays,
            ),
        )
        for team_id, opponent_id, points_for, points_against, yf, pf, ya, pa in sides:
            by_team.setdefault(team_id, []).append(
                TeamGame(
                    game_id=game.game_id,
                    season=game.season,
                    start_date=game.start_date,
                    team_id=team_id,
                    opponent_id=opponent_id,
                    opponent_is_fcs=classification.get((opponent_id, game.season)) != FBS,
                    points_for=points_for,
                    points_against=points_against,
                    yards_for=yf,
                    plays_for=pf,
                    yards_against=ya,
                    plays_against=pa,
                )
            )
    for games in by_team.values():
        games.sort(key=lambda game: (game.start_date, game.game_id))
    return by_team


def load_context(
    conn: sqlite3.Connection, relations: Relations = DEFAULT_RELATIONS
) -> FeatureContext:
    """Run every query this module needs, once.

    Args:
        conn: Open connection.
        relations: Which relations to read. The audit passes truncated views.

    Returns:
        The loaded context.
    """
    schedule = load_schedule(conn, relations)
    classification = load_classifications(conn)
    return FeatureContext(
        schedule=schedule,
        team_games=build_team_games(schedule, load_box_scores(conn, relations), classification),
        classification=classification,
    )


def is_fcs_matchup(game: ScheduledGame, classification: Mapping[tuple[int, int], str]) -> bool:
    """True if either team played that season as FCS.

    Args:
        game: The game.
        classification: ``(team_id, season)`` to subdivision.

    Returns:
        Whether the matchup crosses the subdivision line.
    """
    return (
        classification.get((game.home_team_id, game.season)) != FBS
        or classification.get((game.away_team_id, game.season)) != FBS
    )


def feature_row(
    game: ScheduledGame,
    context: FeatureContext,
    elo: tuple[float, float],
    with_label: bool = True,
) -> dict[str, object]:
    """Build one row of the feature store.

    Args:
        game: The target game.
        context: Loaded data. Under the audit this holds only pre-kickoff games.
        elo: ``(home_elo_pre, away_elo_pre)`` for this game.
        with_label: Include ``label_home_win``. The audit builds rows without it — a
            truncated view has no result for the game under test, and asking for one would
            be exactly the question this module must never ask.

    Returns:
        A dict keyed by the columns of :data:`COLUMNS`.
    """
    home = team_features(
        game, priors_before(context.team_games.get(game.home_team_id, []), game.start_date)
    )
    away = team_features(
        game, priors_before(context.team_games.get(game.away_team_id, []), game.start_date)
    )
    home_elo, away_elo = elo

    row: dict[str, object] = {
        "game_id": game.game_id,
        "season": game.season,
        "start_date": game.start_date,
        "home_team_id": game.home_team_id,
        "away_team_id": game.away_team_id,
        "home_elo_pre": home_elo,
        "away_elo_pre": away_elo,
        "elo_diff": home_elo - away_elo,
        "week": game.week,
        "season_type": game.season_type,
        "neutral_site": int(game.neutral_site),
        "conference_game": int(game.conference_game),
        "fcs_opponent": int(is_fcs_matchup(game, context.classification)),
        "rest_days_home": home.rest_days,
        "rest_days_away": away.rest_days,
        "rest_diff": home.rest_days - away.rest_days,
        "prior_games_home": home.prior_games,
        "prior_games_away": away.prior_games,
        "fcs_games_in_window_home": home.fcs_games_in_window,
        "fcs_games_in_window_away": away.fcs_games_in_window,
        "as_of": max(
            [stamp for stamp in (home.as_of, away.as_of) if stamp is not None], default=None
        ),
    }
    for side, features in (("home", home), ("away", away)):
        row[f"off_ppg_roll_{side}"] = features.off_ppg_roll
        row[f"def_ppg_roll_{side}"] = features.def_ppg_roll
        row[f"off_ypp_roll_{side}"] = features.off_ypp_roll
        row[f"def_ypp_roll_{side}"] = features.def_ypp_roll
        row[f"pace_roll_{side}"] = features.pace_roll
        row[f"prev_season_win_pct_{side}"] = features.prev_season_win_pct
    if with_label:
        row["label_home_win"] = game.home_win
    return row


def build_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    """Compute the whole feature store.

    One row per completed game in the spine. That is every FBS-vs-FBS game plus every
    game against an FCS opponent, in either direction, flagged by ``fcs_opponent``. The
    plan named only FBS-hosted FCS games; the three FCS-hosted ones are included too
    because the asymmetry bought nothing — the FCS side is a fixed-1200 team with
    unobservable rolling stats whichever end of the field it is standing on. Logged in
    ``DECISIONS.md``. Phase 5 filters on the flag.

    Args:
        conn: Open connection to the built database, with ``elo_pregame`` present.

    Returns:
        The feature store, ordered by ``(start_date, game_id)``.

    Raises:
        RuntimeError: If a completed game has no Elo row, which would mean the two tables
            were built from different databases.
    """
    context = load_context(conn)
    elo = load_elo(conn)

    rows = []
    for game in context.schedule:
        if not game.has_result:
            continue
        if game.game_id not in elo:
            raise RuntimeError(
                f"game {game.game_id} has no elo_pregame row; rebuild with `make elo`"
            )
        rows.append(feature_row(game, context, elo[game.game_id]))
    return pd.DataFrame(rows, columns=list(COLUMNS))


def write_frame(frame: pd.DataFrame, path: Path | None = None) -> Path:
    """Write the feature store to parquet.

    A single file, not partitioned by season: ~9k rows does not need it, and one file is
    one thing to check the hash of. Logged in ``DECISIONS.md``.

    Args:
        frame: The frame from :func:`build_frame`.
        path: Destination; defaults to ``config.FEATURE_STORE_PATH``.

    Returns:
        The path written.
    """
    destination = path or config.FEATURE_STORE_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    return destination


def read_frame(path: Path | None = None) -> pd.DataFrame:
    """Read the feature store back.

    Args:
        path: Source; defaults to ``config.FEATURE_STORE_PATH``.

    Returns:
        The stored frame.

    Raises:
        FileNotFoundError: If the store has not been built.
    """
    source = path or config.FEATURE_STORE_PATH
    if not source.exists():
        raise FileNotFoundError(f"no feature store at {source}; run `make features` first")
    return pd.read_parquet(source)


def coverage_report(frame: pd.DataFrame) -> str:
    """Render per-season row counts and null counts.

    Nulls are a documented output of this phase, not a defect, so they are printed on
    every build rather than left for somebody to go looking for.

    Args:
        frame: The built frame.

    Returns:
        A printable report.
    """
    out = ["", f"Feature store: {len(frame)} rows, {len(FEATURE_COLUMNS)} features", "-" * 78]
    out.append(
        f"{'season':>6}  {'rows':>6}  {'fcs':>5}  {'week 1':>7}  {'null ppg':>9}  {'null ypp':>9}"
    )
    for season, block in frame.groupby("season", sort=True):
        out.append(
            f"{season:>6}  {len(block):>6}  {int(block['fcs_opponent'].sum()):>5}  "
            f"{int((block['prior_games_home'] == 0).sum()):>7}  "
            f"{int(block['off_ppg_roll_home'].isna().sum()):>9}  "
            f"{int(block['off_ypp_roll_home'].isna().sum()):>9}"
        )

    out += ["", "Nulls by column (kept and documented; never back-filled)", "-" * 78]
    for spec in FEATURE_SPECS:
        nulls = int(frame[spec.name].isna().sum())
        if nulls or spec.nullable:
            flag = "" if spec.nullable else "  <-- UNEXPECTED: spec says never null"
            out.append(f"  {spec.name:<28} {nulls:>6}{flag}")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the feature store and print the coverage report.

    Args:
        argv: Command-line arguments; None reads ``sys.argv``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description="Build the parquet feature store.")
    parser.add_argument("--dry-run", action="store_true", help="compute and report without writing")
    args = parser.parse_args(argv)

    if not config.DB_PATH.exists():
        raise SystemExit(f"no database at {config.DB_PATH}; run `make ingest` first")

    conn = connect(config.DB_PATH)
    try:
        frame = build_frame(conn)
    finally:
        conn.close()

    if args.dry_run:
        print(f"computed {len(frame)} rows (not written)")
    else:
        print(f"wrote {write_frame(frame)}: {len(frame)} rows")
    print(coverage_report(frame))
    print("\nRun `make audit` before trusting any of it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
