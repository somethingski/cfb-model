"""The leakage audit — Phase 4's hard gate.

The claim under test is the one in ``CLAUDE.md``: *every feature for a game uses only
information available strictly before that game's kickoff*. This module tries to break it.

**Method: recompute under truncation.** For a sampled game *g* with kickoff *t*, SQL views
expose a database in which the only games that exist are those with ``start_date < t``.
Every feature for *g* is recomputed from that view — by calling the production functions in
``build.py``, not a second implementation of them — and compared to the row sitting in the
parquet store. Any disagreement means the stored row was computed from something the
truncated database cannot see, which is leakage, or from something that varies between
runs, which is nondeterminism. Both are hard failures.

Two details of the design are load-bearing and easy to skim past:

* **Elo is re-walked, not re-read.** ``elo_pregame`` is a stored table, so comparing it to
  itself would prove nothing. The truncated view instead contains every prior game *plus g
  itself with its result blanked out*, and Phase 3's own :func:`cfb.elo.pipeline.run_elo`
  is run over it. The walk snapshots ratings before applying a result and skips games that
  have none, so what comes back is exactly "what were these teams rated going in", derived
  from nothing later than *t*.
* **Truncation is strict.** ``start_date < t`` excludes games kicking off at the same
  instant as *g*, while the production Elo walk orders by ``(start_date, game_id)`` and
  would have applied a simultaneous game with a lower id first. The two agree only because
  no team plays twice at the same instant — which is asserted here rather than assumed
  (:func:`assert_no_simultaneous_team_games`). That makes this the stronger check: it tests
  *strictly before kickoff*, not *before kickoff in some tie-break order*.

``CLAUDE.md``: if this fails, **the fix is in the features, never in the audit.** Do not
loosen a tolerance, shrink a sample, or skip a check to get green.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import io
import random
import sqlite3
import tokenize
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from cfb import config
from cfb.elo.engine import EloParams
from cfb.elo.pipeline import load_classifications, load_games, load_params, run_elo
from cfb.features import build
from cfb.features.build import FEATURE_SPECS, FeatureSpec, ScheduledGame
from cfb.ingest.schema import connect

SAMPLE_SIZE: int = 200
"""Games recomputed under truncation, on top of the pinned edge cases."""

AUDIT_SEED: int = 20260726
"""Fixed so a failure is reproducible and a pass cannot be re-rolled until it is green."""

FLOAT_TOLERANCE: float = 1e-9
"""Floats must agree to here; everything else must be exactly equal."""

LABEL_CORRELATION_LIMIT: float = 0.99
"""A feature correlating with the label this strongly within one season is treated as label
leakage rather than as a very good feature."""

CHRONOLOGY_MIN_SHARE: float = 0.25
"""The chronology sub-check must both keep and delete at least this share of the spine.

A cutoff that deletes almost nothing proves nothing, and neither does one that keeps almost
nothing. The cutoff itself is the median kickoff (:func:`chronology_cutoff`) rather than a
hardcoded date, so this holds on any database the audit is pointed at.
"""

MARKET_TERMS: tuple[str, ...] = (
    "lines",
    "vegas_benchmark",
    "spread",
    "moneyline",
    "over_under",
    "p_home_devig",
)
"""Words that must not appear in the feature-building source. The closing line is the
benchmark, never an input (``CLAUDE.md``), and that rule gets a mechanical check rather
than relying on anyone remembering it during a refactor."""

AUDIT_RELATIONS = build.Relations(games="audit_team_games", stats="audit_stats")
"""What the feature loaders read while the audit is running."""

TRUNCATION_SQL = """
DROP VIEW  IF EXISTS temp.audit_prior_games;
DROP VIEW  IF EXISTS temp.audit_games;
DROP VIEW  IF EXISTS temp.audit_team_games;
DROP VIEW  IF EXISTS temp.audit_stats;
DROP TABLE IF EXISTS temp.audit_cutoff;

-- One row, rewritten per sampled game. Keeping the cutoff in a table rather than baking it
-- into the view text means the views are defined once and never redefined mid-run.
CREATE TABLE temp.audit_cutoff (
    game_id      INTEGER NOT NULL,
    start_date   TEXT    NOT NULL,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL
);

-- The truncated database: everything that had kicked off before the game under test.
CREATE VIEW temp.audit_prior_games AS
    SELECT * FROM main.games
    WHERE start_date < (SELECT start_date FROM temp.audit_cutoff);

-- What the Elo walk sees: the prior games, plus the game under test with its result
-- removed, so the walk emits a pre-game snapshot for it without ever seeing a score.
CREATE VIEW temp.audit_games AS
    SELECT * FROM temp.audit_prior_games
    UNION ALL
    SELECT game_id, season, week, season_type, start_date, start_time_tbd, neutral_site,
           conference_game, home_team_id, away_team_id, NULL, NULL, 0
    FROM main.games
    WHERE game_id = (SELECT game_id FROM temp.audit_cutoff);

-- What the rolling features see: prior games involving either of the two teams. Narrowing
-- to the two teams is what keeps 240 truncated rebuilds to fifteen seconds, and it is safe
-- because no v1 feature reads any other team's games. It also fails closed rather than
-- silently: a future feature that did reach beyond the matchup — strength of schedule, an
-- opponent-adjusted rate — would recompute differently here than it did from the full
-- database, and the audit would fail until somebody widened this view on purpose.
CREATE VIEW temp.audit_team_games AS
    SELECT g.* FROM temp.audit_prior_games g, temp.audit_cutoff c
    WHERE g.home_team_id IN (c.home_team_id, c.away_team_id)
       OR g.away_team_id IN (c.home_team_id, c.away_team_id);

CREATE VIEW temp.audit_stats AS
    SELECT s.* FROM main.game_team_stats s
    WHERE s.game_id IN (SELECT game_id FROM temp.audit_team_games);
"""


# --- results ------------------------------------------------------------------


@dataclass
class AuditResult:
    """What one audit run found.

    Attributes:
        checks: ``(name, detail)`` for each check that passed, for the report.
        failures: Human-readable failures. Empty means the gate is open.
        sampled: Game ids that were recomputed under truncation.
    """

    checks: list[tuple[str, str]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    sampled: list[int] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if nothing failed."""
        return not self.failures

    def record(self, name: str, detail: str) -> None:
        """Note a check that passed.

        Args:
            name: Short check name.
            detail: What it established.
        """
        self.checks.append((name, detail))

    def fail(self, message: str) -> None:
        """Note a failure.

        Args:
            message: What went wrong, with enough detail to act on.
        """
        self.failures.append(message)


# --- truncation ---------------------------------------------------------------


def install_truncation_views(conn: sqlite3.Connection) -> None:
    """Create the temp cutoff table and the truncated views.

    Args:
        conn: Open connection. The views live in ``temp`` and vanish when it closes.
    """
    conn.executescript(TRUNCATION_SQL)


def set_cutoff(conn: sqlite3.Connection, game: ScheduledGame) -> None:
    """Point the truncated views at one game.

    Args:
        conn: Connection with the views installed.
        game: The game under test.
    """
    conn.execute("DELETE FROM temp.audit_cutoff")
    conn.execute(
        "INSERT INTO temp.audit_cutoff (game_id, start_date, home_team_id, away_team_id) "
        "VALUES (?, ?, ?, ?)",
        (game.game_id, game.start_date, game.home_team_id, game.away_team_id),
    )


def truncated_elo(
    conn: sqlite3.Connection,
    game: ScheduledGame,
    classification: dict[tuple[int, int], str],
    params: EloParams,
) -> tuple[float, float]:
    """Re-derive a game's pre-game ratings from games that kicked off before it.

    Runs Phase 3's walk over the truncated view. The game under test is present with its
    result blanked, so it sorts last, receives a snapshot, and moves nothing.

    Args:
        conn: Connection with the views installed and the cutoff set.
        game: The game under test.
        classification: ``(team_id, season)`` to subdivision.
        params: The frozen Elo parameters.

    Returns:
        ``(home_elo_pre, away_elo_pre)``.

    Raises:
        RuntimeError: If the walk produced no row for the game, which would mean the view
            definition and the cutoff have drifted apart.
    """
    run = run_elo(load_games(conn, relation="audit_games"), classification, params)
    for row in reversed(run.rows):
        if row.game_id == game.game_id:
            return row.home_elo_pre, row.away_elo_pre
    raise RuntimeError(f"the truncated walk produced no row for game {game.game_id}")


def recompute_row(
    conn: sqlite3.Connection,
    game: ScheduledGame,
    classification: dict[tuple[int, int], str],
    params: EloParams,
) -> dict[str, object]:
    """Recompute every feature for one game from a truncated database.

    Calls the production functions in ``build.py``. If this module reimplemented them the
    audit would only be testing its own arithmetic against itself.

    Args:
        conn: Connection with the views installed.
        game: The game under test.
        classification: ``(team_id, season)`` to subdivision.
        params: The frozen Elo parameters.

    Returns:
        A feature row, without the label — a truncated view has no result for this game,
        and asking it for one is the question this project must never ask.
    """
    set_cutoff(conn, game)
    context = build.load_context(conn, AUDIT_RELATIONS)
    return build.feature_row(
        game, context, truncated_elo(conn, game, classification, params), with_label=False
    )


# --- comparison ---------------------------------------------------------------


def normalise(value: object) -> object:
    """Collapse pandas' several spellings of "missing" to None, and numpy scalars to Python.

    The numpy unwrapping is not cosmetic housekeeping: the diff this produces is what a
    human reads when the gate fails, and ``stored=np.float64(8.0)`` beside ``stored=20.5``
    makes two identical problems look like two different ones.

    Args:
        value: A stored or recomputed value.

    Returns:
        None if the value is missing, otherwise a plain Python value.
    """
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def compare_row(
    stored: pd.Series, recomputed: dict[str, object], specs: Iterable[FeatureSpec]
) -> list[str]:
    """Compare a stored feature row to its truncated recomputation.

    Nulls are compared as strictly as numbers. A null that recomputes to a value is the
    most likely shape a leak takes here — the stored row saw a game the truncated one
    cannot — so ``None`` never silently equals anything.

    Args:
        stored: The row from the parquet store.
        recomputed: The row from :func:`recompute_row`.
        specs: The columns to compare, carrying their comparison kind.

    Returns:
        One string per disagreeing column; empty if the row is clean.
    """
    diffs: list[str] = []
    for spec in specs:
        want = normalise(stored[spec.name])
        got = normalise(recomputed[spec.name])
        if want is None or got is None:
            if want is not got:
                diffs.append(f"{spec.name}: stored={want!r} truncated={got!r}")
            continue
        if spec.kind == "float":
            if abs(float(want) - float(got)) > FLOAT_TOLERANCE:
                diffs.append(
                    f"{spec.name}: stored={float(want):.12g} truncated={float(got):.12g} "
                    f"(diff {abs(float(want) - float(got)):.3g} > {FLOAT_TOLERANCE:g})"
                )
        elif want != got:
            diffs.append(f"{spec.name}: stored={want!r} truncated={got!r}")
    return diffs


# --- sampling -----------------------------------------------------------------


def pinned_games(frame: pd.DataFrame) -> dict[int, str]:
    """Edge cases that must be audited whether or not the random draw finds them.

    A seeded draw of 200 from 10,373 rows will usually miss the rows where the interesting
    bugs live. These are selected by property rather than hardcoded id, so they keep
    tracking the data if it is rebuilt.

    Args:
        frame: The feature store.

    Returns:
        ``game_id`` to the reason it is pinned.
    """
    pinned: dict[int, str] = {}

    def pin(rows: pd.DataFrame, reason: str, limit: int = 2) -> None:
        for game_id in rows.sort_values(["start_date", "game_id"])["game_id"].head(limit):
            pinned.setdefault(int(game_id), reason)

    pin(frame, "the first games in the data — nothing precedes them", limit=3)
    for season, block in frame.groupby("season"):
        pin(block, f"first games of {season} — rolling stats must all be null", limit=2)
    pin(
        frame[frame["prior_games_home"] == 1], "exactly one prior game — the min-window boundary", 3
    )
    pin(
        frame[frame["season_type"] == "spring_regular"],
        "spring 2021 kickoff in a 2020 season (RISKS #13)",
        2,
    )
    pin(
        frame[frame["season_type"] == "postseason"],
        "postseason — a long gap and a neutral field",
        2,
    )
    pin(frame[frame["neutral_site"] == 1], "neutral site", 2)
    pin(frame[frame["fcs_opponent"] == 1], "FCS opponent — a fixed-1200 rating in the row", 3)
    pin(
        frame[frame["off_ppg_roll_home"].notna() & frame["off_ypp_roll_home"].isna()],
        "a prior game with no box score (RISKS #11) — points roll, yards do not",
        2,
    )
    pin(
        frame[frame["rest_days_home"] == frame["rest_days_home"].min()],
        "the shortest rest in the data",
        2,
    )
    pin(frame.tail(3), "the last games in the data — everything precedes them", 3)
    return pinned


def choose_sample(frame: pd.DataFrame, size: int, seed: int) -> dict[int, str]:
    """Pick the games to recompute: the pinned edge cases plus a seeded random draw.

    Args:
        frame: The feature store.
        size: How many random games to add.
        seed: Random seed.

    Returns:
        ``game_id`` to the reason it was chosen, in kickoff order.
    """
    chosen = pinned_games(frame)
    rng = random.Random(seed)
    remaining = [int(g) for g in frame["game_id"] if int(g) not in chosen]
    for game_id in rng.sample(remaining, min(size, len(remaining))):
        chosen[game_id] = "random"
    order = {int(g): i for i, g in enumerate(frame["game_id"])}
    return {game_id: chosen[game_id] for game_id in sorted(chosen, key=lambda g: order[g])}


# --- checks that do not need truncation ---------------------------------------


def assert_no_simultaneous_team_games(conn: sqlite3.Connection, result: AuditResult) -> None:
    """No team may appear in two games with the same kickoff.

    This is the precondition that makes strict ``start_date <`` truncation equivalent to
    the production walk's ``(start_date, game_id)`` ordering. If it ever breaks, the audit
    starts comparing two genuinely different orderings and its failures become noise.

    Args:
        conn: Open connection.
        result: Result to record into.
    """
    clashes = conn.execute(
        """
        SELECT start_date, team_id, COUNT(*) AS n FROM (
            SELECT start_date, home_team_id AS team_id FROM games
            UNION ALL
            SELECT start_date, away_team_id FROM games
        ) GROUP BY start_date, team_id HAVING n > 1
        """
    ).fetchall()
    if clashes:
        result.fail(
            f"{len(clashes)} teams appear in two games at the same kickoff, e.g. "
            f"{[(row['team_id'], row['start_date']) for row in clashes[:3]]}. Strict "
            "truncation and the Elo walk's tie-break no longer agree."
        )
    else:
        result.record("simultaneity", "no team plays two games at the same instant")


def assert_as_of_precedes_kickoff(frame: pd.DataFrame, result: AuditResult) -> None:
    """Exit criterion 3, measured rather than asserted in prose.

    ``as_of`` is the kickoff of the latest game any rolling or rest feature in the row
    actually read. Every one of them must be strictly earlier than the row's own kickoff.

    Args:
        frame: The feature store.
        result: Result to record into.
    """
    dated = frame[frame["as_of"].notna()]
    offenders = dated[dated["as_of"] >= dated["start_date"]]
    if len(offenders):
        result.fail(
            f"{len(offenders)} rows depend on a game that kicked off at or after their own "
            f"kickoff, e.g. {offenders['game_id'].head(3).tolist()}"
        )
    else:
        result.record(
            "as_of < kickoff",
            f"all {len(dated)} rows with a dependency read nothing later than "
            f"{(pd.to_datetime(dated['start_date']) - pd.to_datetime(dated['as_of'])).min()} "
            "before kickoff",
        )


def assert_no_perfect_label_correlation(frame: pd.DataFrame, result: AuditResult) -> None:
    """Canary for label leakage: no feature may track the label within a season.

    A feature that knows the result would show up as a correlation no pre-game quantity can
    have. This will not catch subtle leakage, which is what the truncation check is for —
    it catches the crude kind loudly and cheaply.

    Args:
        frame: The feature store.
        result: Result to record into.
    """
    numeric = [spec.name for spec in FEATURE_SPECS if spec.kind == "float"] + [
        spec.name for spec in FEATURE_SPECS if spec.kind == "exact" and spec.name != "season_type"
    ]
    worst = (0.0, "", 0)
    for season, block in frame.groupby("season"):
        for column in numeric:
            if column == "as_of":
                continue
            values = block[column].astype(float)
            if values.notna().sum() < 2 or values.nunique(dropna=True) < 2:
                continue
            correlation = abs(values.corr(block["label_home_win"].astype(float)))
            if pd.isna(correlation):
                continue
            if correlation > worst[0]:
                worst = (correlation, column, int(season))
            if correlation >= LABEL_CORRELATION_LIMIT:
                result.fail(
                    f"{column} correlates {correlation:.4f} with the label in {season} "
                    f"(limit {LABEL_CORRELATION_LIMIT}) — that is label leakage, not a feature"
                )
    if worst[1]:
        result.record(
            "label correlation",
            f"strongest within-season correlation is {worst[0]:.4f} "
            f"({worst[1]}, {worst[2]}); limit is {LABEL_CORRELATION_LIMIT}",
        )


def prose_lines(source: str) -> set[int]:
    """Line numbers occupied by docstrings and attribute docstrings.

    A bare string expression is never something the module *does*, so it is prose whether
    it sits under a ``def`` or under a module-level constant.

    Args:
        source: Python source.

    Returns:
        The line numbers to ignore.
    """
    lines: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return lines


def executable_text(source: str) -> str:
    """Everything in a module except its comments and its prose.

    SQL string literals are deliberately kept — a ``FROM lines`` hiding inside a query is
    exactly what this is looking for.

    Args:
        source: Python source.

    Returns:
        Lower-cased executable text.
    """
    ignore = prose_lines(source)
    parts = [
        token.string
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type != tokenize.COMMENT
        and not (token.type == tokenize.STRING and token.start[0] in ignore)
    ]
    return " ".join(parts).lower()


def assert_no_market_source(result: AuditResult) -> None:
    """The feature builder must not read the betting market at all.

    A line-derived feature column is the stop-everything error of this project. Reading the
    module's own source is crude, and that is the point: it keeps working when somebody
    adds a helper six months from now without having read ``CLAUDE.md``. Docstrings are
    exempt, so the module can say "never read the lines table" without tripping its own
    check; SQL strings are not.

    Args:
        result: Result to record into.
    """
    executable = executable_text(inspect.getsource(build))
    found = sorted({term for term in MARKET_TERMS if term in executable})
    if found:
        result.fail(
            f"cfb.features.build mentions {found} in executable code — the closing line is "
            "the benchmark, never a feature"
        )
    else:
        result.record("no market inputs", f"none of {list(MARKET_TERMS)} appear in the builder")


def chronology_cutoff(conn: sqlite3.Connection) -> str:
    """The median kickoff in the spine.

    Derived rather than hardcoded so the sub-check splits any database into two non-trivial
    halves — including the toy ones the poisoned-input tests build.

    Args:
        conn: Open connection.

    Returns:
        A kickoff timestamp with roughly half the schedule on each side.
    """
    total = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    row = conn.execute(
        "SELECT start_date FROM games ORDER BY start_date, game_id LIMIT 1 OFFSET ?",
        (total // 2,),
    ).fetchone()
    return row["start_date"]


def assert_elo_chronology(
    conn: sqlite3.Connection,
    classification: dict[tuple[int, int], str],
    params: EloParams,
    result: AuditResult,
) -> None:
    """Phase 3's chronology test, re-run against the stored table Elo features read.

    The per-game recompute already covers Elo for the sample. This covers half the schedule
    at one cutoff, and it checks the *stored* ``elo_pregame`` rather than an in-memory
    rebuild — which is what the feature store actually consumed.

    Args:
        conn: Open connection with the truncation views installed.
        classification: ``(team_id, season)`` to subdivision.
        params: The frozen Elo parameters.
        result: Result to record into.
    """
    cutoff = chronology_cutoff(conn)
    conn.execute("DELETE FROM temp.audit_cutoff")
    conn.execute(
        "INSERT INTO temp.audit_cutoff (game_id, start_date, home_team_id, away_team_id) "
        "VALUES (?, ?, ?, ?)",
        (-1, cutoff, -1, -1),
    )
    total = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    truncated = run_elo(load_games(conn, relation="audit_prior_games"), classification, params)
    stored = {
        row["game_id"]: (row["home_elo_pre"], row["away_elo_pre"])
        for row in conn.execute("SELECT game_id, home_elo_pre, away_elo_pre FROM elo_pregame")
    }
    differing = [
        row.game_id
        for row in truncated.rows
        if stored.get(row.game_id) != (row.home_elo_pre, row.away_elo_pre)
    ]
    kept, deleted = len(truncated.rows), total - len(truncated.rows)
    if min(kept, deleted) < CHRONOLOGY_MIN_SHARE * total:
        result.fail(
            f"the chronology cutoff kept {kept} and deleted {deleted} of {total} games; a "
            f"split that lopsided proves nothing (minimum share {CHRONOLOGY_MIN_SHARE})"
        )
    if differing:
        result.fail(
            f"{len(differing)} stored Elo ratings change once later games are deleted, e.g. "
            f"{differing[:5]} — a future game is reaching an earlier rating"
        )
    else:
        result.record(
            "elo chronology",
            f"{kept} ratings before {cutoff[:10]} are unchanged by deleting the {deleted} "
            "games after it",
        )


def assert_store_matches_spine(
    conn: sqlite3.Connection, frame: pd.DataFrame, result: AuditResult
) -> None:
    """The store must hold exactly the completed games, once each.

    A silently short feature store is the failure mode that makes every later metric look
    fine while being computed on the wrong population.

    Args:
        conn: Open connection.
        frame: The feature store.
        result: Result to record into.
    """
    expected = conn.execute(
        "SELECT COUNT(*) FROM games WHERE completed = 1 AND home_points IS NOT NULL "
        "AND away_points IS NOT NULL"
    ).fetchone()[0]
    if len(frame) != expected:
        result.fail(
            f"feature store has {len(frame)} rows; the spine has {expected} completed games"
        )
    elif frame["game_id"].duplicated().any():
        result.fail("the feature store contains duplicate game ids")
    else:
        result.record("row coverage", f"{len(frame)} rows, one per completed game, no duplicates")


# --- the run ------------------------------------------------------------------


def run_audit(
    conn: sqlite3.Connection,
    frame: pd.DataFrame,
    sample_size: int = SAMPLE_SIZE,
    seed: int = AUDIT_SEED,
) -> AuditResult:
    """Run every check against a feature store.

    Args:
        conn: Open connection to the database the store was built from.
        frame: The feature store to audit.
        sample_size: Random games to recompute, on top of the pinned edge cases.
        seed: Random seed.

    Returns:
        The result. Callers decide what to do with a failure; ``main`` exits non-zero.
    """
    result = AuditResult()
    params, _ = load_params()
    classification = load_classifications(conn)
    install_truncation_views(conn)

    assert_store_matches_spine(conn, frame, result)
    assert_no_simultaneous_team_games(conn, result)
    assert_as_of_precedes_kickoff(frame, result)
    assert_no_perfect_label_correlation(frame, result)
    assert_no_market_source(result)
    assert_elo_chronology(conn, classification, params, result)

    schedule = {game.game_id: game for game in build.load_schedule(conn)}
    indexed = frame.set_index("game_id")
    sample = choose_sample(frame, sample_size, seed)
    result.sampled = list(sample)

    mismatched = 0
    for game_id, reason in sample.items():
        recomputed = recompute_row(conn, schedule[game_id], classification, params)
        diffs = compare_row(indexed.loc[game_id], recomputed, FEATURE_SPECS)
        if diffs:
            mismatched += 1
            game = schedule[game_id]
            detail = "\n".join(f"      {diff}" for diff in diffs)
            result.fail(
                f"game {game_id} ({game.season} week {game.week}, {reason}) does not survive "
                f"truncation to {game.start_date}:\n{detail}"
            )
    if not mismatched:
        pinned = sum(1 for reason in sample.values() if reason != "random")
        result.record(
            "recompute under truncation",
            f"{len(sample)} games ({pinned} pinned edge cases, {len(sample) - pinned} random, "
            f"seed {seed}) recompute identically from a database truncated at their kickoff",
        )
    return result


def render(result: AuditResult) -> str:
    """Render the audit report.

    Args:
        result: A completed run.

    Returns:
        A printable report.
    """
    out = ["", "LEAKAGE AUDIT", "=" * 78]
    for name, detail in result.checks:
        out.append(f"  PASS  {name:<28} {detail}")
    if result.passed:
        out += [
            "",
            "  Audit passed. This is a gate, not a guarantee: it proves the stored features can be",
            "  rebuilt from a database that ends at kickoff. Read FEATURES.md and challenge "
            "anything",
            "  whose pre-kickoff availability you cannot explain aloud.",
        ]
        return "\n".join(out)

    out += ["", f"  {len(result.failures)} FAILURE(S)", "-" * 78]
    for failure in result.failures:
        out.append(f"  FAIL  {failure}")
    out += [
        "",
        "  The fix is in the features, never in the audit. Do not loosen a tolerance, shrink",
        "  the sample, or skip a check to get green (CLAUDE.md).",
    ]
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    """Audit the built feature store.

    Args:
        argv: Command-line arguments; None reads ``sys.argv``.

    Returns:
        0 if the audit passed, 1 if it did not.
    """
    parser = argparse.ArgumentParser(description="Run the Phase 4 leakage audit.")
    parser.add_argument("--sample", type=int, default=SAMPLE_SIZE, help="random games to recompute")
    parser.add_argument("--seed", type=int, default=AUDIT_SEED, help="random seed")
    parser.add_argument("--store", type=Path, default=None, help="feature store to audit")
    args = parser.parse_args(argv)

    if not config.DB_PATH.exists():
        raise SystemExit(f"no database at {config.DB_PATH}; run `make ingest` first")

    frame = build.read_frame(args.store)
    conn = connect(config.DB_PATH)
    try:
        result = run_audit(conn, frame, sample_size=args.sample, seed=args.seed)
    finally:
        conn.close()

    print(render(result))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
