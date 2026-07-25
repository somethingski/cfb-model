"""Tests for provider selection, sigma fitting, and the benchmark build.

The sigma guard is the leakage boundary for Phase 2, so it gets a poisoned-input case:
a test that only ever passes clean seasons proves the function runs, not that the check
fires.
"""

from __future__ import annotations

import sqlite3

import pytest

from cfb import config
from cfb.vegas import benchmark
from cfb.vegas.benchmark import (
    PROVIDER_LADDER,
    benchmark_row,
    build_benchmark,
    estimate_sigma,
    has_moneyline_pair,
    has_spread,
    pick_line,
)
from tests.conftest import add_game

SIGMA = 16.0


def line(provider: str, spread=None, home_ml=None, away_ml=None) -> dict:
    """A ``lines`` row as :func:`pick_line` sees it."""
    return {
        "provider": provider,
        "spread": spread,
        "home_moneyline": home_ml,
        "away_moneyline": away_ml,
    }


def add_line(conn: sqlite3.Connection, game_id: int, provider: str, spread=None, **kwargs) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO lines (game_id, provider, spread, home_moneyline, away_moneyline) "
        "VALUES (?, ?, ?, ?, ?)",
        (game_id, provider, spread, kwargs.get("home_ml"), kwargs.get("away_ml")),
    )


class TestPickLine:
    def test_prefers_the_earlier_rung(self) -> None:
        rows = [line("Bovada", spread=-3.0), line("consensus", spread=-3.5)]
        assert pick_line(rows, has_spread)["provider"] == "consensus"

    def test_falls_through_to_a_later_rung(self) -> None:
        rows = [line("teamrankings", spread=-7.0), line("ESPN Bet", spread=-6.5)]
        assert pick_line(rows, has_spread)["provider"] == "ESPN Bet"

    def test_skips_rows_the_predicate_rejects(self) -> None:
        """consensus outranks Bovada but never quotes moneylines, so it is skipped here."""
        rows = [
            line("consensus", spread=-3.0),
            line("Bovada", spread=-3.5, home_ml=-160, away_ml=140),
        ]
        assert pick_line(rows, has_spread)["provider"] == "consensus"
        assert pick_line(rows, has_moneyline_pair)["provider"] == "Bovada"

    def test_one_sided_moneyline_is_not_usable(self) -> None:
        """There is no second price to de-vig against; 58 such rows exist in the database."""
        assert pick_line([line("Bovada", home_ml=-160)], has_moneyline_pair) is None
        assert pick_line([line("Bovada", away_ml=140)], has_moneyline_pair) is None

    def test_returns_none_when_nothing_is_usable(self) -> None:
        assert pick_line([line("consensus")], has_spread) is None
        assert pick_line([], has_spread) is None

    def test_unknown_provider_ranks_last_and_deterministically(self) -> None:
        rows = [line("zzz unknown", spread=-1.0), line("aaa unknown", spread=-2.0)]
        assert pick_line(rows, has_spread)["provider"] == "aaa unknown"
        assert pick_line(rows[::-1], has_spread)["provider"] == "aaa unknown"

    def test_unknown_provider_never_outranks_a_ladder_provider(self) -> None:
        rows = [line("aaa unknown", spread=-1.0), line("numberfire", spread=-2.0)]
        assert pick_line(rows, has_spread)["provider"] == "numberfire"

    def test_ladder_has_no_duplicates(self) -> None:
        assert len(set(PROVIDER_LADDER)) == len(PROVIDER_LADDER)


class TestBenchmarkRow:
    def test_spread_is_the_primary_source_even_when_a_moneyline_exists(self) -> None:
        """The decision that keeps the yardstick's construction constant across 2021."""
        rows = [
            line("consensus", spread=-7.0),
            line("Bovada", spread=-7.5, home_ml=-280, away_ml=230),
        ]
        row = benchmark_row(1, rows, SIGMA)
        assert row["source_type"] == "spread"
        assert row["provider"] == "consensus"
        assert row["spread"] == -7.0
        assert row["p_home_devig"] == pytest.approx(0.669126, abs=1e-6)  # Phi(7/16)
        # The moneyline view is carried alongside, from its own provider, for Phase 6.
        assert row["ml_provider"] == "Bovada"
        implied_home, implied_away = 280 / 380, 100 / 330
        assert row["p_home_moneyline"] == pytest.approx(
            implied_home / (implied_home + implied_away), abs=1e-9
        )

    def test_moneyline_fallback_when_no_spread_exists(self) -> None:
        """No such game is in the database today; the branch is tested, not assumed."""
        row = benchmark_row(1, [line("Bovada", home_ml=-200, away_ml=170)], SIGMA)
        assert row["source_type"] == "moneyline"
        assert row["spread"] is None
        assert row["p_home_devig"] == pytest.approx(9 / 14, abs=1e-12)

    def test_returns_none_when_no_price_is_usable(self) -> None:
        """Game 400945031 (2017) has a consensus over/under and nothing else."""
        assert benchmark_row(400945031, [line("consensus")], SIGMA) is None
        assert benchmark_row(1, [], SIGMA) is None

    def test_no_moneyline_leaves_the_sensitivity_columns_null(self) -> None:
        row = benchmark_row(1, [line("consensus", spread=-3.0)], SIGMA)
        assert row["ml_provider"] is None
        assert row["home_moneyline"] is None
        assert row["p_home_moneyline"] is None


class TestEstimateSigma:
    def test_rejects_a_validation_season(self, toy_db: sqlite3.Connection) -> None:
        """Poisoned input: the guard must fire, not merely exist."""
        with pytest.raises(ValueError, match="refusing to fit"):
            estimate_sigma(toy_db, [2014, 2022])

    @pytest.mark.parametrize("season", [2022, 2023, 2024, 2025])
    def test_rejects_every_season_past_the_boundary(
        self, toy_db: sqlite3.Connection, season: int
    ) -> None:
        with pytest.raises(ValueError, match="refusing to fit"):
            estimate_sigma(toy_db, [season])

    def test_boundary_season_itself_is_allowed(self) -> None:
        assert config.TRAIN_LAST_SEASON == 2021

    def test_fits_from_margin_residuals(self, toy_db: sqlite3.Connection) -> None:
        """Two games whose residuals are +10 and -10 give an RMS of exactly 10."""
        for index, (spread, home_points) in enumerate([(-7.0, 24), (-7.0, 4)], start=1):
            add_game(toy_db, game_id=index, season=2015, home_points=home_points, away_points=7)
            add_line(toy_db, index, "consensus", spread=spread)
        assert estimate_sigma(toy_db, [2015]) == pytest.approx(10.0)

    def test_raises_rather_than_returning_a_default_when_empty(
        self, toy_db: sqlite3.Connection
    ) -> None:
        with pytest.raises(ValueError, match="no completed games"):
            estimate_sigma(toy_db, [2015])

    def test_ignores_incomplete_games(self, toy_db: sqlite3.Connection) -> None:
        """A cancelled game has no margin to fit on (RISKS #12)."""
        add_game(toy_db, game_id=1, season=2015, home_points=24, away_points=7)
        add_line(toy_db, 1, "consensus", spread=-7.0)
        add_game(toy_db, game_id=2, season=2015, home_points=None, away_points=None, completed=0)
        add_line(toy_db, 2, "consensus", spread=-40.0)
        assert estimate_sigma(toy_db, [2015]) == pytest.approx(10.0)


class TestBuildBenchmark:
    def test_excludes_rather_than_imputes(self, toy_db: sqlite3.Connection) -> None:
        add_game(toy_db, game_id=1, season=2015)
        add_line(toy_db, 1, "consensus", spread=-7.0)
        add_game(toy_db, game_id=2, season=2015)  # no line row at all
        add_game(toy_db, game_id=3, season=2015)
        add_line(toy_db, 3, "consensus")  # over/under only

        counts = build_benchmark(toy_db, SIGMA)
        assert counts == {
            "games": 3,
            "included": 1,
            "no_line_rows": 1,
            "unusable_line_rows": 1,
        }
        assert [row[0] for row in toy_db.execute("SELECT game_id FROM vegas_benchmark")] == [1]

    def test_rebuild_is_idempotent(self, toy_db: sqlite3.Connection) -> None:
        add_game(toy_db, game_id=1, season=2015)
        add_line(toy_db, 1, "consensus", spread=-7.0)
        build_benchmark(toy_db, SIGMA)
        build_benchmark(toy_db, SIGMA)
        assert toy_db.execute("SELECT COUNT(*) FROM vegas_benchmark").fetchone()[0] == 1


@pytest.mark.integration
class TestBuiltBenchmark:
    """Assertions against the real table. These are the exit-criteria checks."""

    @pytest.fixture(autouse=True)
    def _require_table(self, built_db: sqlite3.Connection) -> None:
        exists = built_db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vegas_benchmark'"
        ).fetchone()
        if not exists:
            pytest.skip("vegas_benchmark not built; run `make benchmark`")

    def test_sigma_is_stable_and_plausible(self, built_db: sqlite3.Connection) -> None:
        """Refit from the database; a change here means the fit or the ladder moved."""
        seasons = [s for s in config.SEASONS if s <= config.TRAIN_LAST_SEASON]
        assert estimate_sigma(built_db, seasons) == pytest.approx(15.98, abs=0.05)

    def test_every_probability_is_a_probability(self, built_db: sqlite3.Connection) -> None:
        assert (
            built_db.execute(
                "SELECT COUNT(*) FROM vegas_benchmark "
                "WHERE p_home_devig <= 0 OR p_home_devig >= 1 OR p_home_devig IS NULL"
            ).fetchone()[0]
            == 0
        )

    def test_mean_matches_the_actual_home_win_rate(self, built_db: sqlite3.Connection) -> None:
        """Exit criterion 3, on the both-FBS population Phase 5 actually trains on.

        The all-games mean sits near 0.61 because 1,192 FBS-vs-FCS games average 0.92;
        that is the schedule, not a calibration failure, and the all-games actual home
        win rate is 0.62. Asserted here so a future ladder change cannot drift the
        benchmark away from reality unnoticed.
        """
        mean_p, actual = built_db.execute(
            """
            SELECT AVG(b.p_home_devig),
                   AVG(CASE WHEN g.home_points > g.away_points THEN 1.0 ELSE 0 END)
            FROM vegas_benchmark b
            JOIN games g ON g.game_id = b.game_id
            JOIN team_seasons th ON th.team_id = g.home_team_id AND th.season = g.season
            JOIN team_seasons ta ON ta.team_id = g.away_team_id AND ta.season = g.season
            WHERE th.classification = 'fbs' AND ta.classification = 'fbs'
              AND g.completed = 1 AND g.home_points IS NOT NULL
            """
        ).fetchone()
        assert 0.55 <= mean_p <= 0.60
        assert mean_p == pytest.approx(actual, abs=0.02)

    def test_sign_convention_holds_on_real_games(self, built_db: sqlite3.Connection) -> None:
        """Every home favourite of 21+ must be above 0.9; every 21+ dog below 0.1."""
        wrong = built_db.execute(
            "SELECT COUNT(*) FROM vegas_benchmark "
            "WHERE (spread <= -21 AND p_home_devig <= 0.9) "
            "   OR (spread >= 21 AND p_home_devig >= 0.1)"
        ).fetchone()[0]
        assert wrong == 0

    def test_favourites_actually_win_more_often(self, built_db: sqlite3.Connection) -> None:
        """The end-to-end sign check: an inverted spread would invert this ordering."""
        favoured, underdog = built_db.execute(
            """
            SELECT AVG(CASE WHEN b.p_home_devig >= 0.5 AND g.home_points > g.away_points
                            THEN 1.0 WHEN b.p_home_devig >= 0.5 THEN 0 END),
                   AVG(CASE WHEN b.p_home_devig < 0.5 AND g.home_points > g.away_points
                            THEN 1.0 WHEN b.p_home_devig < 0.5 THEN 0 END)
            FROM vegas_benchmark b JOIN games g ON g.game_id = b.game_id
            WHERE g.completed = 1 AND g.home_points IS NOT NULL
            """
        ).fetchone()
        assert favoured > 0.7
        assert underdog < 0.3

    def test_moneyline_column_agrees_with_the_spread_column(
        self, built_db: sqlite3.Connection
    ) -> None:
        """The Phase 6 sensitivity check, asserted as a sanity bound now.

        The two are built from different providers by different arithmetic, so they will
        not match exactly. A large mean gap would mean one of them is wrong.
        """
        gap, n = built_db.execute(
            "SELECT AVG(ABS(p_home_devig - p_home_moneyline)), COUNT(p_home_moneyline) "
            "FROM vegas_benchmark WHERE p_home_moneyline IS NOT NULL"
        ).fetchone()
        assert n > 3000
        assert gap < 0.05

    def test_neutral_site_moneyline_mismatch_is_pinned(self, built_db: sqlite3.Connection) -> None:
        """RISKS #16, pinned so it cannot grow or migrate unnoticed.

        CFBD's Bovada home/away moneyline assignment does not always match the side its
        spread is stated from, and it is a neutral-site problem: 8.8% of neutral-site
        rows against 0.1% of ordinary home games. Only ``p_home_moneyline`` is affected —
        the assertion on ordinary home games is what would catch this spreading into the
        primary column.
        """
        total, neutral = built_db.execute(
            f"""
            SELECT COUNT(*), SUM(g.neutral_site)
            FROM vegas_benchmark b JOIN games g ON g.game_id = b.game_id
            WHERE {benchmark.MONEYLINE_MISMATCH_SQL}
            """
        ).fetchone()
        assert total == 31, "the mismatch population moved; re-read RISKS #16 before adjusting"
        assert neutral == 28

        (non_neutral_rate,) = built_db.execute(
            f"""
            SELECT AVG(CASE WHEN {benchmark.MONEYLINE_MISMATCH_SQL} THEN 1.0 ELSE 0 END)
            FROM vegas_benchmark b JOIN games g ON g.game_id = b.game_id
            WHERE g.neutral_site = 0 AND b.p_home_moneyline IS NOT NULL
            """
        ).fetchone()
        assert non_neutral_rate < 0.01, "mismatch is no longer confined to neutral-site games"

    def test_no_line_derived_column_leaked_into_a_feature_table(
        self, built_db: sqlite3.Connection
    ) -> None:
        """Phase 2 output must stay in its own table until Phase 6 reads it deliberately."""
        tables = {
            row[0] for row in built_db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "vegas_benchmark" in tables
        for table in tables - {"vegas_benchmark", "lines"}:
            columns = {
                row[1]
                for row in built_db.execute(f"PRAGMA table_info({table})")  # noqa: S608
            }
            leaked = columns & {"p_home_devig", "spread", "home_moneyline", "away_moneyline"}
            assert not leaked, f"{table} carries line-derived columns {leaked}"


def test_report_runs_against_the_built_table(built_db: sqlite3.Connection) -> None:
    exists = built_db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vegas_benchmark'"
    ).fetchone()
    if not exists:
        pytest.skip("vegas_benchmark not built; run `make benchmark`")
    report = benchmark.coverage_report(built_db)
    assert "Coverage by season" in report
    assert "Excluded" in report
