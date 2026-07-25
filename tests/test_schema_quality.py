"""Data-quality gate tests.

Two halves, both necessary:

* **Poisoned-input tests** prove each assertion can actually fail. A quality check that
  has never been observed failing is not a check.
* **Integration tests** run the same assertions against the real database, and skip when
  the backfill has not been run.
"""

from __future__ import annotations

import sqlite3

import pytest

from cfb.ingest import quality
from tests.conftest import add_game


class TestChecksFireOnPoisonedInput:
    def test_blank_start_date_is_caught(self, toy_db: sqlite3.Connection) -> None:
        add_game(toy_db, game_id=1, start_date="   ")
        with pytest.raises(quality.DataQualityError, match="leakage clock"):
            quality.check_no_null_start_dates(toy_db)

    def test_completed_game_without_a_score_is_caught(self, toy_db: sqlite3.Connection) -> None:
        add_game(toy_db, game_id=1, completed=1, home_points=None)
        with pytest.raises(quality.DataQualityError, match="null score"):
            quality.check_completed_games_have_scores(toy_db)

    def test_short_season_is_caught(self, toy_db: sqlite3.Connection) -> None:
        summaries = [
            quality.SeasonSummary(2023, games=12, completed=12, with_lines=0, with_team_stats=0)
        ]
        with pytest.raises(quality.DataQualityError, match="outside the documented bound"):
            quality.check_season_game_counts(toy_db, summaries)

    def test_oversized_season_is_caught(self, toy_db: sqlite3.Connection) -> None:
        """The failure mode of a broken FBS filter is too many games, not too few."""
        summaries = [
            quality.SeasonSummary(2023, games=3734, completed=3734, with_lines=0, with_team_stats=0)
        ]
        with pytest.raises(quality.DataQualityError, match="outside the documented bound"):
            quality.check_season_game_counts(toy_db, summaries)

    def test_empty_database_is_caught(self, toy_db: sqlite3.Connection) -> None:
        with pytest.raises(quality.DataQualityError, match="no games"):
            quality.run_all_checks(toy_db)


class TestChecksPassOnCleanInput:
    def test_scheduled_game_without_scores_is_allowed(self, toy_db: sqlite3.Connection) -> None:
        add_game(toy_db, game_id=1, completed=0, home_points=None, away_points=None)
        quality.check_completed_games_have_scores(toy_db)

    def test_covid_season_has_its_own_documented_lower_bound(
        self, toy_db: sqlite3.Connection
    ) -> None:
        """2020 is a documented anomaly (RISKS.md #6), not a relaxed global bound."""
        covid = quality.SeasonSummary(
            2020, games=570, completed=570, with_lines=0, with_team_stats=0
        )
        quality.check_season_game_counts(toy_db, [covid])

        same_count_normal_season = quality.SeasonSummary(
            2021, games=570, completed=570, with_lines=0, with_team_stats=0
        )
        with pytest.raises(quality.DataQualityError):
            quality.check_season_game_counts(toy_db, [same_count_normal_season])


class TestSummary:
    def test_counts_coverage_per_season(self, toy_db: sqlite3.Connection) -> None:
        add_game(toy_db, game_id=1, season=2023)
        add_game(toy_db, game_id=2, season=2023)
        toy_db.execute("INSERT INTO lines (game_id, provider, spread) VALUES (1, 'ESPN Bet', -7.0)")
        toy_db.execute(
            """INSERT INTO game_team_stats (game_id, team_id, is_home, stat_name, stat_raw)
               VALUES (1, 87, 1, 'rushingYards', '180')"""
        )

        (summary,) = quality.summarize(toy_db)
        assert summary.games == 2
        assert summary.with_lines == 1
        assert summary.pct_lines == 50.0
        assert summary.pct_team_stats == 50.0

    def test_format_is_printable_with_no_seasons(self) -> None:
        assert "season" in quality.format_summary([])


@pytest.mark.integration
class TestBuiltDatabase:
    def test_all_quality_checks_pass(self, built_db: sqlite3.Connection) -> None:
        summaries = quality.run_all_checks(built_db)
        assert len(summaries) > 0

    def test_every_season_in_the_configured_range_is_present(
        self, built_db: sqlite3.Connection
    ) -> None:
        from cfb import config

        present = {summary.season for summary in quality.summarize(built_db)}
        assert set(config.SEASONS) <= present

    def test_foreign_keys_resolve(self, built_db: sqlite3.Connection) -> None:
        violations = built_db.execute("PRAGMA foreign_key_check").fetchall()
        assert violations == []

    def test_every_game_has_two_distinct_teams(self, built_db: sqlite3.Connection) -> None:
        (bad,) = built_db.execute(
            "SELECT COUNT(*) FROM games WHERE home_team_id = away_team_id"
        ).fetchone()
        assert bad == 0

    def test_no_line_belongs_to_an_unknown_game(self, built_db: sqlite3.Connection) -> None:
        (orphans,) = built_db.execute(
            "SELECT COUNT(*) FROM lines l WHERE NOT EXISTS "
            "(SELECT 1 FROM games g WHERE g.game_id = l.game_id)"
        ).fetchone()
        assert orphans == 0
