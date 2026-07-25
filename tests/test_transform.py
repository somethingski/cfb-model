"""Unit tests for the pure JSON-to-row mappers.

Chronology and parsing bugs are silent, so each helper is exercised in isolation,
including the cases where it must refuse rather than guess.
"""

from __future__ import annotations

import pytest

from cfb.ingest import transform


def game(**overrides):
    """Build a minimal ``/games`` record, overridable per test."""
    base = {
        "id": 401525434,
        "season": 2023,
        "week": 1,
        "seasonType": "regular",
        "startDate": "2023-08-26T18:30:00.000Z",
        "startTimeTBD": False,
        "completed": True,
        "neutralSite": True,
        "conferenceGame": False,
        "homeId": 87,
        "homeTeam": "Notre Dame",
        "homeClassification": "fbs",
        "homeConference": "FBS Independents",
        "homePoints": 42,
        "awayId": 2426,
        "awayTeam": "Navy",
        "awayClassification": "fbs",
        "awayConference": "American Athletic",
        "awayPoints": 3,
    }
    base.update(overrides)
    return base


class TestToUtcIso:
    def test_normalises_a_zulu_timestamp(self) -> None:
        assert transform.to_utc_iso("2023-08-26T18:30:00.000Z") == "2023-08-26T18:30:00+00:00"

    def test_converts_an_offset_to_utc(self) -> None:
        assert transform.to_utc_iso("2023-08-26T14:30:00-04:00") == "2023-08-26T18:30:00+00:00"

    def test_is_idempotent(self) -> None:
        once = transform.to_utc_iso("2023-08-26T18:30:00.000Z")
        assert transform.to_utc_iso(once) == once

    def test_missing_value_is_none_not_a_guess(self) -> None:
        assert transform.to_utc_iso(None) is None
        assert transform.to_utc_iso("   ") is None

    def test_naive_timestamp_is_refused(self) -> None:
        """A timestamp with no zone would silently shift the leakage clock."""
        with pytest.raises(ValueError, match="no timezone"):
            transform.to_utc_iso("2023-08-26T18:30:00")

    def test_unparseable_timestamp_raises(self) -> None:
        with pytest.raises(ValueError):
            transform.to_utc_iso("week one, sometime")


class TestIsFbsInvolved:
    def test_keeps_fbs_versus_fbs(self) -> None:
        assert transform.is_fbs_involved(game())

    def test_keeps_fbs_hosting_fcs(self) -> None:
        assert transform.is_fbs_involved(game(awayClassification="fcs"))

    def test_keeps_fbs_visiting_an_fcs_host(self) -> None:
        """Army at Yale (2014 week 5) is the case the API's own filter drops."""
        assert transform.is_fbs_involved(game(homeClassification="fcs", awayClassification="fbs"))

    def test_drops_games_with_no_fbs_team(self) -> None:
        assert not transform.is_fbs_involved(
            game(homeClassification="iii", awayClassification="iii")
        )

    def test_drops_games_with_missing_classifications(self) -> None:
        assert not transform.is_fbs_involved(game(homeClassification=None, awayClassification=None))


class TestParseStatValue:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("17", 17.0), ("0", 0.0), ("-3", -3.0), ("4.5", 4.5), (" 12 ", 12.0)],
    )
    def test_parses_plain_numbers(self, raw: str, expected: float) -> None:
        assert transform.parse_stat_value(raw) == expected

    @pytest.mark.parametrize("raw", ["29:44", "5-35", "18-30", "0-0", "1-15"])
    def test_composite_stats_become_none_rather_than_a_derived_number(self, raw: str) -> None:
        """Splitting "5-35" into penalties and yards is a Phase 4 decision, not an ingest one."""
        assert transform.parse_stat_value(raw) is None

    def test_missing_values_are_none(self) -> None:
        assert transform.parse_stat_value(None) is None
        assert transform.parse_stat_value("") is None


class TestGameRow:
    def test_maps_the_documented_columns(self) -> None:
        row = transform.game_row(game())
        assert row["game_id"] == 401525434
        assert row["start_date"] == "2023-08-26T18:30:00+00:00"
        assert row["neutral_site"] == 1
        assert row["conference_game"] == 0
        assert row["completed"] == 1
        assert row["home_team_id"] == 87
        assert row["away_points"] == 3

    def test_missing_kickoff_is_refused(self) -> None:
        with pytest.raises(ValueError, match="leakage clock"):
            transform.game_row(game(startDate=None))

    def test_unknown_conference_flag_stays_null_rather_than_becoming_false(self) -> None:
        assert transform.game_row(game(conferenceGame=None))["conference_game"] is None

    def test_scheduled_game_keeps_null_scores(self) -> None:
        row = transform.game_row(game(completed=False, homePoints=None, awayPoints=None))
        assert row["completed"] == 0
        assert row["home_points"] is None


class TestTeamsFromGame:
    def test_extracts_both_sides_with_per_season_classification(self) -> None:
        rows = transform.teams_from_game(game(awayClassification="fcs"))
        assert [row["team_id"] for row in rows] == [87, 2426]
        assert rows[0]["season"] == 2023
        assert rows[1]["classification"] == "fcs"

    def test_skips_a_side_with_no_identity(self) -> None:
        assert len(transform.teams_from_game(game(awayId=None, awayTeam=None))) == 1


class TestLineRows:
    def test_stores_the_home_perspective_spread_verbatim(self) -> None:
        record = {
            "id": 1,
            "lines": [
                {
                    "provider": "ESPN Bet",
                    "spread": -16.5,
                    "spreadOpen": -14.0,
                    "overUnder": 54.5,
                    "overUnderOpen": None,
                    "homeMoneyline": -900,
                    "awayMoneyline": 600,
                }
            ],
        }
        (row,) = transform.line_rows(record)
        assert row["spread"] == -16.5  # negative means the home team is favoured
        assert row["over_under_open"] is None
        assert row["home_moneyline"] == -900

    def test_game_without_lines_yields_nothing(self) -> None:
        assert transform.line_rows({"id": 1, "lines": []}) == []
        assert transform.line_rows({"id": 1, "lines": None}) == []


class TestStatRows:
    def test_keeps_the_raw_string_alongside_the_parsed_value(self) -> None:
        record = {
            "id": 1,
            "teams": [
                {
                    "teamId": 2320,
                    "homeAway": "home",
                    "stats": [
                        {"category": "rushingYards", "stat": "180"},
                        {"category": "possessionTime", "stat": "29:44"},
                    ],
                }
            ],
        }
        rows = transform.stat_rows(record)
        assert rows[0] == {
            "game_id": 1,
            "team_id": 2320,
            "is_home": 1,
            "stat_name": "rushingYards",
            "stat_value": 180.0,
            "stat_raw": "180",
        }
        assert rows[1]["stat_value"] is None
        assert rows[1]["stat_raw"] == "29:44"

    def test_marks_the_away_side(self) -> None:
        record = {
            "id": 1,
            "teams": [{"teamId": 5, "homeAway": "away", "stats": [{"category": "x", "stat": "1"}]}],
        }
        assert transform.stat_rows(record)[0]["is_home"] == 0
