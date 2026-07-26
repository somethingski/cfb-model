"""Rest-day arithmetic, hand-checked, including the cases with no previous game.

Rest days are the one feature here whose value is a date subtraction, and date subtraction
is where silent off-by-one lives. Two decisions are pinned by these tests because both were
choices rather than facts:

* **UTC calendar days, not elapsed 24-hour periods.** A Saturday game followed by the next
  Saturday reads as 7 whatever time either kicked off. Elapsed hours would give 6 for a
  late-to-early pair and 7 for an early-to-late one, which is noise dressed as signal.
* **A 30-day cap, applied to season openers and to a team's first ever game alike.**
  Confirmed by Sean; logged in ``DECISIONS.md``. An offseason is ~240 days and would tower
  over every in-season value, and "no previous game" and "a long time ago" mean the same
  thing physically: fully rested.
"""

from __future__ import annotations

import pytest

from cfb.features.build import REST_DAYS_CAP, calendar_days_between, rest_days


def kickoff(month: int, day: int, hour: int = 18, year: int = 2014) -> str:
    """An ISO-8601 UTC kickoff."""
    return f"{year}-{month:02d}-{day:02d}T{hour:02d}:00:00+00:00"


# --- ordinary in-season arithmetic --------------------------------------------


@pytest.mark.parametrize(
    ("earlier", "later", "expected"),
    [
        (kickoff(9, 6), kickoff(9, 13), 7),
        (kickoff(9, 6), kickoff(9, 20), 14),
        (kickoff(11, 22), kickoff(11, 28), 6),
        (kickoff(9, 6), kickoff(9, 9), 3),
    ],
    ids=["a week", "a bye week", "to thanksgiving", "a tuesday game"],
)
def test_hand_checked_gaps(earlier: str, later: str, expected: int) -> None:
    """Counted off a calendar."""
    assert calendar_days_between(earlier, later) == expected


def test_kickoff_time_does_not_move_the_answer() -> None:
    """Saturday to Saturday is 7 days whether the second game is at noon or at midnight.

    This is the whole reason the function works on dates rather than on a timedelta.
    """
    saturday_night = kickoff(9, 6, hour=23)
    assert calendar_days_between(saturday_night, kickoff(9, 13, hour=16)) == 7
    assert calendar_days_between(saturday_night, kickoff(9, 13, hour=23)) == 7


def test_days_run_forwards_only() -> None:
    """A negative gap means the caller passed games out of order, which must not be silent."""
    with pytest.raises(ValueError, match="precedes"):
        calendar_days_between(kickoff(9, 13), kickoff(9, 6))


def test_a_month_boundary_is_not_special() -> None:
    """31 August to 6 September is 6 days, not 5 and not 7."""
    assert calendar_days_between(kickoff(8, 31), kickoff(9, 6)) == 6


# --- the cap ------------------------------------------------------------------


def test_an_ordinary_week_is_under_the_cap() -> None:
    """The cap must not be quietly rewriting normal rows."""
    assert rest_days(kickoff(9, 13), kickoff(9, 6)) == 7


def test_a_season_opener_is_capped() -> None:
    """A 2015 opener following a 2014 bowl is ~240 days; it is stored as 30."""
    opener = rest_days(kickoff(9, 5, year=2015), kickoff(1, 1, year=2015))
    assert opener == REST_DAYS_CAP
    assert calendar_days_between(kickoff(1, 1, year=2015), kickoff(9, 5, year=2015)) == 247


def test_a_team_with_no_previous_game_is_capped_too() -> None:
    """The first games of 2014, and any team's first game after entering FBS.

    Same value as an offseason by design: both mean fully rested, and giving one of them a
    null would put a hole in an otherwise dense column for no gain.
    """
    assert rest_days(kickoff(8, 30), None) == REST_DAYS_CAP


def test_the_cap_is_exactly_thirty_days() -> None:
    """Pinned so a change to the constant has to be a decision, not a drift."""
    assert REST_DAYS_CAP == 30
    assert rest_days(kickoff(10, 1), kickoff(9, 1)) == 30
    assert rest_days(kickoff(9, 30), kickoff(9, 1)) == 29


# --- the bowl-season case that is not an opener -------------------------------


def test_a_bowl_game_keeps_its_real_gap_when_it_is_under_the_cap() -> None:
    """A 27 December bowl after a 30 November regular-season finale is 27 days, not capped.

    Postseason rest is a real difference between teams and the cap must not flatten it.
    """
    assert rest_days(kickoff(12, 27), kickoff(11, 30)) == 27


def test_a_late_bowl_crosses_the_cap() -> None:
    """A 1 January bowl after a 30 November finale is 32 days, so it caps at 30."""
    assert calendar_days_between(kickoff(11, 30), kickoff(1, 1, year=2015)) == 32
    assert rest_days(kickoff(1, 1, year=2015), kickoff(11, 30)) == REST_DAYS_CAP
