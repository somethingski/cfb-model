"""Pure functions mapping CFBD JSON onto database rows.

Every function here is total, side-effect free, and unit-tested in isolation. The
chronology helpers in particular (``to_utc_iso``) are the kind of code whose bugs are
silent, so they are kept small and tested rather than inlined into the backfill loop.

Nothing in this module invents a value. A field the API did not supply becomes ``None``
and is handled downstream by exclusion, never by imputation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

FBS = "fbs"


def to_utc_iso(raw: str | None) -> str | None:
    """Normalise a CFBD timestamp to a UTC ISO-8601 string.

    CFBD returns kickoff times like ``2023-08-26T18:30:00.000Z``. Storing a single
    canonical spelling means later phases can compare and order ``start_date`` as plain
    text without re-parsing, which is what makes the leakage clock cheap to enforce.

    Args:
        raw: A timestamp string from the API, or None.

    Returns:
        The instant as ``YYYY-MM-DDTHH:MM:SS+00:00``, or None if ``raw`` is None/empty.

    Raises:
        ValueError: If ``raw`` is present but not a parseable timestamp.
    """
    if raw is None or not str(raw).strip():
        return None
    text = str(raw).strip()
    # ``fromisoformat`` accepts "Z" only from 3.11; be explicit so the intent is visible.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError(f"kickoff timestamp has no timezone, refusing to guess: {raw!r}")
    return parsed.astimezone(UTC).replace(microsecond=0).isoformat()


def is_fbs_involved(game: dict[str, Any]) -> bool:
    """Return True if either side of the game is an FBS team.

    The CFBD ``classification`` filter keeps only games *hosted* by an FBS team, which
    silently drops the rare FBS-team-at-FCS-host game (e.g. Army at Yale, 2014 week 5).
    Filtering client-side keeps the game spine complete, and Phase 3 needs FBS-vs-FCS
    games to exist in the database in order to apply an explicit FCS policy to them.

    Args:
        game: A raw ``/games`` record.

    Returns:
        True if the home or away team is classified FBS.
    """
    return FBS in (game.get("homeClassification"), game.get("awayClassification"))


def parse_stat_value(raw: Any) -> float | None:
    """Parse a box-score stat into a number, or None when it is not a scalar.

    CFBD returns every stat as a string, and five of them are composites rather than
    numbers: ``possessionTime`` (``"29:44"``), ``completionAttempts`` (``"18-30"``),
    ``totalPenaltiesYards`` (``"5-35"``), ``thirdDownEff`` and ``fourthDownEff``
    (``"5-15"``). Splitting those into components here would be a modelling decision
    made at ingest time, so instead the raw string is stored verbatim alongside and this
    returns None. Phase 4 decides what a composite means; ingestion only records it.

    Args:
        raw: The ``stat`` value from the API.

    Returns:
        The value as a float, or None if it is absent or not a plain number.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _as_int_bool(value: Any) -> int:
    """Coerce an API boolean to SQLite's 0/1, treating a missing value as false."""
    return 1 if value else 0


def game_row(game: dict[str, Any]) -> dict[str, Any]:
    """Map a ``/games`` record onto a ``games`` table row.

    Args:
        game: A raw ``/games`` record.

    Returns:
        A dict keyed by column name.

    Raises:
        ValueError: If the game has no parseable kickoff time. The leakage clock cannot
            have holes, so this fails at ingest rather than becoming a null downstream.
    """
    start_date = to_utc_iso(game.get("startDate"))
    if start_date is None:
        raise ValueError(f"game {game.get('id')} has no startDate; leakage clock would have a hole")
    return {
        "game_id": game["id"],
        "season": game["season"],
        "week": game["week"],
        "season_type": game["seasonType"],
        "start_date": start_date,
        "start_time_tbd": _as_int_bool(game.get("startTimeTBD")),
        "neutral_site": _as_int_bool(game.get("neutralSite")),
        "conference_game": None
        if game.get("conferenceGame") is None
        else _as_int_bool(game["conferenceGame"]),
        "home_team_id": game["homeId"],
        "away_team_id": game["awayId"],
        "home_points": game.get("homePoints"),
        "away_points": game.get("awayPoints"),
        "completed": _as_int_bool(game.get("completed")),
    }


def teams_from_game(game: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract both teams' identity and per-season affiliation from a game record.

    ``/teams/fbs`` lists only FBS teams, but the game spine references FCS opponents as
    foreign keys. Their identity and per-season classification are copied from the game
    payload itself — the API's own fields, not a guess.

    Args:
        game: A raw ``/games`` record.

    Returns:
        One dict per side with ``team_id``, ``school``, ``season``, ``conference``,
        ``classification``.
    """
    rows = []
    for side in ("home", "away"):
        team_id = game.get(f"{side}Id")
        school = game.get(f"{side}Team")
        if team_id is None or not school:
            continue
        rows.append(
            {
                "team_id": team_id,
                "school": school,
                "season": game["season"],
                "conference": game.get(f"{side}Conference"),
                "classification": game.get(f"{side}Classification"),
            }
        )
    return rows


def line_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Map a ``/lines`` record onto ``lines`` table rows, one per provider.

    Sign convention, verified against a known game and recorded in ``DECISIONS.md``:
    ``spread`` is stated from the **home team's** perspective, so a negative spread means
    the home team is favoured. Stored exactly as the API reports it; no re-signing here.

    Args:
        record: A raw ``/lines`` record with a nested ``lines`` list.

    Returns:
        A list of row dicts, empty when the game has no posted lines.
    """
    rows = []
    for line in record.get("lines") or []:
        provider = line.get("provider")
        if not provider:
            continue
        rows.append(
            {
                "game_id": record["id"],
                "provider": provider,
                "spread": line.get("spread"),
                "spread_open": line.get("spreadOpen"),
                "over_under": line.get("overUnder"),
                "over_under_open": line.get("overUnderOpen"),
                "home_moneyline": line.get("homeMoneyline"),
                "away_moneyline": line.get("awayMoneyline"),
            }
        )
    return rows


def stat_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Map a ``/games/teams`` record onto tidy ``game_team_stats`` rows.

    Long format is deliberate: stat categories vary across eras, and a row-per-stat
    absorbs that without schema churn.

    Args:
        record: A raw ``/games/teams`` record with a nested ``teams`` list.

    Returns:
        A list of row dicts, one per (team, stat).
    """
    rows = []
    for team in record.get("teams") or []:
        team_id = team.get("teamId")
        if team_id is None:
            continue
        is_home = 1 if team.get("homeAway") == "home" else 0
        for stat in team.get("stats") or []:
            name = stat.get("category")
            if not name:
                continue
            raw = stat.get("stat")
            if raw is None:
                continue
            rows.append(
                {
                    "game_id": record["id"],
                    "team_id": team_id,
                    "is_home": is_home,
                    "stat_name": name,
                    "stat_value": parse_stat_value(raw),
                    "stat_raw": str(raw),
                }
            )
    return rows
