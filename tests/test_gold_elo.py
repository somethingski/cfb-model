"""Gold regression for the Elo engine — a synthetic league worked out by hand.

The same reasoning as the Phase 2 fixture: there is no external source to look an Elo
rating up in, so the fixture is only evidence if a human did the arithmetic independently
and the engine is compared against *that*. ``gold/elo_fixture.json`` ships with every
``hand_computed`` slot null and this file fails until they are filled in.

The worksheet asks for the ratings each team carries **out** of each game, because that is
what a hand computation actually produces — you cannot get the pre-game ratings of game 5
without having worked out games 1 through 4 first. The column the pipeline stores is the
pre-game one, so :func:`test_stored_pregame_ratings_are_the_hand_computed_chain` chains the
hand-computed ratings forward itself, applying the season regression with the formula
written out here rather than by calling the code under test.

Unlike Phase 2, the fixture stores no pipeline values at all. The league is six synthetic
games, so the engine's side is recomputed live on every run — nothing to copy from, and no
stale numbers that could pass after the engine changed underneath them.
"""

from __future__ import annotations

import json

import pytest

from cfb import config
from cfb.elo.engine import EloParams, update
from cfb.elo.pipeline import Game, run_elo

FIXTURE_PATH = config.GOLD_DIR / "elo_fixture.json"
TOLERANCE = 1e-6
"""Six decimal places, per the plan. The hand-computed values are recorded to eight, so a
chained comparison still has two digits of headroom over calculator rounding."""


def load_fixture() -> dict:
    """Read the worksheet, or an empty stand-in when it is missing."""
    if not FIXTURE_PATH.exists():
        return {"games": [], "teams": []}
    return json.loads(FIXTURE_PATH.read_text())


FIXTURE = load_fixture()
GAMES = FIXTURE.get("games", [])

pytestmark = pytest.mark.skipif(not FIXTURE_PATH.exists(), reason=f"no fixture at {FIXTURE_PATH}")


def params() -> EloParams:
    """The parameters the fixture was worked out under."""
    return EloParams.from_dict(FIXTURE["params"])


def team_ids() -> dict[str, int]:
    """Team name to id."""
    return {team["name"]: team["team_id"] for team in FIXTURE["teams"]}


def fcs_names() -> set[str]:
    """Names of the teams that are FCS in some season of the fixture."""
    return {team["name"] for team in FIXTURE["teams"] if "fcs" in team["classification"].values()}


@pytest.fixture(scope="module")
def league() -> dict:
    """Run the production walk over the synthetic league.

    Returns:
        ``{"rows": {game_id: PregameRow}, "post": {game_id: (home, away)},
        "ratings": {name: rating}}``, where ``post`` is what the engine says each team
        carried out of the game.
    """
    ids = team_ids()
    classification = {
        (team["team_id"], int(season)): value
        for team in FIXTURE["teams"]
        for season, value in team["classification"].items()
    }
    games = [
        Game(
            game_id=game["game_id"],
            season=game["season"],
            start_date=game["start_date"],
            neutral_site=bool(game["neutral_site"]),
            home_team_id=ids[game["home"]],
            away_team_id=ids[game["away"]],
            home_points=game["home_points"],
            away_points=game["away_points"],
            completed=True,
        )
        for game in FIXTURE["games"]
    ]
    settings = params()
    run = run_elo(games, classification, settings)
    rows = {row.game_id: row for row in run.rows}

    post = {}
    for game in FIXTURE["games"]:
        row = rows[game["game_id"]]
        home, away = update(
            row.home_elo_pre,
            row.away_elo_pre,
            game["home_points"],
            game["away_points"],
            settings,
            neutral_site=bool(game["neutral_site"]),
        )
        # The FCS side of the game keeps its fixed rating; only the FBS side updates.
        post[game["game_id"]] = (
            home if row.home_is_fbs else settings.fcs,
            away if row.away_is_fbs else settings.fcs,
        )

    by_id = {team_id: name for name, team_id in ids.items()}
    return {
        "rows": rows,
        "post": post,
        "ratings": {by_id[team_id]: rating for team_id, rating in run.ratings.items()},
    }


# --- the gate -----------------------------------------------------------------


def test_human_has_done_the_arithmetic() -> None:
    """Red until a person has worked this league by hand."""
    missing = [
        f"game {game['game_id']} ({game['case']}): {field}"
        for game in GAMES
        for field, value in game["hand_computed"].items()
        if value is None
    ]
    missing += [
        f"{block}: {team}"
        for block in ("season_end_2014", "final_ratings")
        for team, value in FIXTURE[block]["hand_computed"].items()
        if value is None
    ]
    if missing:
        pytest.fail(
            "hand_computed values are still blank:\n  "
            + "\n  ".join(missing)
            + f"\n\nWork them out from the method block in {FIXTURE_PATH} with a calculator "
            "— not by running this project's code — then fill them in."
        )
    if not FIXTURE.get("human_verified"):
        pytest.fail(
            f"{FIXTURE_PATH} has human_verified=false. Set human_verified/verified_by/"
            "verified_on once every row checks out."
        )


def test_fixture_exercises_every_rule() -> None:
    """A fixture of six ordinary home games would not pin much."""
    assert len(GAMES) == 6, "the plan asks for six games"
    assert any(game["neutral_site"] for game in GAMES), "no neutral-site game"
    assert len({game["season"] for game in GAMES}) == 2, "no season boundary, so no regression"
    fcs = fcs_names()
    assert any(game["home"] in fcs or game["away"] in fcs for game in GAMES), "no FCS opponent"
    upsets = [
        game for game in GAMES if game["away"] in fcs and game["away_points"] > game["home_points"]
    ]
    assert upsets, "no FBS loss to an FCS opponent, which is the case the policy exists for"


# --- the engine against the hand computation ----------------------------------


@pytest.mark.parametrize("game", GAMES, ids=[str(game["game_id"]) for game in GAMES])
def test_post_game_ratings_match_the_hand_computation(game: dict, league: dict) -> None:
    """Each game's update, against a number a human produced independently."""
    produced = league["post"][game["game_id"]]
    for side, index in (("home", 0), ("away", 1)):
        expected = game["hand_computed"][f"{side}_elo_{'post'}"]
        if expected is None:
            pytest.skip("not hand-computed yet; test_human_has_done_the_arithmetic fails first")
        assert produced[index] == pytest.approx(expected, abs=TOLERANCE), (
            f"game {game['game_id']} {side} ({game[side]}): "
            f"engine {produced[index]!r}, hand {expected!r}"
        )


def test_stored_pregame_ratings_are_the_hand_computed_chain(league: dict) -> None:
    """The column that actually reaches the database, derived from the hand computation.

    A team's pre-game rating is the rating it carried out of its previous game, regressed
    if a season boundary intervened, or the initial rating if it has not played yet. The
    chain is built here from the hand-computed numbers only — the regression is spelled out
    rather than delegated to :func:`~cfb.elo.engine.run_season_regression`, so this stays a
    check on the code rather than a restatement of it.
    """
    settings = params()
    if any(value is None for game in GAMES for value in game["hand_computed"].values()):
        pytest.skip("not hand-computed yet; test_human_has_done_the_arithmetic fails first")

    fcs = fcs_names()
    carried: dict[str, float] = {}
    season = GAMES[0]["season"]

    for game in sorted(GAMES, key=lambda g: (g["start_date"], g["game_id"])):
        if game["season"] != season:
            carried = {
                team: rating + (settings.mean - rating) * settings.regression
                for team, rating in carried.items()
            }
            season = game["season"]

        row = league["rows"][game["game_id"]]
        for side, produced in (("home", row.home_elo_pre), ("away", row.away_elo_pre)):
            team = game[side]
            expected = settings.fcs if team in fcs else carried.get(team, settings.initial)
            assert produced == pytest.approx(expected, abs=TOLERANCE), (
                f"game {game['game_id']} {side} ({team}) went in rated {produced!r}; "
                f"the hand-computed chain says {expected!r}"
            )

        for side in ("home", "away"):
            if game[side] not in fcs:
                carried[game[side]] = game["hand_computed"][f"{side}_elo_post"]


def test_end_of_season_ratings_agree_with_the_per_game_numbers(league: dict) -> None:
    """Internal consistency of the worksheet, before it is used to check anything.

    The end-of-2014 block must be what the 2014 games leave behind. If the two disagree,
    one of them is a slip and the chain test above would be checking the wrong thing.
    """
    expected = FIXTURE["season_end_2014"]["hand_computed"]
    if any(value is None for value in expected.values()):
        pytest.skip("not hand-computed yet; test_human_has_done_the_arithmetic fails first")

    last_2014: dict[str, float] = {}
    for game in sorted(
        (g for g in GAMES if g["season"] == 2014), key=lambda g: (g["start_date"], g["game_id"])
    ):
        for side in ("home", "away"):
            if game[side] not in fcs_names():
                last_2014[game[side]] = game["hand_computed"][f"{side}_elo_post"]
    assert last_2014 == pytest.approx(expected, abs=TOLERANCE)


def test_final_ratings_match_the_hand_computation(league: dict) -> None:
    """The whole run, end to end."""
    expected = FIXTURE["final_ratings"]["hand_computed"]
    if any(value is None for value in expected.values()):
        pytest.skip("not hand-computed yet; test_human_has_done_the_arithmetic fails first")
    for team, value in expected.items():
        assert league["ratings"][team] == pytest.approx(value, abs=TOLERANCE)


def test_fcs_opponent_is_the_fixed_rating_and_is_never_tracked(league: dict) -> None:
    """The FCS policy, on data small enough to check by eye."""
    settings = params()
    fcs = fcs_names()
    for game in GAMES:
        row = league["rows"][game["game_id"]]
        if game["home"] in fcs:
            assert row.home_elo_pre == settings.fcs
            assert league["post"][game["game_id"]][0] == settings.fcs
        if game["away"] in fcs:
            assert row.away_elo_pre == settings.fcs
            assert league["post"][game["game_id"]][1] == settings.fcs
    assert not (fcs & league["ratings"].keys()), "an FCS team acquired a tracked rating"
