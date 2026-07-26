"""Grid search for K, home-field advantage, and the pre-season regression coefficient.

This module is the phase's *second* leakage boundary, and the subtler one. The
chronological walk in :mod:`cfb.elo.pipeline` stops a game's own result from reaching its
rating; nothing there stops a *parameter* fitted on 2023 from reaching a 2023 prediction.
So the search is confined to seasons through :data:`cfb.config.TRAIN_LAST_SEASON`, the
values are written to ``elo_params.json``, and every later phase reads them from that file
rather than re-fitting.

Two guards make that a mechanism rather than a promise:

* :func:`grid_search` raises on any season past the training boundary, with a poisoned-input
  test proving it fires.
* Games are loaded with ``through_season`` set, so the walk itself never even sees a
  validation or test season. Elo is path-dependent — a later game cannot change an earlier
  rating — so truncating the walk is exact, not an approximation.

The objective is log loss on **FBS-vs-FBS games only**. Games against a fixed-1200 FCS
opponent are near-foregone conclusions; including them would let the grid buy log loss on
games this project never models, and ``RISKS.md`` #3 keeps them out of training anyway.
Log loss rather than Brier because it punishes confident misses harder, which is the
failure mode a too-large K produces.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

from cfb import config
from cfb.elo.engine import EloParams
from cfb.elo.pipeline import load_classifications, load_games, run_elo, score, scoreable
from cfb.ingest.schema import connect

K_GRID: tuple[float, ...] = tuple(20.0 + 2.5 * step for step in range(13))
"""K from 20 to 50 in steps of 2.5, the plan's range."""

HFA_GRID: tuple[float, ...] = tuple(40.0 + 5.0 * step for step in range(11))
"""Home-field advantage from 40 to 90 in steps of 5, the plan's range."""

REGRESSION_GRID: tuple[float, ...] = (0.0, 0.10, 0.15, 0.20, 0.25, 1.0 / 3.0, 0.5)
"""Pre-season regression coefficients.

The plan proposed ``{1/4, 1/3, 1/2}``. On that set the winner was ``1/4`` — the lowest
value, i.e. a grid edge, which exit criterion 3 says to treat as a broken grid rather than
a result. Widened downward as the criterion prescribes, the objective turns out to have a
clean interior minimum at 0.20, and ``0.0`` is kept as the null: it scores clearly worse
(0.55798 against 0.55199), which is what earns pre-season regression its place in the
system instead of assuming it.
"""


@dataclass(frozen=True)
class GridResult:
    """One point of the grid and what it scored."""

    params: EloParams
    log_loss: float
    brier: float
    n: int

    @property
    def at_grid_edge(self) -> list[str]:
        """Names of parameters sitting on an endpoint of their grid.

        Exit criterion 3 asks for this explicitly: a winner at the edge means the true
        optimum is probably outside the range, so the grid is wrong — or the objective is.
        """
        edges = []
        if self.params.k in (K_GRID[0], K_GRID[-1]):
            edges.append("k")
        if self.params.hfa in (HFA_GRID[0], HFA_GRID[-1]):
            edges.append("hfa")
        if self.params.regression in (REGRESSION_GRID[0], REGRESSION_GRID[-1]):
            edges.append("regression")
        return edges


def grid_search(
    conn: sqlite3.Connection,
    seasons: Sequence[int],
    k_grid: Sequence[float] = K_GRID,
    hfa_grid: Sequence[float] = HFA_GRID,
    regression_grid: Sequence[float] = REGRESSION_GRID,
    base: EloParams | None = None,
) -> list[GridResult]:
    """Score every combination of K, HFA, and regression on the training seasons.

    Args:
        conn: Open connection to the built database.
        seasons: Seasons to fit on. Must all be at or before
            :data:`cfb.config.TRAIN_LAST_SEASON`.
        k_grid: Candidate K values.
        hfa_grid: Candidate home-field advantages.
        regression_grid: Candidate pre-season regression coefficients.
        base: Parameters supplying the values not being tuned (the priors). Defaults to
            :class:`~cfb.elo.engine.EloParams` defaults.

    Returns:
        Every grid point, sorted by log loss, best first.

    Raises:
        ValueError: If any requested season is past the training boundary. This is the
            leakage gate for tuning: parameters are fitted quantities, and fitting them on
            a season the model is later scored on leaks that season's outcomes into the
            feature.
    """
    illegal = sorted({season for season in seasons if season > config.TRAIN_LAST_SEASON})
    if illegal:
        raise ValueError(
            f"Elo parameters may only be tuned on seasons through {config.TRAIN_LAST_SEASON}; "
            f"refusing to tune on {illegal}. Tuning on a season the model is evaluated on "
            "leaks through the feature."
        )

    base = base or EloParams()
    last_season = max(seasons)
    games = load_games(conn, through_season=last_season)
    classification = load_classifications(conn)

    results = []
    for k, hfa, regression in product(k_grid, hfa_grid, regression_grid):
        params = EloParams(
            k=k,
            hfa=hfa,
            regression=regression,
            mean=base.mean,
            initial=base.initial,
            newcomer=base.newcomer,
            fcs=base.fcs,
        )
        run = run_elo(games, classification, params)
        metrics = score(scoreable(run.rows, seasons=seasons), params)
        results.append(
            GridResult(
                params=params,
                log_loss=metrics["log_loss"],
                brier=metrics["brier"],
                n=int(metrics["n"]),
            )
        )

    # Ties broken by the smaller K, then the smaller HFA: with a flat objective, prefer the
    # less reactive system rather than whichever point the iteration happened to reach first.
    return sorted(results, key=lambda r: (r.log_loss, r.params.k, r.params.hfa))


def sensitivity(results: Sequence[GridResult], attribute: str) -> list[tuple[float, float]]:
    """Best achievable log loss at each value of one parameter.

    A grid search that reports only its winner hides whether the objective actually has a
    shape. If log loss is flat to four decimals across the whole K range, the "tuned" K is
    noise, and that is worth knowing before it is written down as a finding.

    Args:
        results: Grid results.
        attribute: ``"k"``, ``"hfa"``, or ``"regression"``.

    Returns:
        ``(value, best_log_loss)`` pairs, ordered by value.
    """
    best: dict[float, float] = {}
    for result in results:
        value = getattr(result.params, attribute)
        best[value] = min(best.get(value, float("inf")), result.log_loss)
    return sorted(best.items())


def write_params(result: GridResult, seasons: Sequence[int], path=config.ELO_PARAMS_PATH) -> None:
    """Freeze the winning parameters to disk with their provenance.

    Args:
        result: The winning grid point.
        seasons: The seasons it was fitted on, recorded so the file states its own
            leakage boundary.
        path: Where to write. Defaults to :data:`cfb.config.ELO_PARAMS_PATH`.
    """
    payload = {
        "params": result.params.to_dict(),
        "fitted_on": f"{min(seasons)}-{max(seasons)}",
        "objective": "log loss, FBS vs FBS completed games",
        "objective_value": result.log_loss,
        "brier": result.brier,
        "games_scored": result.n,
        "grid": {
            "k": list(K_GRID),
            "hfa": list(HFA_GRID),
            "regression": list(REGRESSION_GRID),
        },
        "note": (
            "Fitted on training seasons only and frozen. Later phases read these values; "
            "they do not re-fit. k, hfa and regression are tuned; mean, initial, newcomer "
            "and fcs are priors chosen before any fitting."
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def report(results: Sequence[GridResult], top: int = 5) -> str:
    """Render the grid search for human review.

    Args:
        results: Grid results, best first.
        top: How many leaders to list.

    Returns:
        A printable report.
    """
    winner = results[0]
    out = ["", f"Grid search: {len(results)} combinations, {winner.n} games scored", "-" * 78]
    out.append(f"{'rank':>4}  {'K':>6}  {'HFA':>5}  {'regress':>8}  {'log loss':>9}  {'brier':>7}")
    for rank, result in enumerate(results[:top], start=1):
        out.append(
            f"{rank:>4}  {result.params.k:>6.1f}  {result.params.hfa:>5.0f}  "
            f"{result.params.regression:>8.4f}  {result.log_loss:>9.5f}  {result.brier:>7.5f}"
        )
    out.append(
        f"{'worst':>4}  {results[-1].params.k:>6.1f}  {results[-1].params.hfa:>5.0f}  "
        f"{results[-1].params.regression:>8.4f}  {results[-1].log_loss:>9.5f}  "
        f"{results[-1].brier:>7.5f}"
    )

    out += ["", "Sensitivity: best log loss at each value", "-" * 78]
    for attribute in ("k", "hfa", "regression"):
        pairs = sensitivity(results, attribute)
        rendered = "  ".join(f"{value:g}:{loss:.5f}" for value, loss in pairs)
        out.append(f"  {attribute:<11} {rendered}")

    edges = winner.at_grid_edge
    out += ["", "Verdict", "-" * 78]
    if edges:
        out.append(
            f"  WINNER SITS ON A GRID EDGE ({', '.join(edges)}). Exit criterion 3 treats this "
            "as evidence the grid or the objective is wrong, not as a result. Widen and rerun."
        )
    else:
        out.append("  Winner is interior to every grid.")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the grid search and write ``elo_params.json``.

    Args:
        argv: Command-line arguments; None reads ``sys.argv``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description="Tune Elo parameters on training seasons only.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the search without writing elo_params.json",
    )
    args = parser.parse_args(argv)

    if not config.DB_PATH.exists():
        raise SystemExit(f"no database at {config.DB_PATH}; run `make ingest` first")

    seasons = [season for season in config.SEASONS if season <= config.TRAIN_LAST_SEASON]
    conn = connect(config.DB_PATH)
    try:
        print(f"tuning on {seasons[0]}-{seasons[-1]} only ({len(seasons)} seasons)")
        results = grid_search(conn, seasons)
        print(report(results))
        if args.dry_run:
            print(f"\ndry run: {config.ELO_PARAMS_PATH} not written")
        else:
            write_params(results[0], seasons)
            print(f"\nwrote {config.ELO_PARAMS_PATH}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
