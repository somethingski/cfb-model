"""Generates ``FEATURES.md`` from the feature specs and the built store.

The documentation is generated rather than written because a hand-maintained table drifts
from the code, and this particular table is the artefact the project is judged on: exit
criterion 3 asks for every feature's definition *and the latest timestamp of information it
depends on*. A stale row there is worse than no row.

Two halves, and the split matters. The **declared** half comes from
:data:`cfb.features.build.ALL_SPECS` — what each feature is and what it is allowed to read.
The **measured** half is read off the built parquet store — how many nulls each column
actually has, and how far before kickoff the latest input actually sat. Declaring a
timestamp proves nothing; the audit is what proves it, and the measured half is where you
can see it.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from cfb import config
from cfb.features import build
from cfb.features.build import FEATURE_SPECS, KEY_SPECS, LABEL_SPEC, FeatureSpec

EXCLUSIONS: tuple[tuple[str, str], ...] = (
    (
        "Vegas lines (spread, total, moneyline)",
        "The de-vigged closing line is the benchmark this project is measured against. "
        "Feeding it to the model would make 'approaching the line' circular. Not a "
        "judgement call and not revisitable — `cfb.features.audit.assert_no_market_source` "
        "fails the build if the feature module so much as queries the `lines` table.",
    ),
    (
        "Anything from the game's own box score",
        "Yards, plays, turnovers and time of possession for the game being predicted are "
        "known only after it is over. `game_team_stats` is read for prior games only.",
    ),
    (
        "Recruiting rankings and returning production",
        "Coverage is inconsistent across the season range and the fields change meaning "
        "between years. Revisit only if Phase 6 motivates it.",
    ),
    (
        "Weather",
        "CFBD does not reliably carry a pre-kickoff forecast, only conditions recorded at "
        "or after the game. A post-kickoff observation is exactly what this phase excludes.",
    ),
    (
        "Season-level aggregates (final records, end-of-year rankings, season totals)",
        "Leakage by construction when applied to a mid-season game. Every rolling column "
        "here is as-of-kickoff; `prev_season_win_pct` reaches back only to a season that "
        "had finished before this game was scheduled.",
    ),
    (
        "Post-game Elo",
        "Never stored at all, since Phase 1. A column that does not exist cannot be joined "
        "into a feature by accident.",
    ),
)


def spec_table(specs: Sequence[FeatureSpec], frame: pd.DataFrame | None) -> list[str]:
    """Render one markdown table of feature specs.

    Args:
        specs: The specs to render.
        frame: The built store, for measured null counts. None omits that column.

    Returns:
        Markdown lines.
    """
    header = "| Feature | Definition | Latest information it depends on | Null policy |"
    divider = "|---|---|---|---|"
    if frame is not None:
        header += " Nulls |"
        divider += "---|"
    lines = [header, divider]
    for spec in specs:
        cells = [f"`{spec.name}`", spec.definition, spec.depends_on, spec.null_policy]
        if frame is not None:
            cells.append(f"{int(frame[spec.name].isna().sum()):,}")
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
    return lines


def measured_section(frame: pd.DataFrame) -> list[str]:
    """Render what the built store actually contains.

    Args:
        frame: The built store.

    Returns:
        Markdown lines.
    """
    dated = frame[frame["as_of"].notna()]
    margin = pd.to_datetime(dated["start_date"]) - pd.to_datetime(dated["as_of"])
    lines = [
        "## Measured on the built store",
        "",
        f"- **{len(frame):,} rows**, one per completed game in the spine, "
        f"{config.FIRST_SEASON}–{config.LAST_SEASON}.",
        f"- **{int(frame['fcs_opponent'].sum()):,} rows** involve an FCS opponent, flagged by "
        "`fcs_opponent`. Phase 5 excludes them from training (`RISKS.md` #3).",
        f"- **{int((frame['prior_games_home'] == 0).sum()):,} rows** are a home team's first "
        "game of its season, so every rolling column is null by design.",
        f"- The latest input any row reads sits **{margin.min()}** before its own kickoff at "
        f"the tightest, {margin.max()} at the loosest. The audit asserts this is strictly "
        "positive for every row.",
        f"- **{int(frame['as_of'].isna().sum())} rows** read nothing at all: the opening games "
        f"of {config.FIRST_SEASON}, where neither team had ever played.",
        "",
        "### Rows and nulls by season",
        "",
        "| Season | Rows | FCS opponent | First games | Null `off_ppg_roll_home` |",
        "|---|---|---|---|---|",
    ]
    for season, block in frame.groupby("season", sort=True):
        lines.append(
            f"| {season} | {len(block):,} | {int(block['fcs_opponent'].sum())} | "
            f"{int((block['prior_games_home'] == 0).sum())} | "
            f"{int(block['off_ppg_roll_home'].isna().sum())} |"
        )
    return lines


def render(frame: pd.DataFrame | None) -> str:
    """Render the whole of ``FEATURES.md``.

    Args:
        frame: The built store, or None to render the declared half alone.

    Returns:
        The markdown document.
    """
    lines = [
        "# FEATURES",
        "",
        "**Generated by `python -m cfb.features.docs`. Do not edit by hand** — edit "
        "`FEATURE_SPECS` in `src/cfb/features/build.py` and regenerate, so that the table "
        "and the columns cannot disagree.",
        "",
        "Every feature below uses only information available **strictly before that game's "
        "kickoff**. That is the load-bearing claim of this project, and it is not taken on "
        "trust: `make audit` rebuilds a sample of these rows from a database truncated at "
        "each game's kickoff and demands the identical answer. If the audit fails, the fix "
        "is in the features, never in the audit.",
        "",
        "## How the cutoff is enforced",
        "",
        "One function, `cfb.features.build.priors_before`, selects a team's games with "
        "`start_date < kickoff`. Every rolling and rest feature is computed from that list "
        "and from nothing else; the feature functions are pure and never touch the "
        "database. `team_features` asserts the same rule a second time at the point of use, "
        "so a future caller that builds a window some other way raises instead of leaking.",
        "",
        "Elo columns come from `elo_pregame`, which Phase 3 writes by walking the schedule "
        "in kickoff order and snapshotting ratings *before* applying each result. The audit "
        "does not read that table back — it re-runs the walk over the truncated view and "
        "compares.",
        "",
        "## Features",
        "",
    ]
    lines += spec_table(FEATURE_SPECS, frame)
    lines += [
        "",
        "## Keys",
        "",
        "Not features. They identify the row and are dropped before training.",
        "",
    ]
    lines += spec_table(KEY_SPECS, None)
    lines += [
        "",
        "## Label",
        "",
        "",
    ]
    lines += spec_table([LABEL_SPEC], None)
    lines += [
        "",
        "The label lives in the same file for convenience. It is read only by Phase 5's "
        "training code and never by feature computation — `feature_row(..., "
        "with_label=False)` is what the audit calls, because a database truncated at "
        "kickoff has no result to give.",
        "",
        "## Deliberately excluded",
        "",
    ]
    for name, reason in EXCLUSIONS:
        lines += [f"**{name}.** {reason}", ""]

    if frame is not None:
        lines += measured_section(frame)
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Write ``FEATURES.md``.

    Args:
        argv: Command-line arguments; None reads ``sys.argv``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description="Generate FEATURES.md.")
    parser.add_argument(
        "--out", type=Path, default=config.PROJECT_ROOT / "FEATURES.md", help="destination"
    )
    args = parser.parse_args(argv)

    try:
        frame = build.read_frame()
    except FileNotFoundError:
        print("no feature store yet; documenting the declared half only")
        frame = None

    args.out.write_text(render(frame))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
