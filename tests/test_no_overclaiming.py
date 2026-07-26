"""Scan the published prose for market-beating language (``RISKS.md`` #7).

The claim this project makes is fixed: **calibration approaching the de-vigged closing
line**. The failure mode the risk register names is drift — a README sentence that reads
better than the result, written months after the result, by someone who remembers the
number as rounder than it was. This is the mechanical control for that, and it is the Phase
6 deliverable of exit criterion 3.

**How the check works.** Over-claiming is not a word list, because the sentence the project
most needs to be able to write — "this model does not beat the market" — contains the
banned word. So each occurrence of a flagged term must appear in a **sentence** that also
carries a negation or alarm marker. That is what separates "does not beat the line" from
"beats the line", and it is a check that can fail, which a bare word list scoped to allow
the negation would not be.

The unit is a sentence rather than a line, and that is not a detail: this project hard-wraps
its prose at 88 columns, so "A model that beat the line here **would** be evidence of
leakage" has its claim and its negation on different lines. A line-based scan flagged
exactly that sentence in ``README.md`` the first time it was run. Paragraphs are unwrapped
before splitting.

**Its limits, stated rather than discovered later.** An over-claim spread across two
sentences slips through ("The model beats the line." followed by "That would be a leak."
is caught; the reverse order is not), and a sufficiently creative sentence always can. Exit
criterion 3 asks for a *human* grep as well, and this does not replace it.

**What is not scanned, and why.** ``CLAUDE.md``, ``DECISIONS.md``, ``RISKS.md`` and
``plans/`` are where the rule itself is written down, so they discuss beating the market
constantly and by design. Source code is excluded for the same reason: identifiers like
``model_beats_vegas`` and ``seasons_beating_vegas`` are named after the alarm they raise,
which is the opposite of an over-claim. What is scanned is the prose a reader actually
meets.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from cfb import config

SCANNED_FILES: tuple[str, ...] = (
    "README.md",
    "FEATURES.md",
    "app/README.md",
)
"""Committed prose outside ``results/``. Every ``results/*.md`` file is scanned as well."""

FLAGGED: tuple[tuple[str, str], ...] = (
    (r"\bbeats?\b", "beating the market"),
    (r"\bbeaten\b", "beating the market"),
    (r"\bbeating\b", "beating the market"),
    (r"\boutperform\w*", "outperformance"),
    (r"\bprofitab\w*", "profitability"),
    (r"\bprofits?\b", "profitability"),
    (r"\bedges?\b", "having an edge"),
    (r"\balpha\b", "alpha"),
    (r"\bcrush\w*", "crushing the market"),
    (r"\bfree money\b", "free money"),
    (r"\bguarantee\w*", "a guarantee"),
    (r"\bsure thing\b", "certainty"),
    (r"\b(positive|\+)\s*ev\b", "positive expected value"),
    (r"\bexploit\w*", "exploiting the market"),
)
"""Terms that need a reason to appear, and what each one would be claiming."""

NEGATION_MARKERS: tuple[str, ...] = (
    "not",
    "n't",
    "never",
    "no ",
    "would",
    "cannot",
    "short of",
    "worse",
    "alarm",
    "leak",
    "bug",
    "fail",
    "rather than",
    "instead of",
    "is treated as",
    "does not",
    "without",
)
"""What turns a flagged word into an honest sentence.

A line containing one of these is asserting the *absence* of the thing, or naming it as a
failure condition. "The model does not beat the market" passes; "the model beats the
market" does not.
"""

INNOCENT_SENSES: tuple[str, ...] = (
    "edge case",
    "edge cases",
    "grid edge",
    "at a grid edge",
    "edge of",
    "leading edge",
)
"""Uses of a flagged word that have nothing to do with the betting market.

``edge`` is the only real offender: this project talks about grid edges and edge cases
constantly, and the plan still names "edge" as a term to grep for. Listing the innocent
senses keeps the market sense flagged instead of dropping the word from the scan.
"""


def sentences(text: str) -> list[str]:
    """Split prose into sentences, unwrapping hard-wrapped paragraphs first.

    Markdown blocks are separated by blank lines; within a block every newline is a wrap,
    not a break. Joining the block back into one string before splitting on sentence
    punctuation is what stops a claim and its negation from being scanned separately.

    Args:
        text: File contents.

    Returns:
        Sentences, in order, with whitespace collapsed.
    """
    found = []
    for block in re.split(r"\n\s*\n", text):
        unwrapped = " ".join(block.split())
        found += [part for part in re.split(r"(?<=[.!?])\s+", unwrapped) if part.strip()]
    return found


def offending_sentences(text: str) -> list[tuple[str, str]]:
    """Find flagged terms that are neither negated nor an innocent sense.

    Args:
        text: File contents.

    Returns:
        ``(claim, sentence)`` for each offence; empty when the prose is clean.
    """
    offences = []
    for sentence in sentences(text):
        lowered = sentence.lower()
        if any(sense in lowered for sense in INNOCENT_SENSES):
            continue
        if any(marker in lowered for marker in NEGATION_MARKERS):
            continue
        for pattern, claim in FLAGGED:
            if re.search(pattern, lowered):
                offences.append((claim, sentence))
                break
    return offences


def scanned_paths() -> list[Path]:
    """List the committed prose files that exist right now."""
    paths = [config.PROJECT_ROOT / name for name in SCANNED_FILES]
    paths += sorted(config.RESULTS_DIR.glob("*.md"))
    return [path for path in paths if path.exists()]


def test_there_is_prose_to_scan():
    """A scan over zero files passes trivially, which is the one way this test lies."""
    paths = scanned_paths()
    assert paths, "no prose found to scan; the check would pass vacuously"
    assert any(path.parent == config.RESULTS_DIR for path in paths), (
        "no results/*.md found; run `make evaluate` so the published numbers are scanned too"
    )


@pytest.mark.parametrize("path", scanned_paths(), ids=lambda path: path.name)
def test_published_prose_does_not_claim_to_beat_the_market(path: Path):
    offences = offending_sentences(path.read_text())
    assert not offences, "\n".join(
        f"{path.name} reads as a claim of {claim}: {sentence}" for claim, sentence in offences
    )


# --- The scanner has to be able to fail ---------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "The model beats the closing line on held-out data.",
        "Our model outperforms the market on 2,400 games.",
        "This is a profitable strategy against the spread.",
        "The model has a real edge over the book.",
        "We crush the closing line in 2024.",
        "A positive EV play every week.",
        "The features exploit a market inefficiency.",
    ],
)
def test_the_scanner_catches_an_over_claim(line: str):
    assert offending_sentences(line), f"the scan let an over-claim through: {line!r}"


@pytest.mark.parametrize(
    "line",
    [
        "The model does not beat the market.",
        "A model that beat the line here would be evidence of leakage.",
        "Scoring better than the line is treated as a leakage alarm, not a result.",
        "The model is short of the line by 0.0149 Brier.",
        "40 pinned edge cases, selected by property rather than by id.",
        "the winner sits at a grid edge on learning_rate",
        "Nothing here has been evaluated for profitability, and no such claim is made.",
    ],
)
def test_the_scanner_allows_the_honest_sentences(line: str):
    assert not offending_sentences(line), f"the scan flagged an honest sentence: {line!r}"


def test_the_scanner_reads_a_wrapped_sentence_as_one_sentence():
    """The bug the first version of this file had, pinned so it cannot come back.

    The claim is on one line and the negation that makes it honest is on the next, exactly
    as ``README.md`` wraps it.
    """
    wrapped = "A model that beat the line here would\nbe evidence of leakage, not a result."
    assert not offending_sentences(wrapped)


def test_the_scanner_would_catch_an_over_claim_added_to_a_real_file(tmp_path: Path):
    """End to end: poison a copy of the published results and demand the scan fails."""
    if not config.RESULTS_TABLE_PATH.exists():
        pytest.skip("no results table; run `make evaluate` first")
    poisoned = tmp_path / "results_table.md"
    poisoned.write_text(
        config.RESULTS_TABLE_PATH.read_text()
        + "\n## Conclusion\n\nThe model beats the closing line and is profitable.\n"
    )
    offences = offending_sentences(poisoned.read_text())
    assert offences, "poisoning the results table did not trip the scan"
    assert any(claim == "beating the market" for claim, _ in offences)
