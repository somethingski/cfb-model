# cfb-model

A college football game-outcome model, built in phases with an explicit anti-leakage
framework.

**Status: Phase 6 (evaluation).** CFBD data lands in SQLite, the de-vigged closing line is
built as the benchmark, a custom Elo rating runs the schedule in kickoff order, a parquet
feature store is built behind a leakage audit that recomputes a sample of its rows from a
database truncated at each game's kickoff, a LightGBM classifier is fitted on 2014-2021 and
calibrated on 2022, and the 2023-2025 seasons have now been scored once.

The project's claim is **calibration approaching the de-vigged Vegas closing line** — not
beating it. On 2,398 held-out games the model is short of the line by **0.0149 Brier**.
Full numbers: [`results/results_table.md`](results/results_table.md) and
[`results/model_card.md`](results/model_card.md). The prose write-up is Phase 7.

## Layout

| Path | Purpose |
|---|---|
| `src/cfb/` | Package: `config`, then `ingest`, `vegas`, `elo`, `features`, `model`, `eval` by phase |
| `plans/` | The phase-by-phase build plans |
| `gold/` | Committed hand-verified fixtures |
| `tests/` | pytest suite, including the leakage audit from Phase 4 |
| `data/`, `cache/` | Gitignored: SQLite DB, feature parquet, raw API responses |

## Development

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

On macOS, LightGBM's wheel links against the OpenMP runtime, which Apple does not ship:
`import lightgbm` fails with `Library not loaded: @rpath/libomp.dylib` until you run
`brew install libomp`.

Copy `.env.example` to `.env` and add a
[CollegeFootballData.com](https://collegefootballdata.com/) API key before Phase 1.

## Ingestion

```bash
make ingest-check   # one live request, confirms the API key works
make ingest         # backfill 2014-2025 into data/cfb.sqlite
```

Every API response is cached under `cache/`, so the backfill is idempotent and resumable:
a second run reproduces the first with zero network calls, and the run reports its network
and cache counts so that claim is checkable. `games.start_date` is the canonical kickoff
clock every later phase orders by.

Coverage gaps (missing lines, missing box scores) are reported per season and left alone —
later phases exclude them explicitly and `RISKS.md` records them. Nothing is interpolated.

## Elo

```bash
make elo-tune       # refit K, HFA and regression on 2014-2021 only, freeze to elo_params.json
make elo            # walk the schedule in kickoff order, write elo_pregame
```

The ratings written for a game are read *before* that game's result is applied, and games
are visited in `(start_date, game_id)` order. Only pre-game ratings are stored — a
post-game column is one join away from a feature that knows its own result.
`tests/test_elo_chronology.py` rebuilds the ratings on a database with every later game
deleted and demands the surviving rows are identical; each of its checks is also run
against a deliberately broken walk, so a check that could never fail would itself fail.

Parameters are fitted on training seasons only and then frozen. Elo alone lands between
the naive home-field baseline and the closing line, and well short of the line:

| 2014-2021, FBS vs FBS, n=5,911 | Brier | Log loss |
|---|---|---|
| naive home baseline (57.5%) | 0.2443 | 0.6817 |
| Elo only | 0.1876 | 0.5520 |
| de-vigged closing line | 0.1708 | 0.5105 |

## Features and the leakage audit

```bash
make features       # build data/features/features.parquet and regenerate FEATURES.md
make audit          # rebuild the store, then try to break it
```

One row per completed game, 28 features, all of them deliberately boring: Elo levels and
their difference, scheduling metadata, rest days, rolling per-team offence and defence over
that team's *previous* games this season, and last season's win percentage. Every column is
documented in [`FEATURES.md`](FEATURES.md), which is generated from the code rather than
maintained by hand.

The cutoff rule is written once, in `priors_before`: a team's games with
`start_date < kickoff`. Feature functions are pure and never touch the database, so the
audit's job is to attack the *selector*, which is where the shift-by-one bug lives.

`make audit` is the hard gate, and `make train` depends on it. For a seeded sample of 200
random games plus 40 pinned edge cases, it builds SQL views exposing only the games that
had kicked off before that game, recomputes every feature by calling the production code,
and demands the identical answer — floats to 1e-9, everything else exactly, and nulls as
strictly as numbers. Elo is re-walked rather than re-read, over a view containing the game
itself with its result blanked. Alongside that it checks that no feature tracks the label
within a season, that the feature module never so much as mentions the betting market, and
that the stored ratings survive deleting half the schedule.

An audit that has only ever passed proves nothing, so `tests/test_audit_catches_leakage.py`
builds a toy league and poisons its feature store four ways — a rolling stat with the shift
dropped, a post-game Elo rating, a back-filled null, and a builder that queries the `lines`
table — and demands the audit fails **and names the poisoned column** each time.

If the audit fails, the fix is in the features. Never in the audit.

## Training and calibration

```bash
make train          # rebuilds the store, re-runs the audit, then fits and calibrates
```

Splits are season-forward and immutable: **train 2014-2021, validation 2022, test
2023-2025**. Hyperparameters are chosen by forward-chaining cross-validation — fit
2014-2018 score 2019, fit 2014-2019 score 2020, fit 2014-2020 score 2021 — over a
24-point LightGBM grid. An isotonic calibrator is then fitted on 2022 alone and its output
clipped to [0.02, 0.98]. FBS-vs-FCS games are excluded from the model frame, leaving 9,085
of 10,373 rows.

The test seasons are kept out by mechanism rather than by care: `assert_no_test_rows` runs
inside `fit_booster` and `assert_validation_only` inside `fit_calibrator`, so the check is
on the only path to a fitted object. `tests/test_train.py` poisons the training frame with
a 2024 row and the calibrator with a 2021 row and demands both raise; with the guards
stubbed out, those four tests go red.

Validation, 2022, 776 FBS-vs-FBS games:

| | Brier | Log loss |
|---|---|---|
| naive home baseline (57.5%) | 0.2440 | — |
| model, raw | 0.2024 | 0.5926 |
| model, calibrated | 0.1951 | 0.5689 |
| de-vigged closing line | 0.1862 | 0.5493 |

The model lands between the naive baseline and the line, closer to the line, and **short of
it by 0.0089 Brier** — which is the intended result. A model that beat the line here would
be evidence of leakage, and `make train` fails with a non-zero exit code if it does.
Everything searched, chosen and scored is in [`models/train_report.json`](models/train_report.json);
two runs of `make train` produce that file byte-for-byte identically, and the same booster.

Both calibration figures are in-sample — the calibrator was fitted on the season it is
scored on. They are not evidence that it generalises, and Phase 6 is where that is tested.

## Evaluation

```bash
make evaluate       # score the held-out seasons and write results/
```

The only code in the project that reads a test-season label. It runs against the saved
model rather than re-fitting one, and refuses to start if the artifacts no longer reproduce
the validation Brier recorded in `train_report.json` — so evaluation cannot change the
thing it measures.

Four systems on one identical game set: **2,398 FBS-vs-FBS games across 2023-2025, every
one of them carrying a closing line**, so nothing was dropped for a missing benchmark. Each
season's coverage is printed rather than assumed, since a season quietly missing its bowls
would shrink the test set toward the easier part of a year and still look fine.

| 2023-2025, n=2,398 | Brier | Log loss |
|---|---|---|
| naive home baseline (57.5%) | 0.2427 | 0.6784 |
| Elo only | 0.1902 | 0.5589 |
| model, calibrated | 0.1913 | 0.5631 |
| de-vigged closing line | 0.1763 | 0.5223 |

**The model is short of the line by 0.0149 Brier**, closing 77.5% of the distance between
the naive baseline and the closing line. That is the intended result. Scoring better than
the line would be evidence of leakage, and `make evaluate` exits non-zero if it ever does.

Two findings worth stating plainly, because they are the first things a careful reader will
notice:

- **The isotonic calibrator did not generalise.** Fitted on 2022, it improved that season
  in-sample and made every held-out season *worse* than leaving the model raw (0.1913
  against 0.1883). `RISKS.md` #25 predicted this before the run. The shipped model was not
  switched to raw on the strength of that, because picking a variant on test-season scores
  is selecting a model on the test set.
- **A one-parameter Elo logistic scores better than the shipped calibrated model.** The
  26-feature booster earns its keep only before calibration, and only narrowly. The honest
  reading is that the rolling box-score features carry little signal Elo does not already
  carry — a finding about the feature set, recorded as `RISKS.md` #26.

`tests/test_no_overclaiming.py` guards the prose itself, so that no published sentence can
drift into claiming an edge over the market: every flagged term has to sit in a sentence
that also negates it, and seven poisoned sentences prove the scan fires. It caught a
sentence in this very section on its first run. `gold/eval_fixture.json` holds one week of
predictions for a human to score in a spreadsheet, and `tests/test_gold_eval.py` fails
until that is done.

See `CLAUDE.md` for project conventions, `DECISIONS.md` for the choice log, and
`RISKS.md` for known gaps.
