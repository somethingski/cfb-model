.PHONY: ingest ingest-check benchmark elo elo-tune features audit train evaluate reproduce test lint

# Stubs fail loudly so a green `make reproduce` can never be mistaken for a real run.

ingest:
	python -m cfb.ingest.backfill

# One live request to confirm CFBD_API_KEY works before committing to a full backfill.
ingest-check:
	python -m cfb.ingest.backfill --check

# The yardstick. Derived from `lines`; safe to rebuild at any time.
benchmark:
	python -m cfb.vegas.benchmark

# Pre-game ratings. Derived from the game spine; safe to rebuild at any time.
elo:
	python -m cfb.elo.pipeline

# Refits K, HFA and the regression coefficient on training seasons only, then freezes them
# to elo_params.json. Run `make elo` afterwards so the table matches the file.
elo-tune:
	python -m cfb.elo.tune

# The parquet feature store, plus the generated FEATURES.md that documents it.
features:
	python -m cfb.features.build
	python -m cfb.features.docs

# The Phase 4 gate. Rebuilds the store first, so it always runs against a fresh build.
# If this fails, the fix is in the features, never in the audit (CLAUDE.md).
audit: features
	python -m cfb.features.audit

# Depending on audit is what makes the gate mechanical rather than honour-system:
# there is no path to a trained model that does not run the leakage audit first.
train: audit
	python -m cfb.model.train

# The only target that reads a test-season label. Deliberately does NOT depend on `train`:
# evaluation must not be able to change the model it is measuring. It fails instead if the
# artifacts are missing, or if they no longer reproduce train_report.json's validation Brier.
evaluate:
	python -m cfb.eval.evaluate

reproduce: ingest benchmark elo audit train evaluate
	@echo "reproduce: not implemented until phase 7" && exit 1

test:
	pytest

lint:
	ruff check src tests
	ruff format --check src tests
