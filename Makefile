.PHONY: ingest ingest-check benchmark elo elo-tune features train evaluate reproduce test lint

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

features:
	@echo "features: not implemented until phase 4" && exit 1

train:
	@echo "train: not implemented until phase 5" && exit 1

evaluate:
	@echo "evaluate: not implemented until phase 6" && exit 1

reproduce: ingest features train evaluate
	@echo "reproduce: not implemented until phase 7" && exit 1

test:
	pytest

lint:
	ruff check src tests
	ruff format --check src tests
