.PHONY: ingest features train evaluate reproduce test lint

# Stubs fail loudly so a green `make reproduce` can never be mistaken for a real run.

ingest:
	@echo "ingest: not implemented until phase 1" && exit 1

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
