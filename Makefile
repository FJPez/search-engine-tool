.PHONY: help install sync run lint format typecheck test cov check clean hooks export-reqs

help:
	@echo "Targets:"
	@echo "  install     Install all deps (incl. dev) via uv sync"
	@echo "  run         Launch the interactive search shell"
	@echo "  hooks       Install pre-commit git hooks"
	@echo "  lint        Run ruff lint"
	@echo "  format      Run ruff format"
	@echo "  typecheck   Run ty type checker"
	@echo "  test        Run pytest"
	@echo "  cov         Run pytest with coverage"
	@echo "  check       Run lint + typecheck + test"
	@echo "  export-reqs Regenerate requirements.txt from uv.lock"
	@echo "  clean       Remove caches and build artefacts"

install sync:
	uv sync

run:
	uv run python -m src.main

hooks:
	uv run pre-commit install

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run ty check

test:
	uv run pytest || [ $$? -eq 5 ]

cov:
	uv run pytest --cov=src --cov-report=term-missing

check: lint typecheck test

export-reqs:
	uv export --no-hashes --format requirements-txt --no-dev > requirements.txt

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov **/__pycache__
