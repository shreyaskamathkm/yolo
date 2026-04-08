VENV ?= .venv
PYTHON ?= python3
PIP ?= pip

.PHONY: help setup install \
        test test-model test-tools test-utils \
        lint pre-commit-install pre-commit-run pre-commit-all \
        docs clean test-all ci

# Default target
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Setup"
	@echo "  install              Install all dependencies (uses ambient pip, for CI)"
	@echo "  setup                Create venv if needed, pip install, and pre-commit install"
	@echo ""
	@echo "Testing"
	@echo "  test                 Run all tests with coverage"
	@echo "  test-model           Run model tests only"
	@echo "  test-tools           Run tools tests only"
	@echo "  test-utils           Run utils tests only"
	@echo ""
	@echo "Code Quality"
	@echo "  lint                 Run pre-commit hooks on all files"
	@echo "  pre-commit-install   Install pre-commit hooks"
	@echo "  pre-commit-run       Run pre-commit on changed files only"
	@echo "  pre-commit-all       Run pre-commit on all files"
	@echo ""
	@echo "Docs"
	@echo "  docs                 Build HTML documentation"
	@echo ""
	@echo "CI"
	@echo "  test-all             Run lint + all tests"
	@echo "  ci                   Full CI pipeline (install-dev + install-editable + lint + test)"
	@echo ""
	@echo "Misc"
	@echo "  clean                Remove build artifacts, caches, and coverage reports"

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	@echo "--- Installing dependencies ---"
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

setup:
	@if [ ! -d "$(VENV)" ]; then \
		echo "--- Creating virtual environment at $(VENV) ---"; \
		$(PYTHON) -m venv $(VENV); \
	else \
		echo "--- Virtual environment $(VENV) already exists, skipping creation ---"; \
	fi
	@$(MAKE) install PIP=$(VENV)/bin/pip
	@echo "--- Installing pre-commit hooks ---"
	$(VENV)/bin/pre-commit install
	@echo "--- Development environment ready ---"

# ── Tests ────────────────────────────────────────────────────────────────────

test:
	@echo "--- Running all tests with coverage ---"
	pytest tests/ --cov=yolo --cov-report=term-missing --cov-report=html

test-model:
	@echo "--- Running model tests ---"
	pytest tests/test_model/ -v

test-tools:
	@echo "--- Running tools tests ---"
	pytest tests/test_tools/ -v

test-utils:
	@echo "--- Running utils tests ---"
	pytest tests/test_utils/ -v

# ── Code Quality ─────────────────────────────────────────────────────────────

pre-commit-install:
	@echo "--- Installing pre-commit hooks ---"
	pre-commit install

pre-commit-run:
	@echo "--- Running pre-commit on changed files ---"
	pre-commit run

pre-commit-all:
	@echo "--- Running pre-commit on all files ---"
	pre-commit run --all-files

lint: pre-commit-all

# ── Docs ─────────────────────────────────────────────────────────────────────

docs:
	@echo "--- Building HTML documentation ---"
	$(MAKE) -C docs html

# ── CI ───────────────────────────────────────────────────────────────────────

test-all: lint test
	@echo "--- All checks passed ---"

ci: setup lint test
	@echo "--- CI pipeline complete ---"

# ── Clean ────────────────────────────────────────────────────────────────────

clean:
	@echo "--- Cleaning build artifacts and caches ---"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/
	rm -rf .coverage htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf docs/_build/