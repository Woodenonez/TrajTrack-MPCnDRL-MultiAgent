.PHONY: help install test lint format typecheck eval build-solver

PYTHON ?= python
UV     ?= uv

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package and dev dependencies
	$(UV) sync

test:  ## Run fast unit tests (excludes slow/solver/model tests)
	$(UV) run pytest -m "not slow and not requires_solver and not requires_model" -v

test-all:  ## Run all tests including slow ones
	$(UV) run pytest -v

lint:  ## Run ruff linter
	$(UV) run ruff check .

format:  ## Auto-format code with ruff
	$(UV) run ruff format .

typecheck:  ## Run mypy type-checker
	$(UV) run mypy src/drl_mpc_nav

eval:  ## Run a short hybrid evaluation (no plot, 1 trial)
	$(UV) run python -m drl_mpc_nav.cli.evaluate \
	  --decision-mode hybrid --rl-index 0 --max-steps 10 --plot False

build-solver:  ## Rebuild the MPC solver (destructive — prompts before overwriting default)
	$(UV) run python -m drl_mpc_nav.cli.build_mpc_solver --verbose True
