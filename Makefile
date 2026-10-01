.PHONY: help install test lint format typecheck eval build-solver

PYTHON ?= python
UV ?= uv

help:
	@printf '%s\n' 'install  test  lint  format  typecheck  eval  build-solver'

install:
	$(UV) sync --locked

test:
	$(UV) run --locked pytest -q

lint:
	$(UV) run --locked ruff check src/drl_mpc_nav tests/unit tests/test_package_cli.py

format:
	$(UV) run --locked ruff format src/drl_mpc_nav tests/unit tests/test_package_cli.py

typecheck:
	@# Check package sources; the canonical scripts and third-party stubs have separate boundaries.
	$(UV) run --locked mypy --follow-imports=silent --disable-error-code=import-untyped src/drl_mpc_nav

eval:
	$(UV) run --locked $(PYTHON) scripts/run_experiment.py --workflow ddpg-eval --visualization off --trials 1 --max-steps 10 --solver-directory mpc_solver/candidate

build-solver:
	$(UV) run --locked $(PYTHON) scripts/build_solver.py --build-directory mpc_solver/candidate
