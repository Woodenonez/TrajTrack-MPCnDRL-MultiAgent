"""Quantitative DDPG evaluation using the retained experiment launcher."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.run_experiment import ExperimentArgs


def _launcher() -> ModuleType:
    repository = Path(__file__).resolve().parents[3]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from scripts import run_experiment

    return run_experiment


def parse_args(argv: Sequence[str] | None = None) -> ExperimentArgs:
    """Parse canonical experiment options with evaluation selected by default."""
    import tyro

    launcher = _launcher()
    return tyro.cli(
        launcher.ExperimentArgs,
        default=launcher.ExperimentArgs(workflow="ddpg-eval"),
        args=argv,
    )


def main(argv: Sequence[str] | None = None) -> None:
    launcher = _launcher()
    launcher.run_experiment(parse_args(argv))


if __name__ == "__main__":
    main()
