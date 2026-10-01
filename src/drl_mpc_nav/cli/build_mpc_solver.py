"""Build command delegating to the candidate-safe solver builder."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> None:
    repository = Path(__file__).resolve().parents[3]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from scripts.build_solver import main as build_main

    build_main(argv)


if __name__ == "__main__":
    main()
