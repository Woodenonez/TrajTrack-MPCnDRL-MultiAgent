"""Training module entry point; see ``python -m drl_mpc_nav.cli.train --help``.

This delegates to the retained trainer in ``src/training.py``. Its active
variants are index 0 (image/CUDA) and index 1 (ray/CPU).
"""

from __future__ import annotations

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    from training import main as training_main

    training_main(argv)


if __name__ == "__main__":
    main()
