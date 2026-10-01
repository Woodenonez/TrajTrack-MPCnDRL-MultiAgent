#!/usr/bin/env python3
"""Run the reusable training module from any working directory."""

from pathlib import Path
import sys


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from training import main

    main()
