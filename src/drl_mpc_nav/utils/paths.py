"""Project-wide path resolution helpers."""
from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the absolute path to the repository root.

    The root is defined as the directory containing ``pyproject.toml``,
    found by walking up from this file's location.

    Returns:
        Absolute :class:`~pathlib.Path` to the repository root.

    Raises:
        FileNotFoundError: If ``pyproject.toml`` cannot be found.
    """
    candidate = Path(__file__).resolve()
    for parent in candidate.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError(
        "Could not locate pyproject.toml in any parent directory of "
        f"{__file__}"
    )


def config_path(filename: str) -> Path:
    """Return the absolute path to a file inside the ``config/`` directory.

    Args:
        filename: File name (e.g. ``"mpc_default.yaml"``).

    Returns:
        Absolute :class:`~pathlib.Path`.
    """
    return repo_root() / "config" / filename


def model_path(relative: str) -> Path:
    """Return the absolute path to a shipped policy under ``pretrained_model/``.

    Args:
        relative: Path relative to ``pretrained_model/`` (e.g. ``"ddpg/ray/best_model.pt"``).

    Returns:
        Absolute :class:`~pathlib.Path`.
    """
    return repo_root() / "pretrained_model" / relative
