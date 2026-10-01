"""Optional copies of existing experiment values, with no additional model calls."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EpisodeRecord:
    """Samples at the end of each completed outer control iteration.

    Environment and MPC states use their own clocks: the environment may still
    reflect the synchronization before the latest MPC solve. RL actions are the
    policy outputs applied in pure RL (acceleration or discrete index). Hybrid
    policy proposals are not applied controls; their references are recorded.
    MPC actions are the applied velocity controls, absent for pure RL. Timing
    retains the existing runner's labels and units, including legacy quirks.
    """

    environment_states: list[np.ndarray] = field(default_factory=list)
    mpc_states: list[np.ndarray] = field(default_factory=list)
    chosen_references: list[np.ndarray] = field(default_factory=list)
    rl_actions: list[np.ndarray] = field(default_factory=list)
    mpc_actions: list[np.ndarray] = field(default_factory=list)
    timing_ms: list[float] = field(default_factory=list)
    success: bool = False
    collided: bool = False
    budget_exhausted: bool = False
    termination: str = "unknown"


@dataclass
class RunMetadata:
    workflow: str
    methods: list[str]
    scene: tuple[int, int, int] | None
    case_index: int | None
    seed: int | None
    trials: int
    max_steps: int | None
    visualization: bool
    commit: str
    dirty: bool
    versions: dict[str, str]
    artifact_paths: dict[str, str]
    artifact_sha256: dict[str, str]
    mpc_yaml: str
    details: dict[str, Any] = field(default_factory=dict)


def write_episode(record: EpisodeRecord, path: Path) -> None:
    """Write independent, pickle-free arrays without replacing an earlier run."""
    arrays: dict[str, np.ndarray] = {
        name: np.asarray(value) for name, value in asdict(record).items()
    }
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def write_metadata(metadata: RunMetadata, path: Path) -> None:
    """Write metadata once; callers reserve the run directory first."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(asdict(metadata), stream, indent=2, allow_nan=False)
        stream.write("\n")
