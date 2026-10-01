"""Reference-trajectory filter that blends an original MPC reference with an
RL-proposed reference using an exponentially decaying weight."""
from __future__ import annotations

import numpy as np


def ref_traj_filter(
    original: np.ndarray,
    new: np.ndarray,
    decay: float = 1.0,
) -> np.ndarray:
    """Blend *new* into *original* with exponentially decaying influence.

    At ``decay=1`` the result equals *new* at the first waypoint and
    converges back to *original* at subsequent waypoints.  At ``decay=0``
    the function returns a copy of *original* unchanged.

    Args:
        original: Reference trajectory from the MPC, shape ``(N, d)``.
        new:      Candidate trajectory from the RL policy, shape ``(N, d)``.
        decay:    Initial blend weight in ``[0, 1]``.  The weight is squared
                  at every step; once it falls below 0.01 it is clamped to
                  zero.

    Returns:
        Filtered trajectory with the same shape as *original*.
    """
    filtered = original.copy()
    w = float(decay)
    for i in range(filtered.shape[0]):
        filtered[i, :] = (1.0 - w) * filtered[i, :] + w * new[i, :]
        w *= w
        if w < 1e-2:
            w = 0.0
    return filtered
