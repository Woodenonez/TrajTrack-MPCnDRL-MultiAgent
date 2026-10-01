"""Lightweight wall-clock timer for profiling code sections."""
from __future__ import annotations

import timeit


class PieceTimer:
    """Measure elapsed time for a short section of code.

    Usage::

        timer = PieceTimer()
        do_work()
        elapsed_ms = timer(ms=True)

    Methods:
        __call__: Return elapsed time since construction or last :meth:`reset`.
        reset:    Restart the timer.
    """

    def __init__(self) -> None:
        self._instant = timeit.default_timer()

    def __call__(self, round_decimals: int = 4, ms: bool = False) -> float:
        """Return elapsed time.

        Args:
            round_decimals: Number of decimal places to round to.
            ms:             If ``True``, return milliseconds; otherwise seconds.

        Returns:
            Elapsed time as a rounded float.
        """
        elapsed = timeit.default_timer() - self._instant
        if ms:
            elapsed *= 1000.0
        return round(elapsed, round_decimals)

    def reset(self) -> None:
        """Restart the timer from the current instant."""
        self._instant = timeit.default_timer()
