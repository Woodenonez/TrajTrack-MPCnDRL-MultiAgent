"""Unit tests for drl_mpc_nav.utils."""
from pathlib import Path

from drl_mpc_nav.utils.paths import config_path, model_path, repo_root
from drl_mpc_nav.utils.timing import PieceTimer


class TestPieceTimer:
    def test_elapsed_positive(self):
        timer = PieceTimer()
        elapsed = timer()
        assert elapsed >= 0.0

    def test_ms_mode_larger(self):
        timer = PieceTimer()
        import time
        time.sleep(0.01)
        secs = timer()
        timer.reset()
        time.sleep(0.01)
        ms = timer(ms=True)
        assert ms > secs  # milliseconds > seconds for same duration

    def test_reset_restarts_clock(self):
        import time
        timer = PieceTimer()
        time.sleep(0.05)
        before_reset = timer()
        timer.reset()
        after_reset = timer()
        assert after_reset < before_reset


class TestPaths:
    def test_repo_root_contains_pyproject(self):
        root = repo_root()
        assert (root / "pyproject.toml").exists()

    def test_config_path_returns_path(self):
        p = config_path("mpc_default.yaml")
        assert isinstance(p, Path)
        assert p.name == "mpc_default.yaml"

    def test_model_path_returns_path(self):
        p = model_path("ddpg/ray/best_model.pt")
        assert isinstance(p, Path)
        assert "ddpg" in str(p)
        assert p.is_file()
