"""Package entry points delegate to the maintained command interfaces."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PackageCliTests(unittest.TestCase):
    def _help(self, module: str) -> str:
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=tempfile.gettempdir(),
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_training_module_exposes_canonical_options(self):
        help_text = self._help("drl_mpc_nav.cli.train")
        self.assertIn("--mode", help_text)
        self.assertIn("--evaluation-episodes", help_text)
        self.assertIn("--no-evaluation", help_text)
        self.assertNotIn("--variant-index", help_text)

    def test_evaluation_module_defaults_to_canonical_evaluation(self):
        help_text = self._help("drl_mpc_nav.cli.evaluate")
        self.assertIn("--workflow", help_text)
        self.assertIn("default: ddpg-eval", help_text)
        self.assertIn("--solver-directory", help_text)

    def test_builder_module_exposes_candidate_destination(self):
        help_text = self._help("drl_mpc_nav.cli.build_mpc_solver")
        self.assertIn("--build-directory", help_text)
        self.assertIn("mpc_solver/candidate", help_text)


if __name__ == "__main__":
    unittest.main()
