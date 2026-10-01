"""Unit tests for drl_mpc_nav.hybrid.ref_filter."""
import numpy as np

from drl_mpc_nav.hybrid.ref_filter import ref_traj_filter


class TestRefTrajFilter:
    def _make(self, n=5, d=2):
        original = np.zeros((n, d))
        new = np.ones((n, d))
        return original, new

    def test_decay_zero_returns_original(self):
        original, new = self._make()
        result = ref_traj_filter(original, new, decay=0.0)
        np.testing.assert_array_equal(result, original)

    def test_decay_one_first_row_equals_new(self):
        original, new = self._make()
        result = ref_traj_filter(original, new, decay=1.0)
        np.testing.assert_array_almost_equal(result[0], new[0])

    def test_output_shape_preserved(self):
        original = np.random.rand(10, 3)
        new = np.random.rand(10, 3)
        result = ref_traj_filter(original, new, decay=0.5)
        assert result.shape == original.shape

    def test_does_not_modify_original(self):
        original = np.zeros((5, 2))
        new = np.ones((5, 2))
        original_copy = original.copy()
        ref_traj_filter(original, new, decay=1.0)
        np.testing.assert_array_equal(original, original_copy)

    def test_blend_is_monotonically_decreasing_weight(self):
        """Later waypoints should be closer to *original* than earlier ones."""
        original = np.zeros((6, 2))
        new = np.ones((6, 2)) * 10
        result = ref_traj_filter(original, new, decay=0.9)
        # row 0 should be heavily weighted towards new; row -1 towards original
        assert result[0, 0] > result[-1, 0]
