"""Tests for diff.py."""

import numpy as np
import jax.numpy as jnp
from jax_quant_finance import math

from jax import config
config.update("jax_enable_x64", True)

from absl.testing import absltest
from absl.testing import parameterized

from jax._src import test_util as jtu

class DiffOpsTest(jtu.JaxTestCase):

    def test_diffs(self):
        x = np.array([1, 2, 3, 4, 5])
        dx = math.diff(x, order=1, exclusive=False)
        np.testing.assert_array_equal(dx, [1, 1, 1, 1, 1])

        dx1 = math.diff(x, order=1, exclusive=True)
        np.testing.assert_array_equal(dx1, [1, 1, 1, 1])

        dx2 = math.diff(x, order=2, exclusive=False)
        np.testing.assert_array_equal(dx2, [1, 2, 2, 2, 2])

    
    # TODO: check grad works for diff
    # def test_diffs_differentiable(self):
    #     """Tests that the diffs op is differentiable."""
    #     x = jnp.asarray(2.0)
    #     xv = jnp.stack([x, x * x, x * x * x], axis=0)

    #     # Produces [x, x^2 - x, x^3 - x^2]
    #     dxv = ops.diff(xv)
    #     np.testing.assert_array_equal(dxv, [2., 2., 4.])

    #     gradient = grad(ops.diff(xv), x)[0]
    #     # The sum of [1, 2x-1, 3x^2-2x] at x = 2 is 12.
    #     self.assertEqual(gradient, 12.0)

    @parameterized.named_parameters(
        {
            'testcase_name': 'exclusive_0',
            'exclusive': True,
            'axis': 0,
            'dx_true': np.array([[9, 18, 27, 36]])
        }, {
            'testcase_name': 'exclusive_1',
            'exclusive': True,
            'axis': 1,
            'dx_true': np.array([[1, 1, 1], [10, 10, 10]])
        }, {
            'testcase_name': 'nonexclusive_0',
            'exclusive': False,
            'axis': 0,
            'dx_true': np.array([[1, 2, 3, 4], [9, 18, 27, 36]]),
        }, {
            'testcase_name': 'nonexclusive_1',
            'exclusive': False,
            'axis': 1,
            'dx_true': np.array([[1, 1, 1, 1], [10, 10, 10, 10]]),
        },
    )
    def test_batched_axis(self, exclusive, axis, dx_true):
        """Tests batch diff works with axis argument use of exclusivity."""
        x = np.array([[1, 2, 3, 4], [10, 20, 30, 40]])
        dx = math.diff(x, order=1, exclusive=exclusive, axis=axis)
        self.assertArraysEqual(dx, dx_true)


if __name__ == '__main__':
    absltest.main(testLoader=jtu.JaxTestLoader())