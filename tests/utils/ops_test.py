"""Tests for ops."""

import numpy as np
import jax.numpy as jnp
from jax import grad

from jax import config
config.update("jax_enable_x64", True)

from absl.testing import absltest
from absl.testing import parameterized

from jax._src import test_util as jtu

import jax_quant_finance.utils.ops as ops

class OpsTest(jtu.JaxTestCase):

    @parameterized.named_parameters(
        {
            "testcase_name": "scalar",
            "x": 3.0,
            "y": 0.0,
            "expected": 0.0
        }, {
            "testcase_name": "vector",
            "x": [3.0, 2.0, 1.5],
            "y": [0.0, 1.0, 2.5],
            "expected": [0.0, 2.0, 0.6]
        }
    )
    def test_divide_non_nan(self, x, y, expected):
        computed = ops.divide_no_nan(x, y)

        np.testing.assert_allclose(computed, expected)

if __name__ == '__main__':
    absltest.main(testLoader=jtu.JaxTestLoader())