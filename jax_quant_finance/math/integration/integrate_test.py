"""Tests for numeric integration methods."""

import collections
import math
import numpy as np
import numpy as np
import jax.numpy as jnp
import jax_quant_finance as jqf

from jax import config
config.update("jax_enable_x64", True)

from absl.testing import absltest
from absl.testing import parameterized

from jax._src import test_util as jtu # pylint: disable=g-direct-tensorflow-import

jqf_int = jqf.math.integration

IntegrationTestCase = collections.namedtuple('IntegrationTestCase', [
    'func',
    'lower',
    'upper',
    'antiderivative',
])

# pylint:disable=g-long-lambda
BASIC_TEST_CASES = [
    IntegrationTestCase(
        func=lambda x: np.exp(2 * x + 1),
        lower=1.0,
        upper=3.0,
        antiderivative=lambda x: np.exp(2 * x + 1) / 2,
    ),
    IntegrationTestCase(
        func=lambda x: x**5,
        lower=-10.0,
        upper=100.0,
        antiderivative=lambda x: x**6 / 6,
    ),
    IntegrationTestCase(
        func=lambda x: (x**3 + x**2 - 4 * x + 1) / (x**2 + 1)**2,
        lower=0.0,
        upper=10.0,
        antiderivative=lambda x: sum([
            2.5 / (x**2 + 1),
            0.5 * np.log(x**2 + 1),
            np.arctan(x),
        ]),
    ),
    IntegrationTestCase(
        func=lambda x: (np.sinh(2 * x) + 3 * np.sinh(x)) /
        (np.cosh(x)**2 + 2 * np.cosh(0.5 * x)**2),
        lower=2.0,
        upper=4.0,
        antiderivative=lambda x: sum([
            np.log(np.cosh(x)**2 + np.cosh(x) + 1),
            (4 / np.sqrt(3)) * np.arctan((1 + 2 * np.cosh(x)) / np.sqrt(3.0)),
        ]),
    ),
    IntegrationTestCase(
        func=lambda x: np.exp(2 * x) * np.sqrt(np.exp(x) + np.exp(2 * x)),
        lower=2.0,
        upper=4.0,
        antiderivative=lambda x: sum([
            np.sqrt((np.exp(x) + np.exp(2 * x))**3) / 3,
            -(1 + 2 * np.exp(x)) * np.sqrt(np.exp(x) + np.exp(2 * x)) / 8,
            np.log(np.sqrt(1 + np.exp(x)) + np.exp(0.5 * x)) / 8,
        ]),
    ),
    IntegrationTestCase(
        func=lambda x: np.exp(-x**2),
        lower=0.0,
        upper=1.0,
        antiderivative=lambda x: 0.5 * np.sqrt(np.pi) * math.erf(x),
    ),
]

TEST_CASE_RAPID_CHANGE = IntegrationTestCase(
    func=lambda x: 1.0 / np.sqrt(x + 1e-6),
    lower=0.0,
    upper=1.0,
    antiderivative=lambda x: 2.0 * np.sqrt(x + 1e-6),
)


class IntegrationTest(jtu.JaxTestCase):

  def _test_batches_and_types(self, integrate_function, args):
    """Checks handling batches and dtypes."""
    dtypes = [np.float32, np.float64]
    a = [[0.0, 0.0], [0.0, 0.0]]
    b = [[np.pi / 2, np.pi], [1.5 * np.pi, 2 * np.pi]]
    a = [a, a]
    b = [b, b]
    k = np.array([[[[1.0]]], [[[2.0]]]])
    func = lambda x: np.array(k, dtype=x.dtype) * np.sin(x)
    ans = [[[1.0, 2.0], [1.0, 0.0]], [[2.0, 4.0], [2.0, 0.0]]]

    results = []
    for dtype in dtypes:
      lower = np.array(a, dtype=dtype)
      upper = np.array(b, dtype=dtype)
      results.append(integrate_function(func, lower, upper, **args))

    for i in range(len(results)):
      assert results[i].dtype == dtypes[i]
      assert np.allclose(results[i], ans, atol=1e-3)

  def _test_accuracy(self, integrate_function, args, test_case, max_rel_error):
    func = test_case.func
    lower = np.array(test_case.lower, dtype=np.float64)
    upper = np.array(test_case.upper, dtype=np.float64)
    exact = test_case.antiderivative(
        test_case.upper) - test_case.antiderivative(test_case.lower)
    approx = integrate_function(func, lower, upper, **args)
    assert np.abs(approx - exact) <= np.abs(exact) * max_rel_error


  def test_integrate_accuracy(self):
    for test_case in BASIC_TEST_CASES:
      self._test_accuracy(jqf_int.integrate, {}, test_case, 1e-8)
      for method in jqf_int.IntegrationMethod:
        self._test_accuracy(jqf_int.integrate, {'method': method}, test_case,
                            1e-8)


  def test_integrate_int_limits(self):
    for method in jqf_int.IntegrationMethod:
      result = jqf_int.integrate(np.sin, 0, 1, method=method, dtype=np.float64)
      self.assertAllClose(0.459697694, result, atol=1e-9)


  def test_simpson_accuracy(self):
    for test_case in BASIC_TEST_CASES:
      self._test_accuracy(jqf_int.simpson, {}, test_case, 1e-8)

  def test_simpson_rapid_change(self):
    self._test_accuracy(jqf_int.simpson, {'num_points': 1001},
                        TEST_CASE_RAPID_CHANGE, 2e-1)
    self._test_accuracy(jqf_int.simpson, {'num_points': 10001},
                        TEST_CASE_RAPID_CHANGE, 3e-2)
    self._test_accuracy(jqf_int.simpson, {'num_points': 100001},
                        TEST_CASE_RAPID_CHANGE, 5e-4)
    self._test_accuracy(jqf_int.simpson, {'num_points': 1000001},
                        TEST_CASE_RAPID_CHANGE, 3e-6)

if __name__ == '__main__':
    absltest.main(testLoader=jtu.JaxTestLoader())