"""Tests for cashflows module."""

import numpy as np
import jax_quant_finance as jqf

from jax import config
config.update("jax_enable_x64", True)

from absl.testing import absltest
from absl.testing import parameterized

from jax._src import test_util as jtu


class CashflowsTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      })
  def test_pv_from_yields_no_group(self, dtype):
    yield_rate = 0.04
    coupon_rate = 0.04
    # Fifteen year bond with semi-annual coupons.
    cashflows = np.array(
        [coupon_rate * 500] * 29 + [1000 + coupon_rate * 500], dtype=dtype)
    times = np.linspace(0.5, 15, num=30).astype(dtype)
    expected_pv = 995.50315587
    actual_pv = jqf.rates.analytics.cashflows.pv_from_yields(
            cashflows=cashflows, times=times, yields=yield_rate, dtype=dtype)
    np.testing.assert_allclose(expected_pv, actual_pv)

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      })
  def test_pv_from_yields_grouped(self, dtype):
    yield_rates = [0.07, 0.05]
    # 2 and 3 year bonds with 1000 face value and 4%, 6% semi-annual coupons.
    cashflows = np.array([20, 20, 20, 1020, 30, 30, 30, 30, 30, 1030],
                         dtype=dtype)
    times = np.array([0.5, 1, 1.5, 2, 0.5, 1, 1.50, 2, 2.5, 3], dtype=dtype)
    groups = np.array([0] * 4 + [1] * 6)
    expected_pvs = np.array([942.71187528177757, 1025.7777300221542])
    actual_pvs = jqf.rates.analytics.cashflows.pv_from_yields(
            cashflows, times, yield_rates, groups=groups, dtype=dtype)
    np.testing.assert_allclose(expected_pvs, actual_pvs)

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      })
  def test_pv_zero_yields(self, dtype):
    yield_rates = [0., 0.]
   # 2 and 3 year bonds with 1000 face value and 4%, 6% semi-annual coupons.
    cashflows = np.array([20, 20, 20, 1020, 30, 30, 30, 30, 30, 1030],
                         dtype=dtype)
    times = np.array([0.5, 1, 1.5, 2, 0.5, 1, 1.50, 2, 2.5, 3], dtype=dtype)
    groups = np.array([0] * 4 + [1] * 6)
    expected_pvs = np.array([1080., 1180.])
    actual_pvs = jqf.rates.analytics.cashflows.pv_from_yields(
            cashflows, times, yield_rates, groups=groups, dtype=dtype)
    np.testing.assert_allclose(expected_pvs, actual_pvs)

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      })
  def test_pv_infinite_yields(self, dtype):
    """Tests in the limit of very large yields."""
    yield_rates = [300., 300.]
    # 2 and 3 year bonds with 1000 face value and 4%, 6% semi-annual coupons.
    cashflows = np.array([20, 20, 20, 1020, 30, 30, 30, 30, 30, 1030],
                         dtype=dtype)
    times = np.array([0.5, 1, 1.5, 2, 0.5, 1, 1.50, 2, 2.5, 3], dtype=dtype)
    groups = np.array([0] * 4 + [1] * 6)
    expected_pvs = np.array([0., 0.])
    actual_pvs = jqf.rates.analytics.cashflows.pv_from_yields(
            cashflows, times, yield_rates, groups=groups, dtype=dtype)
    np.testing.assert_allclose(expected_pvs, actual_pvs, atol=1e-9)

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      })
  def test_yields_from_pvs_no_group(self, dtype):
    coupon_rate = 0.04
    # Fifteen year bond with semi-annual coupons.
    cashflows = np.array(
        [coupon_rate * 500] * 29 + [1000 + coupon_rate * 500], dtype=dtype)
    pv = 995.50315587
    times = np.linspace(0.5, 15, num=30).astype(dtype)
    expected_yield_rate = np.array(0.04)
    actual_yield_rate = jqf.rates.analytics.cashflows.yields_from_pv(
            cashflows, times, [pv], dtype=dtype)
    np.testing.assert_allclose(expected_yield_rate, actual_yield_rate, atol=1e-6)

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      })
  def test_yields_from_pv_grouped(self, dtype):
    # 2 and 3 year bonds with 1000 face value and 4%, 6% semi-annual coupons.
    cashflows = np.array([20, 20, 20, 1020, 30, 30, 30, 30, 30, 1030],
                         dtype=dtype)
    times = np.array([0.5, 1, 1.5, 2, 0.5, 1, 1.50, 2, 2.5, 3], dtype=dtype)
    groups = np.array([0] * 4 + [1] * 6)
    pvs = np.array([942.71187528177757, 1025.7777300221542])
    expected_yield_rates = [0.07, 0.05]
    actual_yield_rates = jqf.rates.analytics.cashflows.yields_from_pv(
            cashflows, times, pvs, groups=groups, dtype=dtype)
    np.testing.assert_allclose(
        expected_yield_rates, actual_yield_rates, atol=1e-7)

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      })
  def test_yield_saturated_pv(self, dtype):
    # 2 and 3 year bonds with 1000 face value and 4%, 6% semi-annual coupons.
    cashflows = np.array([20, 20, 20, 1020, 30, 30, 30, 30, 30, 1030],
                         dtype=dtype)
    times = np.array([0.5, 1, 1.5, 2, 0.5, 1, 1.50, 2, 2.5, 3], dtype=dtype)
    groups = np.array([0] * 4 + [1] * 6)
    pvs = np.array([1080., 1180.])
    expected_yields = [0., 0.]
    actual_yields = jqf.rates.analytics.cashflows.yields_from_pv(
            cashflows, times, pvs, groups=groups, dtype=dtype)
    np.testing.assert_allclose(expected_yields, actual_yields, atol=1e-9)

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      })
  def test_yield_small_pv(self, dtype):
    """Tests in the limit where implied yields are high."""
    # 2 and 3 year bonds with 1000 face value and 4%, 6% semi-annual coupons.
    cashflows = np.array([20, 20, 20, 1020, 30, 30, 30, 30, 30, 1030],
                         dtype=dtype)
    times = np.array([0.5, 1, 1.5, 2, 0.5, 1, 1.50, 2, 2.5, 3], dtype=dtype)
    groups = np.array([0] * 4 + [1] * 6)
    pvs = np.array([7.45333412e-05, 2.27476813e-08])
    expected_yields = [25.0, 42.0]
    actual_yields = jqf.rates.analytics.cashflows.yields_from_pv(
            cashflows,
            times,
            pvs,
            groups=groups,
            dtype=dtype,)
    np.testing.assert_allclose(expected_yields, actual_yields, atol=1e-9)

  @parameterized.named_parameters(
      {
          'testcase_name': 'SinglePrecision',
          'dtype': np.float32
      }, {
          'testcase_name': 'DoublePrecision',
          'dtype': np.float64
      })
  def test_discount_factors(self, dtype):
    """Tests docstring discount factors."""
    # 2 and 3 year bonds with 1000 face value and 4%, 6% semi-annual coupons.
    cashflows = [[20, 20, 20, 1020, 0, 0],
                 [30, 30, 30, 30, 30, 1030]]
    discount_factors = [[0.96, 0.93, 0.9, 0.87, 1.0, 1.0],
                        [0.97, 0.95, 0.93, 0.9, 0.88, 0.86]]
    expected_prices = [943.2, 1024.7]
    actual_prices = jqf.rates.analytics.cashflows.present_value(
            cashflows,
            discount_factors,
            dtype=dtype)
    np.testing.assert_allclose(expected_prices, actual_prices, atol=1e-9)


if __name__ == '__main__':
    absltest.main(testLoader=jtu.JaxTestLoader())