import numpy as np
import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

from jax_quant_finance.experimental.rates.analytics.cashflows import yields_from_pv, pv_from_yields

def test_pv_from_yields_no_group(dtype):
    yield_rate = 0.04
    coupon_rate = 0.04

    cashflows = jnp.asarray(
        [coupon_rate * 500] * 29 + [1000 + coupon_rate * 500], dtype=dtype
    )

    times = jnp.linspace(0.5, 15, num=30, dtype=dtype)
    expected_pv = 995.50315587
    actual_pv = pv_from_yields(cashflows=cashflows, times=times, yields=yield_rate, dtype=dtype )

    return jnp.allclose(expected_pv, actual_pv)

test1 = test_pv_from_yields_no_group(jnp.float64)

print(test1)


def test_yields_from_pvs_no_group(dtype):
    coupon_rate = 0.04
    cashflows = jnp.array(
        [coupon_rate * 500] * 29 + [1000 + coupon_rate * 500], dtype=dtype)
    pv = 995.50315587
    times = jnp.linspace(0.5, 15, num=30, dtype=dtype)
    expected_yield_rate = 0.04    
    actual_yield_rate = yields_from_pv(cashflows=cashflows, times=times, present_values=[pv], dtype=dtype)

    return jnp.allclose(expected_yield_rate, actual_yield_rate)


def test_pv_from_yields_grouped():
    # 跟tf的test类似，不过这个更明确，分成两组
    cashflows_list = [np.array([20, 20, 20, 1020]), np.array([30, 30, 30, 30, 30, 1030])]
    times_list = [0.07, 0.05]
    yield_rate_list = [np.array([0.5, 1, 1.5, 2]), np.array([0.5, 1, 1.50, 2, 2.5, 3])]
    
    expected_pvs = np.array([942.71187528177757, 1025.7777300221542])
    
    # 使用tree_map
    actual_pvs = jax.tree_map(lambda cashflows, times, yield_rate: pv_from_yields(cashflows, times, yield_rate), cashflows_list, times_list, yield_rate_list)
    print(f"test_pv_from_yields_grouped {actual_pvs}")

    # np.testing.assert_allclose(expected_pvs, actual_pvs)

test2 = test_yields_from_pvs_no_group(jnp.float64)

print(test2)
