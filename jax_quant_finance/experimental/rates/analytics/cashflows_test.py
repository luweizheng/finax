import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

from jax_quant_finance.experimental.rates.analytics.cashflows import pv_from_yields
from jax_quant_finance.experimental.rates.analytics.cashflows import yields_from_pv

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


test2 = test_yields_from_pvs_no_group(jnp.float64)

print(test2)
