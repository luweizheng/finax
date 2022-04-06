import pytest
import jax.numpy as jnp

from jax_quant_finance.black_sholes.vanilla_prices import option_price

def test_option_prices():
    """Tests that the BS prices are correct."""
    forwards = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    strikes = jnp.array([3.0, 3.0, 3.0, 3.0, 3.0])
    volatilities = jnp.array([0.0001, 102.0, 2.0, 0.1, 0.4])
    expiries = 1.0
    computed_prices = option_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards)
    expected_prices = jnp.array(
        [0.0, 2.0, 2.0480684764112578, 1.0002029716043364, 2.0730313058959933])
    assert jnp.all(jnp.allclose(expected_prices, computed_prices, 1e-6))

def test_option_prices_normal():
    """Tests that the prices using normal model are correct."""
    forwards = jnp.array([0.01, 0.02, 0.03, 0.03, 0.05])
    strikes = jnp.array([0.03, 0.03, 0.03, 0.03, 0.03])
    volatilities = jnp.array([0.0001, 0.001, 0.01, 0.005, 0.02])
    expiries = 1.0
    computed_prices = option_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards,
            is_normal_volatility=True)

    expected_prices = jnp.array(
        [0.0, 0.0, 0.0039894228040143, 0.0019947114020072, 0.0216663094117537])
    assert jnp.all(jnp.allclose(expected_prices, computed_prices, 1e-6))