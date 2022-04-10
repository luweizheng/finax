import pytest
import numpy as np
import jax.numpy as jnp

from jax.config import config
config.update("jax_enable_x64", True)

from jax import jit
from jax import random
from jax_quant_finance.black_sholes.vanilla_prices import option_price


def test_option_prices():
    """Tests that the BS prices are correct."""
    dtype = jnp.float64
    forwards = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
    strikes = jnp.array([3.0, 3.0, 3.0, 3.0, 3.0], dtype=dtype)
    volatilities = jnp.array([0.0001, 102.0, 2.0, 0.1, 0.4], dtype=dtype)
    expiries = jnp.array(1.0, dtype=dtype)
    option_price_fn = jit(option_price)
    computed_prices = option_price_fn(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards)
    expected_prices = jnp.array(
        [0.0, 2.0, 2.0480684764112578, 1.0002029716043364, 2.0730313058959933], dtype=np.float64)
    
    assert jnp.all(jnp.allclose(expected_prices, computed_prices, 1e-10))

def test_option_prices_normal():
    """Tests that the prices using normal model are correct."""
    dtype = jnp.float64
    forwards = jnp.array([0.01, 0.02, 0.03, 0.03, 0.05], dtype=dtype)
    strikes = jnp.array([0.03, 0.03, 0.03, 0.03, 0.03], dtype=dtype)
    volatilities = jnp.array([0.0001, 0.001, 0.01, 0.005, 0.02], dtype=dtype)
    expiries = 1.0
    computed_prices = option_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards,
            is_normal_volatility=True)

    expected_prices = jnp.array(
        [0.0, 0.0, 0.0039894228040143, 0.0019947114020072, 0.0216663094117537], dtype=dtype)
    assert jnp.all(jnp.allclose(expected_prices, computed_prices, 1e-10))

def test_option_prices_normal_float32():
    """Tests that the prices using normal model are correct."""
    dtype = jnp.float32
    forwards = jnp.array([0.01, 0.02, 0.03, 0.03, 0.05], dtype=dtype)
    strikes = jnp.array([0.03, 0.03, 0.03, 0.03, 0.03], dtype=dtype)
    volatilities = jnp.array([0.0001, 0.001, 0.01, 0.005, 0.02], dtype=dtype)
    expiries = 1.0
    computed_prices = option_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards,
            is_normal_volatility=True,
            dtype=dtype)

    expected_prices = jnp.array(
        [0.0, 0.0, 0.0039894228040143, 0.0019947114020072, 0.0216663094117537], dtype=dtype)
    assert jnp.all(jnp.allclose(expected_prices, computed_prices, 1e-6))

def test_price_zero_vol():
    """Tests that zero volatility is handled correctly."""
    # If the volatility is zero, the option's value should be correct.
    dtype = jnp.float64

    forwards = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=dtype)
    strikes = jnp.array([1.1, 0.9, 1.1, 0.9], dtype=dtype)
    volatilities = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=dtype)
    expiries = 1.0
    is_call_options = jnp.array([True, True, False, False])
    expected_prices = jnp.array([0.0, 0.1, 0.1, 0.0])
    computed_prices = option_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards,
            is_call_options=is_call_options,
            dtype=jnp.float64)
    assert jnp.all(jnp.allclose(expected_prices, computed_prices, 1e-10))

def test_price_zero_expiry():
    """Tests that zero expiry is correctly handled."""
    # If the expiry is zero, the option's value should be correct.
    forwards = jnp.array([1.0, 1.0, 1.0, 1.0])
    strikes = jnp.array([1.1, 0.9, 1.1, 0.9])
    volatilities = jnp.array([0.1, 0.2, 0.5, 0.9])
    expiries = 0.0
    is_call_options = jnp.array([True, True, False, False])
    expected_prices = jnp.array([0.0, 0.1, 0.1, 0.0])
    computed_prices = option_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards,
            is_call_options=is_call_options,
            dtype=jnp.float64)
    assert jnp.all(jnp.allclose(expected_prices, computed_prices, 1e-10))

def test_price_long_expiry_calls():
    """Tests that very long expiry call option behaves like the asset."""
    forwards = jnp.array([1.0, 1.0, 1.0, 1.0])
    strikes = jnp.array([1.1, 0.9, 1.1, 0.9])
    volatilities = jnp.array([0.1, 0.2, 0.5, 0.9])
    expiries = 1e10
    expected_prices = forwards
    computed_prices = option_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards,
            dtype=jnp.float64)
    assert jnp.all(jnp.allclose(expected_prices, computed_prices, 1e-10))

def test_price_vol_and_expiry_scaling():
    """Tests that the price is invariant under vol->k vol, T->T/k**2."""
    key = random.PRNGKey(1234)
    forwards = jnp.exp(random.normal(key))
    volatilities = jnp.exp(random.normal(key) / 2)
    strikes = jnp.exp(random.normal(key))
    expiries = jnp.exp(random.normal(key))
    scaling = 5.0
    base_prices = option_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards,
            dtype=jnp.float64)
    scaled_prices = option_price(
            volatilities=volatilities * scaling,
            strikes=strikes,
            expiries=expiries / scaling / scaling,
            forwards=forwards,
            dtype=jnp.float64)
    assert jnp.all(jnp.allclose(base_prices, scaled_prices, 1e-10))

