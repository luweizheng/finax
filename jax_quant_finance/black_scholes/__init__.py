"""JAX Quantitative Finance volatility surfaces and vanilla options."""

from jax_quant_finance.black_scholes import vanilla_prices

option_price = vanilla_prices.option_price
barrier_price = vanilla_prices.barrier_price
swaption_price = vanilla_prices.swaption_price

_allowed_symbols = [
    'option_price',
    'barrier_price',

    'swaption_price'
]