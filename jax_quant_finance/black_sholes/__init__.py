"""JAX Quantitative Finance volatility surfaces and vanilla options."""

from jax_quant_finance.black_scholes import vanilla_prices

option_price = vanilla_prices.option_price

_allowed_symbols = [
    'option_price'
]