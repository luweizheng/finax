"""JAX Quantitative Finance volatility surfaces and vanilla options."""

from jax_quant_finance.black_scholes import vanilla_prices

option_price = vanilla_prices.option_price
barrier_price = vanilla_prices.barrier_price
swaption_price = vanilla_prices.swaption_price
binary_price = vanilla_prices.binary_price
asset_or_nothing_price = vanilla_prices.asset_or_nothing_price

_allowed_symbols = [
    'option_price',
    'barrier_price',
    'binary_price',
    'asset_or_nothing_price',
    'swaption_price',
]