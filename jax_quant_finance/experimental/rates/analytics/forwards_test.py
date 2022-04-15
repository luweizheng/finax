import numpy as np
import jax.numpy as jnp
from jax_quant_finance.experimental.rates.analytics.forwards import forward_rates
from jax.config import config
config.update("jax_enable_x64", True)


# test forwards rates

# Discount factors at start dates
df_start_dates = [[0.95, 0.9, 0.75], [0.95, 0.99, 0.85]]
# Discount factors at end dates
df_end_dates = [[0.8, 0.6, 0.5], [0.8, 0.9, 0.5]]
# Daycount fractions between the dates
daycount_fractions = [[0.5, 1.0, 2], [0.6, 0.4, 4.0]]

res = forward_rates(df_start_dates, df_end_dates,daycount_fractions, dtype=jnp.float64)

print(res)

