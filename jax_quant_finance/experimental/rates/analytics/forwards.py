import jax.numpy as jnp
from jax_quant_finance.utils.ops import divide_no_nan


def forward_rates(df_start_dates, df_end_dates, daycount_fractions, dtype=None, name=None):
    df_start_dates = jnp.asarray(df_start_dates, dtype=dtype)
    df_end_dates = jnp.asarray(df_end_dates, dtype=dtype)
    daycount_fractions = jnp.asarray(daycount_fractions, dtype=dtype)
    daycount = divide_no_nan(df_start_dates, df_end_dates, dtype=dtype) - 1
    return divide_no_nan(daycount, daycount_fractions)
    




def forward_rates_from_yields(yields, times, groups=None, dtype=None, name=None):
    yields = jnp.asarray(yields, dtype=dtype)
    dtype = dtype or yields.dtype
    times = jnp.asarray(times, dtype=dtype) 
    if groups is not None:
        groups = jnp.asarray(groups)
    else:
        groups = jnp.zeros_like(yields, dtype=jnp.int32)
    
    rate_times = yields * times
    # TODO : Implement diff and segment ops in math
