import jax.numpy as jnp
from jax_quant_finance.experimental.math.dividend_no_nan import divide_no_nan


def forward_rates(df_start_dates, df_end_dates, daycount_fractions, dtype=None, name=None):
    df_start_dates = jnp.asarray(df_start_dates, dtype=dtype)
    df_end_dates = jnp.asarray(df_end_dates, dtype=dtype)
    daycount_fractions = jnp.asarray(daycount_fractions, dtype=dtype)

    # divide no nan
    # use double where tricks https://github.com/google/jax/issues/5039 to enable grad work normally
    #_df_end_dates = jnp.where(df_end_dates==0.0, 1.0, df_end_dates)
    #daycount = jnp.where(df_end_dates==0.0, 1.0, df_start_dates/_df_end_dates-1)
    #_daycount_fractions = jnp.where(daycount_fractions==0.0, 1.0, daycount_fractions)
    #return jnp.where(daycount==0.0, 1.0, daycount/_daycount_fractions)
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
    # TODO : Impl diff and segment ops in math
