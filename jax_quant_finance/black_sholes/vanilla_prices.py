"""Black Scholes prices of European options."""
from typing import Union

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

__all__ = [
    'option_price',
]

@jit
def divide_no_nan(x, y):
  return jnp.where(y != 0, x / y, 0)


_SQRT_2 = np.sqrt(2.0, dtype=np.float64)

@jit
def _ncdf(x):
  return (jax.lax.erf(x / _SQRT_2) + 1) / 2


def option_price(*,
                 volatilities,
                 strikes,
                 expiries,
                 spots = None,
                 forwards = None,
                 discount_rates = None,
                 dividend_rates = None,
                 discount_factors = None,
                 is_call_options = None,
                 is_normal_volatility: bool = False,
                 dtype:jnp.dtype = jnp.float64,
                 ) -> jnp.ndarray:
    """Computes the Black Scholes price for a batch of call or put options.

    #### Example

    ```python
        # Price a batch of 5 vanilla call options.
        volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])
        forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Strikes will automatically be broadcasted to shape [5].
        strikes = np.array([3.0])
        # Expiries will be broadcast to shape [5], i.e. each option has strike=3
        # and expiry = 1.
        expiries = 1.0
        computed_prices = tff.black_scholes.option_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            forwards=forwards)
    # Expected print output of computed prices:
    # [ 0.          2.          2.04806848  1.00020297  2.07303131]
    ```

    #### References:
    [1] Hull, John C., Options, Futures and Other Derivatives. Pearson, 2018.
    [2] Wikipedia contributors. Black-Scholes model. Available at:
        https://en.wikipedia.org/w/index.php?title=Black%E2%80%93Scholes_model

    Args:
        volatilities: Real `Tensor` of any shape and dtype. The volatilities to
        expiry of the options to price.
        strikes: A real `Tensor` of the same dtype and compatible shape as
        `volatilities`. The strikes of the options to be priced.
        expiries: A real `Tensor` of same dtype and compatible shape as
        `volatilities`. The expiry of each option. The units should be such that
        `expiry * volatility**2` is dimensionless.
        spots: A real `Tensor` of any shape that broadcasts to the shape of the
        `volatilities`. The current spot price of the underlying. Either this
        argument or the `forwards` (but not both) must be supplied.
        forwards: A real `Tensor` of any shape that broadcasts to the shape of
        `volatilities`. The forwards to maturity. Either this argument or the
        `spots` must be supplied but both must not be supplied.
        discount_rates: An optional real `Tensor` of same dtype as the
        `volatilities` and of the shape that broadcasts with `volatilities`.
        If not `None`, discount factors are calculated as e^(-rT),
        where r are the discount rates, or risk free rates. At most one of
        `discount_rates` and `discount_factors` can be supplied.
        Default value: `None`, equivalent to r = 0 and discount factors = 1 when
        `discount_factors` also not given.
        dividend_rates: An optional real `Tensor` of same dtype as the
        `volatilities` and of the shape that broadcasts with `volatilities`.
        Default value: `None`, equivalent to q = 0.
        discount_factors: An optional real `Tensor` of same dtype as the
        `volatilities`. If not `None`, these are the discount factors to expiry
        (i.e. e^(-rT)). Mutually exclusive with `discount_rates`. If neither is
        given, no discounting is applied (i.e. the undiscounted option price is
        returned). If `spots` is supplied and `discount_factors` is not `None`
        then this is also used to compute the forwards to expiry. At most one of
        `discount_rates` and `discount_factors` can be supplied.
        Default value: `None`, which maps to e^(-rT) calculated from
        discount_rates.
        is_call_options: A boolean `Tensor` of a shape compatible with
        `volatilities`. Indicates whether the option is a call (if True) or a put
        (if False). If not supplied, call options are assumed.
        is_normal_volatility: An optional Python boolean specifying whether the
        `volatilities` correspond to lognormal Black volatility (if False) or
        normal Black volatility (if True).
        Default value: False, which corresponds to lognormal volatility.
        dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
        of any supplied non-`Tensor` arguments to `Tensor`.
        Default value: `None` which maps to the default dtype inferred by
            TensorFlow.
        name: str. The name for the ops created by this function.
        Default value: `None` which is mapped to the default name `option_price`.

    Returns:
        option_prices: A `Tensor` of the same shape as `forwards`. The Black
        Scholes price of the options.

    Raises:
        ValueError: If both `forwards` and `spots` are supplied or if neither is
        supplied.
        ValueError: If both `discount_rates` and `discount_factors` is supplied.
    """
    print(type(expiries))
    if (spots is None) == (forwards is None):
        raise ValueError('Either spots or forwards must be supplied but not both.')
    if (discount_rates is not None) and (discount_factors is not None):
        raise ValueError('At most one of discount_rates and discount_factors may '
                        'be supplied')
    
    dtype = dtype or jnp.float64

    if discount_rates is not None:
        discount_factors = jnp.exp(-discount_rates * expiries)    
    elif discount_factors is not None:
        discount_rates = -jnp.log(discount_factors) / expiries
    else:
        discount_rates = jnp.array(0.0, dtype=dtype)
        discount_factors = jnp.array(1.0, dtype=dtype)

    if dividend_rates is None:
        dividend_rates = jnp.asarray(0.0, dtype=dtype)

    if forwards is None:
        forwards = spots * jnp.exp((discount_rates - dividend_rates) * expiries)

    sqrt_var = volatilities * jnp.sqrt(expiries)
    if not is_normal_volatility:  # lognormal model
        d1 = divide_no_nan(jnp.log(forwards / strikes), sqrt_var) + sqrt_var / 2
        d2 = d1 - sqrt_var
        undiscounted_calls = jnp.where(sqrt_var > 0,
                                    forwards * _ncdf(d1) - strikes * _ncdf(d2),
                                    jnp.maximum(forwards - strikes, 0.0))
    else:  # normal model
        d1 = divide_no_nan((forwards - strikes), sqrt_var)
        undiscounted_calls = jnp.where(
            sqrt_var > 0.0, (forwards - strikes) * _ncdf(d1) +
            sqrt_var * jnp.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi),
            jnp.maximum(forwards - strikes, 0.0))

    if is_call_options is None:
        return discount_factors * undiscounted_calls
    
    undiscounted_forward = forwards - strikes
    undiscounted_puts = undiscounted_calls - undiscounted_forward
    predicate = jnp.broadcast_to(is_call_options, undiscounted_calls.shape)
    
    return discount_factors * jnp.where(predicate, undiscounted_calls, undiscounted_puts)