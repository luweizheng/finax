"""Black Scholes prices of European options."""
from typing import Union, List

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
                 volatilities: Union[jnp.ndarray, np.ndarray, List[float]],
                 strikes: Union[jnp.ndarray, np.ndarray, List[float]],
                 expiries: Union[jnp.ndarray, np.ndarray, List[float]],
                 spots: Union[jnp.ndarray, np.ndarray, List[float]] = None,
                 forwards: Union[jnp.ndarray, np.ndarray, List[float]] = None,
                 discount_rates: Union[jnp.ndarray, np.ndarray, List[float]] = None,
                 dividend_rates: Union[jnp.ndarray, np.ndarray, List[float]] = None,
                 discount_factors: Union[jnp.ndarray, np.ndarray, List[float]] = None,
                 is_call_options: Union[jnp.ndarray, np.ndarray, List[bool]] = None,
                 is_normal_volatility: bool = False,
                 dtype: jnp.dtype = jnp.float64
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

    if (spots is None) == (forwards is None):
        raise ValueError('Either spots or forwards must be supplied but not both.')
    if (discount_rates is not None) and (discount_factors is not None):
        raise ValueError('At most one of discount_rates and discount_factors may '
                        'be supplied')
    
    dtype = dtype or jnp.float64
    strikes = jnp.asarray(strikes, dtype=dtype)
    volatilities = jnp.asarray(volatilities, dtype=dtype)
    expiries = jnp.asarray(expiries, dtype=dtype)

    if discount_rates is not None:
        discount_rates = jnp.asarray(discount_rates, dtype=dtype)
        discount_factors = jnp.exp(-discount_rates * expiries)    
    elif discount_factors is not None:
        discount_factors = jnp.asarray(discount_factors, dtype=dtype)
        discount_rates = -jnp.log(discount_factors) / expiries
    else:
        discount_rates = jnp.asarray(0.0, dtype=dtype)
        discount_factors = jnp.asarray(1.0, dtype=dtype)

    if dividend_rates is None:
        dividend_rates = jnp.asarray(0.0, dtype=dtype)

    # if forwards is None, spots must be supplied.
    if forwards is not None:
        forwards = jnp.asarray(forwards, dtype=dtype)
    else:
        spots = jnp.asarray(spots, dtype=dtype)
        forwards = spots * jnp.exp((discount_rates - dividend_rates) * expiries)

    sqrt_var = volatilities * jnp.sqrt(expiries)
    
    # lognormal model
    if not is_normal_volatility:
        d1 = divide_no_nan(jnp.log(forwards / strikes), sqrt_var) + sqrt_var / 2
        d2 = d1 - sqrt_var
        undiscounted_calls = jnp.where(sqrt_var > 0,
                                    forwards * _ncdf(d1) - strikes * _ncdf(d2),
                                    jnp.maximum(forwards - strikes, 0.0))
      # normal model
    else:
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

def barrier_price(*,
                  volatilities: Union[jnp.ndarray, np.ndarray, List[float]],
                  strikes: Union[jnp.ndarray, np.ndarray, List[float]],
                  expiries: Union[jnp.ndarray, np.ndarray, List[float]],
                  spots: Union[jnp.ndarray, np.ndarray, List[float]],
                  barriers: Union[jnp.ndarray, np.ndarray, List[float]],
                  rebates: Union[jnp.ndarray, np.ndarray, List[float]] = None,
                  discount_rates: Union[jnp.ndarray, np.ndarray, List[float]] = None,
                  dividend_rates: Union[jnp.ndarray, np.ndarray, List[float]] = None,
                  is_barrier_down: Union[jnp.ndarray, np.ndarray, List[float]] = None,
                  is_knock_out: Union[jnp.ndarray, np.ndarray, List[float]] = None,
                  is_call_options: Union[jnp.ndarray, np.ndarray, List[float]] = None,
                  dtype: jnp.dtype = jnp.float64
                  ) -> jnp.ndarray:
    """Prices barrier options in a Black-Scholes Model.

    Computes the prices of options with a single barrier in Black-Scholes world as
    described in Ref. [1]. Note that the barrier is applied continuously.

    #### Example

    This example is taken from Ref. [2], Page 154.

    ```python
    import tf_quant_finance as tff

    dtype = np.float32
    discount_rates = np.array([.08, .08])
    dividend_rates = np.array([.04, .04])
    spots = np.array([100., 100.])
    strikes = np.array([90., 90.])
    barriers = np.array([95. 95.])
    rebates = np.array([3. 3.])
    volatilities = np.array([.25, .25])
    expiries = np.array([.5, .5])
    barriers_type = np.array([5, 1])
    is_barrier_down = np.array([True, False])
    is_knock_out = np.array([False, False])
    is_call_option = np.array([True, True])

    price = tff.black_scholes.barrier_price(
    discount_rates, dividend_rates, spots, strikes,
    barriers, rebates, volatilities,
    expiries, is_barrier_down, is_knock_out, is_call_options)

    # Expected output
    #  `Tensor` with values [9.024, 7.7627]
    ```

    #### References

    [1]: Lee Clewlow, Javier Llanos, Chris Strickland, Caracas Venezuela
    Pricing Exotic Options in a Black-Scholes World, 1994
    https://warwick.ac.uk/fac/soc/wbs/subjects/finance/research/wpaperseries/1994/94-54.pdf
    [2]: Espen Gaarder Haug, The Complete Guide to Option Pricing Formulas,
    2nd Edition, 1997

    Args:
    volatilities: Real `Tensor` of any shape and dtype. The volatilities to
        expiry of the options to price.
    strikes: A real `Tensor` of the same dtype and compatible shape as
        `volatilities`. The strikes of the options to be priced.
    expiries: A real `Tensor` of same dtype and compatible shape as
        `volatilities`. The expiry of each option. The units should be such that
        `expiry * volatility**2` is dimensionless.
    spots: A real `Tensor` of any shape that broadcasts to the shape of the
        `volatilities`. The current spot price of the underlying.
    barriers: A real `Tensor` of same dtype as the `volatilities` and of the
        shape that broadcasts with `volatilities`. The barriers of each option.
    rebates: A real `Tensor` of same dtype as the `volatilities` and of the
        shape that broadcasts with `volatilities`. For knockouts, this is a
        fixed cash payout in case the barrier is breached. For knockins, this is a
        fixed cash payout in case the barrier level is not breached. In the former
        case, the rebate is paid immediately on breach whereas in the latter, the
        rebate is paid at the expiry of the option.
        Default value: `None` which maps to no rebates.
    discount_rates: A real `Tensor` of same dtype as the
        `volatilities` and of the shape that broadcasts with `volatilities`.
        Discount rates, or risk free rates.
        Default value: `None`, equivalent to discount_rate = 0.
    dividend_rates: A real `Tensor` of same dtype as the
        `volatilities` and of the shape that broadcasts with `volatilities`. A
        continuous dividend rate paid by the underlier. If `None`, then
        defaults to zero dividends.
        Default value: `None`, equivalent to zero dividends.
    is_barrier_down: A real `Tensor` of `boolean` values and of the shape
        that broadcasts with `volatilities`. True if barrier is below asset
        price at expiration.
        Default value: `True`.
    is_knock_out: A real `Tensor` of `boolean` values and of the shape
        that broadcasts with `volatilities`. True if option is knock out
        else false.
        Default value: `True`.
    is_call_options: A real `Tensor` of `boolean` values and of the shape
        that broadcasts with `volatilities`. True if option is call else
        false.
        Default value: `True`.
    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
        of any supplied non-`Tensor` arguments to `Tensor`.
        Default value: `None` which maps to the default dtype inferred by
        TensorFlow.
    name: str. The name for the ops created by this function.
        Default value: `None` which is mapped to the default name `barrier_price`.
    Returns:
    option_prices: A `Tensor` of same shape as `spots`. The approximate price of
    the barriers option under black scholes.
    """
    # The computation is done as in Ref [2] where each integral is split into
    # two matrices. The first matrix contains the algebraic terms and the second
    # matrix contains the probability distribution terms. Masks are used to filter
    # appropriate terms for calculating the integral. Then a dot product of each
    # row in the matricies coupled with the masks work to calculate the prices of
    # the barriers option.
    dtype = dtype or jnp.float64
    spots = jnp.asarray(spots, dtype=dtype)
    strikes = jnp.asarray(strikes, dtype=dtype)
    volatilities = jnp.asarray(volatilities, dtype=dtype)
    expiries = jnp.asarray(expiries, dtype=dtype)
    barriers = jnp.asarray(barriers, dtype=dtype)
    if rebates is not None:
        rebates = jnp.asarray(rebates, dtype=dtype)
    else:
        rebates = jnp.asarray(0.0, dtype=dtype)

    # Convert all to tensor and enforce float dtype where required
    if discount_rates is not None:
        discount_rates = jnp.asarray(
            discount_rates, dtype=dtype)
    else:
        discount_rates = jnp.asarray(
            0.0, dtype=dtype)

    if dividend_rates is not None:
        dividend_rates = jnp.asarray(dividend_rates, dtype=dtype)
    else:
        dividend_rates = jnp.asarray(0.0, dtype=jnp.float64)
    
    if is_barrier_down is None:
        is_barrier_down = jnp.bool_(True)
    else:
        is_barrier_down = jnp.asarray(is_barrier_down, dtype=dtype)
        is_barrier_down = jnp.where(is_barrier_down, 1, 0)
    if is_knock_out is None:
        is_knock_out = jnp.bool_(True)
    else:
        is_knock_out = jnp.asarray(is_knock_out, dtype=jnp.float64)
        is_knock_out = jnp.where(is_knock_out, 1, 0)
    if is_call_options is None:
        is_call_options = jnp.bool_(True)
    else:
        is_call_options = jnp.asarray(is_call_options, dtype=jnp.float64)
        is_call_options = jnp.where(is_call_options, 1, 0)

    
    # Indices which range from 0-7 are used to select the appropriate
    # mask for each barrier
    '''
    
    indices = jnp.bitwise.left_shift(
        is_barrier_down, 2) + jnp.bitwise.left_shift(
            is_knock_out, 1) + is_call_options #将此处变为乘积
    '''
    indices = jnp.multiply(is_barrier_down, 4) + jnp.multiply(is_knock_out, 2) + is_call_options 


    # Masks select the appropriate terms for integral approximations
    # Integrals are separated by algebraic terms and probability
    # distribution terms. This give 12 different terms per matrix
    # (6 integrals, 2 terms each)
    # shape = [8, 12]
    mask_matrix_greater_strike = jnp.asarray([ #jax中constant接口
        [1, 1, -1, -1, 0, 0, 1, 1, 1, 1, 0, 0],  # up and in put
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # up and in call
        [0, 0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1],  # up and out put
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # up and out call
        [0, 0, 1, 1, -1, -1, 1, 1, 0, 0, 1, 1],  # down and in put
        [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],  # down and in call
        [1, 1, -1, -1, 1, 1, -1, -1, 0, 0, 1, 1],  # down and out put
        [1, 1, 0, 0, -1, -1, 0, 0, 0, 0, 1, 1]],dtype=jnp.float64)  # down and out call

    mask_matrix_lower_strike = jnp.asarray([
        [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],  # up and in put
        [0, 0, 1, 1, -1, -1, 1, 1, 1, 1, 0, 0],  # up and in call
        [1, 1, 0, 0, -1, -1, 0, 0, 0, 0, 1, 1],  # up and out put
        [1, 1, -1, -1, 1, 1, -1, -1, 0, 0, 1, 1],  # up and out call
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # down and in put
        [1, 1, -1, -1, 0, 0, 1, 1, 1, 1, 0, 0],  # down and in call
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # down and out put
        [0, 0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1]],dtype=jnp.float64)  # down and out call

    # Create masks
    # Masks are shape [strikes.shape, 12]
    masks_lower = mask_matrix_lower_strike[indices,:] #gather API
    masks_greater = mask_matrix_greater_strike[indices,:]
    strikes_greater = jnp.expand_dims(strikes > barriers, axis=-1) #expand_dim API
    masks = jnp.where(strikes_greater, masks_greater, masks_lower)
    masks = jnp.asarray(masks, dtype=dtype)
    one = jnp.constant(1, dtype=jnp.float64)
    call_or_put = jnp.asarray(jnp.where(jnp.allclose(jnp.asarray(is_call_options), 0), -one, one),
                            dtype=jnp.float64)
    below_or_above = jnp.asarray(jnp.where(jnp.allclose(jnp.asarray(is_barrier_down), 0), -one, one),
                            dtype=jnp.float64)

    '''
    # Calculate params for integrals
    sqrt_var = volatilities * tf.math.sqrt(expiries)
    mu = (discount_rates - dividend_rates) - ((volatilities**2) / 2)
    lamda = 1 + (mu / (volatilities**2))
    x = (tf.math.log(spots / strikes) / (sqrt_var)) + (lamda * sqrt_var)
    x1 = (tf.math.log(spots / barriers) / (sqrt_var)) + (lamda * sqrt_var)
    y = (tf.math.log((barriers**2) / (spots * strikes)) / (
        sqrt_var)) + (lamda * sqrt_var)
    y1 = (tf.math.log(barriers / spots) / (sqrt_var)) + (lamda * sqrt_var)
    b = ((mu**2) + (2 * (volatilities**2) * discount_rates)) / (volatilities**2)
    z = (tf.math.log(barriers / spots) / (sqrt_var)) + (b * sqrt_var)
    a = mu / (volatilities**2)

    # Other params used for integrals
    discount_factors = tf.math.exp(
        -discount_rates * expiries, name='discount_factors')
    barriers_ratio = tf.math.divide(barriers, spots, name='barriers_ratio')
    spots_term = call_or_put * spots * tf.math.exp(-dividend_rates * expiries)
    strikes_term = call_or_put * strikes * discount_factors

    # rank is used to stack elements and reduce_sum
    strike_rank = strikes.shape.rank

    # Constructing Matrix with first and second algebraic terms for each
    # integral [strike.shape, 12]
    terms_mat = tf.stack(
        (spots_term, -strikes_term,
            spots_term, -strikes_term,
            spots_term * (barriers_ratio**(2 * lamda)),
            -strikes_term * (barriers_ratio**((2 * lamda) - 2)),
            spots_term * (barriers_ratio**(2 * lamda)),
            -strikes_term * (barriers_ratio**((2 * lamda) - 2)),
            rebates * discount_factors,
            -rebates * discount_factors * (  # pylint: disable=invalid-unary-operand-type
                barriers_ratio**((2 * lamda) - 2)),
            rebates * (barriers_ratio**(a + b)),
            rebates * (barriers_ratio**(a - b))),
        name='term_matrix', axis=strike_rank)

    # Constructing Matrix with first and second norm for each integral
    # [strikes.shape, 12]
    cdf_mat = tf.stack(
        (call_or_put * x,
            call_or_put * (x - sqrt_var),
            call_or_put * x1,
            call_or_put * (x1 - sqrt_var),
            below_or_above * y,
            below_or_above * (y - sqrt_var),
            below_or_above * y1,
            below_or_above * (y1 - sqrt_var),
            below_or_above * (x1 - sqrt_var),
            below_or_above * (y1 - sqrt_var),
            below_or_above * z,
            below_or_above * (z - (2 * b * sqrt_var))),
        name='cdf_matrix', axis=strike_rank)
    cdf_mat = _ncdf(cdf_mat)
    # Calculating and returning price for each option
    return tf.reduce_sum(masks * terms_mat * cdf_mat, axis=strike_rank)
    '''

def binary_price(*,
                 volatilities: Union[jnp.ndarray, np.ndarray, List[float]],
                 strikes: Union[jnp.ndarray, np.ndarray, List[float]],
                 expiries: Union[jnp.ndarray, np.ndarray, List[float]],
                 spots: Union[jnp.ndarray, np.ndarray, List[float]] = None,
                 forwards: Union[jnp.ndarray, np.ndarray, List[float]] = None,
                 discount_rates: Union[jnp.ndarray, np.ndarray, List[float]] = None,
                 dividend_rates: Union[jnp.ndarray, np.ndarray, List[float]] = None,
                 discount_factors: Union[jnp.ndarray, np.ndarray, List[float]] = None,
                 is_call_options: Union[jnp.ndarray, np.ndarray, List[bool]] = None,
                 is_normal_volatility: bool = False,
                 dtype: jnp.dtype = jnp.float64
                 ) -> jnp.ndarray:
    """Computes the Black Scholes price for a batch of binary call or put options.

    The binary call (resp. put) option priced here is that which pays off a unit
    of cash if the underlying asset has a value greater (resp. smaller) than the
    strike price at expiry. Hence the binary option price is the discounted
    probability that the asset will end up higher (resp. lower) than the
    strike price at expiry.

    #### Example

    ```python
    # Price a batch of 5 binary call options.
    volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])
    forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Strikes will automatically be broadcasted to shape [5].
    strikes = np.array([3.0])
    # Expiries will be broadcast to shape [5], i.e. each option has strike=3
    # and expiry = 1.
    expiries = 1.0
    computed_prices = tff.black_scholes.binary_price(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards)
    # Expected print output of prices:
    # [0.         0.         0.15865525 0.99764937 0.85927418]
    ```

    #### References:

    [1] Hull, John C., Options, Futures and Other Derivatives. Pearson, 2018.
    [2] Wikipedia contributors. Binary option. Available at:
    https://en.wikipedia.org/w/index.php?title=Binary_option

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
        discount_rates and discount_factors can be supplied.
        Default value: `None`, equivalent to r = 0 and discount factors = 1 when
        discount_factors also not given.
    dividend_rates: An optional real `Tensor` of same dtype as the
        `volatilities` and of the shape that broadcasts with `volatilities`.
        Default value: `None`, equivalent to q = 0.
    discount_factors: An optional real `Tensor` of same dtype as the
        `volatilities`. If not None, these are the discount factors to expiry
        (i.e. e^(-rT)). If None, no discounting is applied (i.e. the undiscounted
        option price is returned). If `spots` is supplied and `discount_factors`
        is not None then this is also used to compute the forwards to expiry.
        Default value: None, equivalent to discount factors = 1.
    is_call_options: A boolean `Tensor` of a shape compatible with
        `volatilities`. Indicates whether the option is a call (if True) or a put
        (if False). If not supplied, call options are assumed.
    is_normal_volatility: An optional Python boolean specifying whether the
        `volatilities` correspond to lognormal Black volatility (if False) or
        normal Black volatility (if True).
        Default value: False, which corresponds to lognormal volatility.
    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
        of any supplied non-`Tensor` arguments to `Tensor`.
        Default value: None which maps to the default dtype inferred by TensorFlow
        (float32).
    name: str. The name for the ops created by this function.
        Default value: None which is mapped to the default name `binary_price`.

    Returns:
    binary_prices: A `Tensor` of the same shape as `forwards`. The Black
    Scholes price of the binary options.

    Raises:
    ValueError: If both `forwards` and `spots` are supplied or if neither is
        supplied.
    ValueError: If both `discount_rates` and `discount_factors` is supplied.
    """
    if (spots is None) == (forwards is None):
        raise ValueError('Either spots or forwards must be supplied but not both.')
    if (discount_rates is not None) and (discount_factors is not None):
        raise ValueError('At most one of discount_rates and discount_factors may '
                        'be supplied')

    dtype = dtype or jnp.float64
    strikes = jnp.asarray(strikes, dtype=dtype)
    
    volatilities = jnp.asarray(volatilities, dtype=dtype)
    expiries = jnp.asarray(expiries, dtype=dtype)

    if discount_rates is not None:
        discount_rates = jnp.asarray(discount_rates, dtype=dtype)
        discount_factors = jnp.exp(-discount_rates * expiries)    
    elif discount_factors is not None:
        discount_factors = jnp.asarray(discount_factors, dtype=dtype)
        discount_rates = -jnp.log(discount_factors) / expiries
    else:
        discount_rates = jnp.asarray(0.0, dtype=dtype)
        discount_factors = jnp.asarray(1.0, dtype=dtype)

    if dividend_rates is None:
        dividend_rates = jnp.asarray(0.0, dtype=dtype)

    # if forwards is None, spots must be supplied.
    if forwards is not None:
        forwards = jnp.asarray(forwards, dtype=dtype)
    else:
        spots = jnp.asarray(spots, dtype=dtype)
        forwards = spots * jnp.exp((discount_rates - dividend_rates) * expiries)

    sqrt_var = volatilities * jnp.sqrt(expiries)

    if is_normal_volatility:  # normal model
        d2 = (forwards - strikes) / sqrt_var
    else:  # lognormal model
        d2 = jnp.log(forwards / strikes) / sqrt_var - sqrt_var / 2

    zero_volatility_call_payoff = jnp.where(forwards > strikes,
                                            jnp.ones_like(strikes, dtype=dtype),
                                            jnp.empty_like(strikes, dtype=dtype))
    undiscounted_calls = jnp.where(sqrt_var > 0, _ncdf(d2),
                                    zero_volatility_call_payoff)

    if is_call_options is None:
        return discount_factors * undiscounted_calls
    undiscounted_puts = 1 - undiscounted_calls
    predicate = jnp.broadcast_to(is_call_options, undiscounted_calls.shape)
    return discount_factors * jnp.where(predicate, undiscounted_calls,
                                        undiscounted_puts)