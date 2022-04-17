import jax.numpy as jnp
from jax_quant_finance.rates.analytics import cashflows
from jax_quant_finance.utils.ops import divide_no_nan


def swap_price(pay_leg_cashflows,
               receive_leg_cashflows,
               pay_leg_discount_factors,
               receive_leg_discount_factors,
               dtype=None):
    """Computes prices of a batch of generic swaps.

    #### Example
    ```python
    pay_leg_cashflows = [[100, 100, 100], [200, 250, 300]]
    receive_leg_cashflows = [[200, 250, 300, 300], [100, 100, 100, 100]]
    pay_leg_discount_factors = [[0.95, 0.9, 0.8],
                                [0.9, 0.85, 0.8]]
    receive_leg_discount_factors = [[0.95, 0.9, 0.8, 0.75],
                                    [0.9, 0.85, 0.8, 0.75]]
    swap_price(pay_leg_cashflows=pay_leg_cashflows,
                receive_leg_cashflows=receive_leg_cashflows,
                pay_leg_discount_factors=pay_leg_discount_factors,
                receive_leg_discount_factors=receive_leg_discount_factors,
                dtype=jnp.float64)
    # Expected: [615.0, -302.5]
    ```

    Args:
    pay_leg_cashflows: A real `Tensor` of shape
        `batch_shape + [num_pay_cashflows]`, where `num_pay_cashflows` is the
        number of cashflows for each batch element. Cashflows of the pay leg of
        the swaps.
    receive_leg_cashflows: A `Tensor` of the same `dtype` as `pay_leg_cashflows`
        and of shape `batch_shape + [num_receive_cashflows]` where
        `num_pay_cashflows` is the number of cashflows for each batch element.
        Cashflows of the receive leg of the swaps.
    pay_leg_discount_factors: A `Tensor` of the same `dtype` as
        `pay_leg_cashflows` and of compatible shape. Discount factors for each
        cashflow of the pay leg.
    receive_leg_discount_factors: A `Tensor` of the same `dtype` as
        `receive_leg_cashflows` and of compatible shape. Discount factors for each
        cashflow of the receive leg.
    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.
        Default value: None which maps to the default dtype inferred from
        `pay_leg_cashflows`.
    name: Python str. The name to give to the ops created by this function.
        Default value: None which maps to 'floating_coupons'.

    Returns:
    A `Tensor` of the same `dtype` as `coupon_rates` and of shape `batch_shape`.
    Present values of swaps from receiver perspective.
    """
    
    pay_leg_cashflows = jnp.asarray(pay_leg_cashflows, dtype=dtype)
    dtype = dtype or pay_leg_cashflows.dtype
    receive_leg_cashflows = jnp.asarray(receive_leg_cashflows, dtype= dtype)
    pay_leg_discount_factors = jnp.asarray(pay_leg_discount_factors, dtype= dtype)
    receive_leg_discount_factors = jnp.asarray(receive_leg_discount_factors, dtype= dtype)
    
    receive_leg_pv = cashflows.present_value(
        receive_leg_cashflows,
        receive_leg_discount_factors
    )
    
    pay_leg_pv = cashflows.present_value(
        pay_leg_cashflows,
        pay_leg_discount_factors
    )
    
    return receive_leg_pv - pay_leg_pv


def equity_leg_cashflows(
    forward_prices,
    spots,
    notional,
    dividends=None,
    dtype=None):
  """Computes cashflows for a batch of equity legs.

  Equity cashflows are defined as a total equity return between pay dates, say,
  `T_1, ..., T_n`. Let `S_i` represent the value of the equity at time `T_i` and
  `d_i` be a discrete dividend paid at this time. Then the the payment at time
  `T_i` is defined as `(S_i - S_{i - 1}) / S_{i-1} + d_i`. The value of
  the cashflow is then the discounted sum of the paments. See, e.g., [1] for the
  reference.

  #### Example
  ```python
  notional = 10000
  forward_prices = [[110, 120, 140], [210, 220, 240]]
  spots = [100, 200]
  dividends = [[1, 1, 1], [2, 2, 2]]
  equity_leg_cashflows(forward_prices, spots, notional, dividends,
                       dtype=tf.float64)
  # Expected:
  #  [[1000.01, 909.1, 1666.675],
  #   [ 500.01, 476.2, 909.1]]
  ```

  Args:
    forward_prices: A real `Tensor` of shape `batch_shape + [num_cashflows]`,
      where `num_cashflows` is the number of cashflows for each batch element.
      Equity forward prices at leg reset times.
    spots:  A `Tensor` of the same `dtype` as `forward_prices` and of
      shape compatible with `batch_shape`. Spot prices for each batch element
    notional: A `Tensor` of the same `dtype` as `forward_prices` and of
      compatible shape. Notional amount for each cashflow.
    dividends:  A `Tensor` of the same `dtype` as `forward_prices` and of
      compatible shape. Discrete dividends paid at the leg reset times.
      Default value: None which maps to zero dividend.
    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.
      Default value: None which maps to the default dtype inferred from
      `forward_prices`.
    name: Python str. The name to give to the ops created by this function.
      Default value: None which maps to 'equity_leg_cashflows'.

  Returns:
    A `Tensor` of the same `dtype` as `forward_prices` and of shape
    `batch_shape + [num_cashflows]`.

  #### References
  [1] Don M. Chance and Don R Rich,
    The Pricing of Equity Swaps and Swaptions, 1998
    https://jod.pm-research.com/content/5/4/19
  """
  
  forward_prices = jnp.asarray(forward_prices, dtype=dtype)
  dtype = dtype or forward_prices.dtype
  
  spots = jnp.asarray(spots, dtype=dtype)
  notional = jnp.asarray(notional, dtype=dtype)
  dividends = 0 if dividends is None else dividends
  dividends = jnp.asarray(dividends, dtype=dtype)
  spots_expand = jnp.expand_dims(spots, axis=-1)
  forward_prices = jnp.concatenate([spots_expand, forward_prices], axis=-1)
  return divide_no_nan(notional * (forward_prices[...,1:] - forward_prices[...,:-1])
                       +dividends,
                       forward_prices[...,:-1])


def rate_leg_cashflows(coupon_rates,notional,daycount_fractions,dtype=None):
  
    coupon_rates = jnp.asarray(coupon_rates, dtype=dtype)
    dtype = dtype or coupon_rates.dtype
    daycount_fractions = jnp.asarray(daycount_fractions, dtype=dtype)
    notional = jnp.asarray(notional, dtype=dtype)
    
    return notional * daycount_fractions * coupon_rates


def ir_swap_price(
    pay_leg_coupon_rates,
    receive_leg_coupon_rates,
    pay_leg_notional,
    receive_leg_notional,
    pay_leg_daycount_fractions,
    receive_leg_daycount_fractions,
    pay_leg_discount_factors,
    receive_leg_discount_factors,
    dtype=None):
    
    pay_leg_coupon_rates = jnp.asarray(pay_leg_coupon_rates, dtype=dtype)
    dtype = dtype or pay_leg_coupon_rates.dtype
    receive_leg_coupon_rates = jnp.asarray(receive_leg_coupon_rates, dtype=dtype)
    pay_leg_notional = jnp.asarray(pay_leg_notional, dtype=dtype)
    receive_leg_notional = jnp.asarray(receive_leg_notional, dtype=dtype)
    pay_leg_daycount_fractions = jnp.asarray(pay_leg_daycount_fractions, dtype=dtype)
    receive_leg_daycount_fractions = jnp.asarray(receive_leg_daycount_fractions, dtype = dtype)
    pay_leg_discount_factors = jnp.asarray(pay_leg_discount_factors, dtype=dtype)
    receive_leg_discount_factors = jnp.asarray(receive_leg_discount_factors, dtype=dtype)
    
    pay_leg_cashflows = rate_leg_cashflows(
      pay_leg_notional,
      pay_leg_daycount_fractions,
      pay_leg_coupon_rates
    )
    
    receive_leg_cashflows = rate_leg_cashflows(
      receive_leg_notional,
      receive_leg_daycount_fractions,
      receive_leg_coupon_rates
    )
    
    return swap_price(pay_leg_cashflows, 
                      receive_leg_cashflows, 
                      pay_leg_discount_factors, 
                      receive_leg_discount_factors)
  
  
def ir_swap_par_rate_and_annuity(floating_leg_start_times,
                                 floating_leg_end_times,
                                 fixed_leg_payment_times,
                                 fixed_leg_daycount_fractions,
                                 reference_rate_fn,
                                 dtype=None):
    """Utility function to compute par swap rate and annuity.

  Args:
    floating_leg_start_times: A real `Tensor` of the same dtype as `expiries`.
      The times when accrual begins for each payment in the floating leg. The
      shape of this input should be `expiries.shape + [m]` where `m` denotes the
      number of floating payments in each leg.
    floating_leg_end_times: A real `Tensor` of the same dtype as `expiries`. The
      times when accrual ends for each payment in the floating leg. The shape of
      this input should be `expiries.shape + [m]` where `m` denotes the number
      of floating payments in each leg.
    fixed_leg_payment_times: A real `Tensor` of the same dtype as `expiries`.
      The payment times for each payment in the fixed leg. The shape of this
      input should be `expiries.shape + [n]` where `n` denotes the number of
      fixed payments in each leg.
    fixed_leg_daycount_fractions: A real `Tensor` of the same dtype and
      compatible shape as `fixed_leg_payment_times`. The daycount fractions for
      each payment in the fixed leg.
    reference_rate_fn: A Python callable that accepts expiry time as a real
      `Tensor` and returns a `Tensor` of shape `input_shape + [dim]`. Returns
      the continuously compounded zero rate at the present time for the input
      expiry time.
    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.
      Default value: None which maps to the default dtype inferred from
        `floating_leg_start_times`.
    name: Python str. The name to give to the ops created by this function.
      Default value: None which maps to 'ir_swap_par_rate_and_annuity'.

  Returns:
    A tuple with two elements containing par swap rate and swap annuities.
  """
    floating_leg_start_times = jnp.asarray(floating_leg_start_times, dtype=dtype)
    dtype = dtype or floating_leg_start_times.dtype
    floating_leg_end_times = jnp.asarray(floating_leg_end_times, dtype=dtype)
    fixed_leg_payment_times = jnp.asarray(fixed_leg_payment_times, dtype=dtype)
    fixed_leg_daycount_fractions = jnp.asarray(fixed_leg_daycount_fractions, dtype=dtype)

    floating_leg_start_df = jnp.exp(
        -reference_rate_fn(floating_leg_start_times) * floating_leg_start_times)
    floating_leg_end_df = jnp.exp(
        -reference_rate_fn(floating_leg_end_times) * floating_leg_end_times)
    fixed_leg_payment_df = jnp.exp(
        -reference_rate_fn(fixed_leg_payment_times) * fixed_leg_payment_times)
    annuity = jnp.sum(
        fixed_leg_payment_df * fixed_leg_daycount_fractions, axis=-1)
    swap_rate = jnp.sum(
        floating_leg_start_df - floating_leg_end_df, axis=-1) / annuity
    
    return swap_rate, annuity
  
  
def equity_swap_price(
      rate_leg_coupon_rates,
      equity_leg_forward_prices,
      equity_leg_spots,
      rate_leg_notional,
      equity_leg_notional,
      rate_leg_daycount_fractions,
      rate_leg_discount_factors,
      equity_leg_discount_factors,
      equity_dividends=None,
      is_equity_receiver=None,
      dtype=None):
    """Computes prices of a batch of equity swaps.

  The swap consists of an equity and interest rate legs.

  #### Example
  ```python
  rate_leg_coupon_rates = [[0.1, 0.2, 0.05], [0.1, 0.05, 0.2]]
  # Two cashflows of 4 and 3 payments, respectively
  forward_prices = [[110, 120, 140, 150], [210, 220, 240, 0]]
  spots = [100, 200]
  notional = 1000
  pay_leg_daycount_fractions = 0.5
  rate_leg_daycount_fractions = [[0.5, 0.5, 0.5], [0.4, 0.5, 0.6]]
  rate_leg_discount_factors = [[0.95, 0.9, 0.85], [0.98, 0.92, 0.88]]
  equity_leg_discount_factors = [[0.95, 0.9, 0.85, 0.8],
                                 [0.98, 0.92, 0.88, 0.0]]

  equity_swap_price(
      rate_leg_coupon_rates=rate_leg_coupon_rates,
      equity_leg_forward_prices=forward_prices,
      equity_leg_spots=spots,
      rate_leg_notional=notional,
      equity_leg_notional=notional,
      rate_leg_daycount_fractions=rate_leg_daycount_fractions,
      rate_leg_discount_factors=rate_leg_discount_factors,
      equity_leg_discount_factors=equity_leg_discount_factors,
      is_equity_receiver=[True, False],
      dtype=tf.float64)
  # Expected: [216.87770563, -5.00952381]
  forward_rates(df_start_dates, df_end_dates, daycount_fractions,
                dtype=tf.float64)
  ```

  Args:
    rate_leg_coupon_rates: A real `Tensor` of shape
      `batch_shape + [num_rate_cashflows]`, where `num_rate_cashflows` is the
      number of cashflows for each batch element. Coupon rates for the
      interest rate leg.
    equity_leg_forward_prices: A `Tensor` of the same `dtype` as
      `rate_leg_coupon_rates` and of shape
      `batch_shape + [num_equity_cashflows]`, where `num_equity_cashflows` is
      the number of cashflows for each batch element. Equity leg forward
      prices.
    equity_leg_spots: A `Tensor` of the same `dtype` as
      `equity_leg_forward_prices` and of shape compatible with `batch_shape`.
      Spot prices for each batch element of the equity leg.
    rate_leg_notional: A `Tensor` of the same `dtype` as `rate_leg_coupon_rates`
      and of compatible shape. Notional amount for each cashflow.
    equity_leg_notional: A `Tensor` of the same `dtype` as
      `equity_leg_forward_prices` and of compatible shape.  Notional amount for
      each cashflow.
    rate_leg_daycount_fractions: A `Tensor` of the same `dtype` as
      `rate_leg_coupon_rates` and of compatible shape.  Year fractions for the
      coupon accrual.
    rate_leg_discount_factors: A `Tensor` of the same `dtype` as
      `rate_leg_coupon_rates` and of compatible shape. Discount factors for each
      cashflow of the rate leg.
    equity_leg_discount_factors: A `Tensor` of the same `dtype` as
      `equity_leg_forward_prices` and of compatible shape. Discount factors for
      each cashflow of the equity leg.
    equity_dividends: A `Tensor` of the same `dtype` as
      `equity_leg_forward_prices` and of compatible shape. Dividends paid at the
      leg reset times.
      Default value: None which maps to zero dividend.
    is_equity_receiver: A boolean `Tensor` of shape compatible with
      `batch_shape`. Indicates whether the swap holder is equity holder or
      receiver.
      Default value: None which means that all swaps are equity reiver swaps.
    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.
      Default value: None which maps to the default dtype inferred from
      `rate_leg_coupon_rates`.
    name: Python str. The name to give to the ops created by this function.
      Default value: None which maps to 'equity_swap_price'.

  Returns:
    A `Tensor` of the same `dtype` as `rate_leg_coupon_rates` and of shape
    `batch_shape`. Present values of the equity swaps.
  """
    rate_leg_coupon_rates = jnp.asarray(rate_leg_coupon_rates, dtype=dtype)
    dtype = dtype or rate_leg_coupon_rates.dtype
    equity_leg_forward_prices = jnp.asarray(equity_leg_forward_prices, dtype=dtype)
    equity_leg_spots = jnp.asarray(equity_leg_spots, dtype=dtype)
    rate_leg_daycount_fractions = jnp.asarray(rate_leg_daycount_fractions, dtype=dtype,)
    equity_dividends = equity_dividends or 0
    equity_dividends = jnp.asarray(equity_dividends, dtype=dtype,)
    rate_leg_notional = jnp.asarray(rate_leg_notional, dtype=dtype)
    equity_leg_notional = jnp.asarray(equity_leg_notional, dtype=dtype)
    rate_leg_discount_factors = jnp.asarray(rate_leg_discount_factors, dtype=dtype)
    equity_leg_discount_factors = jnp.asarray(equity_leg_discount_factors, dtype=dtype)
    if is_equity_receiver is None:
      is_equity_receiver = True
    is_equity_receiver = jnp.asarray(is_equity_receiver, dtype=jnp.bool_)
    one = jnp.ones([], dtype=dtype)
    equity_receiver = jnp.where(is_equity_receiver, one, -one)
    equity_cashflows = equity_leg_cashflows(
        forward_prices=equity_leg_forward_prices,
        spots=equity_leg_spots,
        notional=equity_leg_notional,
        dividends=equity_dividends)
    rate_cashflows = rate_leg_cashflows(
        coupon_rates=rate_leg_coupon_rates,
        notional=rate_leg_notional,
        daycount_fractions=rate_leg_daycount_fractions)
    return equity_receiver * swap_price(
        rate_cashflows,
        equity_cashflows,
        rate_leg_discount_factors,
        equity_leg_discount_factors)
  


    

  
    
