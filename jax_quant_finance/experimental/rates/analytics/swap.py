import jax.numpy as jnp
from jax_quant_finance.experimental.rates.analytics import cashflows


def swap_price(pay_leg_cashflows,
               receive_leg_cashflows,
               pay_leg_disocunt_factors,
               receive_leg_discount_factors,
               dtype=None,
               name=None):
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
                dtype=tf.float64)
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
    pay_leg_disocunt_factors = jnp.asarray(pay_leg_disocunt_factors, dtype= dtype)
    receive_leg_disocunt_factors = jnp.asarray(receive_leg_disocunt_factors, dtype= dtype)
    
    receive_leg_pv = cashflows.present_value(
        receive_leg_cashflows,
        receive_leg_discount_factors
    )
    
    pay_leg_pv = cashflows.present_value(
        pay_leg_cashflows,
        pay_leg_disocunt_factors
    )
    
    return receive_leg_pv - pay_leg_pv


def equity_leg_cashflows(
    forward_prices,
    spots,
    notional,
    dividends=None,
    dtype=None,
    name=None):
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
  pass
"""
TODO : dividend_no_nan?
"""
    

  
    
