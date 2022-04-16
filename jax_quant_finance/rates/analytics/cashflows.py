import jax.numpy as jnp
from jax.lax import while_loop
from jax.ops import segment_sum



def present_value(cashflows, discount_factors, dtype=jnp.float64):
    """Computes present value of a stream of cashflows given discount factors.


    ```python

    # 2 and 3 year bonds with 1000 face value and 4%, 6% semi-annual coupons.
    # Note that the first four entries in the cashflows are the cashflows of
    # the first bond (group=0) and the next six are the cashflows of the second
    # bond (group=1).
    cashflows = [[20, 20, 20, 1020, 0, 0],
                    [30, 30, 30, 30, 30, 1030]]

    # Corresponding discount factors for the cashflows
    discount_factors = [[0.96, 0.93, 0.9, 0.87, 1.0, 1.0],
                        [0.97, 0.95, 0.93, 0.9, 0.88, 0.86]]

    present_values = present_value(
        cashflows, discount_factors, dtype=np.float64)
    # Expected: [943.2, 1024.7]
    ```

    Args:
    cashflows: A real `Tensor` of shape `batch_shape + [n]`. The set of
        cashflows of underlyings. `n` is the number of cashflows per bond
        and `batch_shape` is the number of bonds. Bonds with different number
        of cashflows should be padded to a common number `n`.
    discount_factors: A `Tensor` of the same `dtype` as `cashflows` and of
        compatible shape. The set of discount factors corresponding to the
        cashflows.
    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.
        Default value: None which maps to the default dtype inferred from
        `cashflows`.
    name: Python str. The name to give to the ops created by this function.
        Default value: None which maps to 'present_value'.

    Returns:
    Real `Tensor` of shape `batch_shape`. The present values of the cashflows.
    """
    cashflows = jnp.asarray(cashflows, dtype=dtype)
    dtype = dtype or cashflows.dtype
    discount_factors = jnp.asarray(discount_factors, dtype=dtype)
    
    discounted = cashflows * discount_factors
    return jnp.sum(discounted, axis=-1)


def pv_from_yields(cashflows,
                   times,
                   yields,
                   groups=None,
                   dtype=None):
    """Computes present value of cashflows given yields.

    For a more complete description of the terminology as well as the mathematics
    of pricing bonds, see Ref [1]. In particular, note that `yields` here refers
    to the yield of the bond as defined in Section 4.4 of Ref [1]. This is
    sometimes also referred to as the internal rate of return of a bond.

    #### Example

    The following example demonstrates the present value computation for two
    bonds. Both bonds have 1000 face value with semi-annual coupons. The first
    bond has 4% coupon rate and 2 year expiry. The second has 6% coupon rate and
    3 year expiry. The yields to maturity (ytm) are 7% and 5% respectively.

    ```python
    dtype = np.float64

    # The first element is the ytm of the first bond and the second is the
    # yield of the second bond.
    yields_to_maturity = np.array([0.07, 0.05], dtype=dtype)

    # 2 and 3 year bonds with 1000 face value and 4%, 6% semi-annual coupons.
    # Note that the first four entries in the cashflows are the cashflows of
    # the first bond (group=0) and the next six are the cashflows of the second
    # bond (group=1).
    cashflows = np.array([20, 20, 20, 1020, 30, 30, 30, 30, 30, 1030],
                            dtype=dtype)

    # The times of the cashflows.
    times = np.array([0.5, 1, 1.5, 2, 0.5, 1, 1.50, 2, 2.5, 3], dtype=dtype)

    # Group entries take values between 0 and 1 (inclusive) as there are two
    # bonds. One needs to assign each of the cashflow entries to one group or
    # the other.
    groups = np.array([0] * 4 + [1] * 6)

    # Produces [942.712, 1025.778] as the values of the two bonds.
    present_values = pv_from_yields(
        cashflows, times, yields_to_maturity, groups=groups, dtype=dtype)
    ```

    #### References:

    [1]: John C. Hull. Options, Futures and Other Derivatives. Ninth Edition.
    June 2006.

    Args:
    cashflows: Real rank 1 `Tensor` of size `n`. The set of cashflows underlying
        the bonds.
    times: Real positive rank 1 `Tensor` of size `n`. The set of times at which
        the corresponding cashflows occur quoted in years.
    yields: Real rank 1 `Tensor` of size `1` if `groups` is None or of size `k`
        if the maximum value in the `groups` is of `k-1`. The continuously
        compounded yields to maturity/internal rate of returns corresponding to
        each of the cashflow groups. The `i`th component is the yield to apply to
        all the cashflows with group label `i` if `groups` is not None. If
        `groups` is None, then this is a `Tensor` of size `[1]` and the only
        component is the yield that applies to all the cashflows.
    groups: Optional int `Tensor` of size `n` containing values between 0 and
        `k-1` where `k` is the number of related cashflows.
        Default value: None. This implies that all the cashflows are treated as a
        single group.
    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.
        Default value: None which maps to the default dtype inferred from
        `cashflows`.
    name: Python str. The name to give to the ops created by this function.
        Default value: None which maps to 'pv_from_yields'.

    Returns:
    Real rank 1 `Tensor` of size `k` if groups is not `None` else of size `[1]`.
        The present value of the cashflows. The `i`th component is the present
        value of the cashflows in group `i` or to the entirety of the cashflows
        if `groups` is None.
    """
    cashflows = jnp.asarray(cashflows, dtype=dtype)
    
    dtype = dtype or cashflows.dtype
    
    times = jnp.asarray(times, dtype=dtype)
    yields = jnp.asarray(yields, dtype=dtype)
    
    if groups is not None:
        groups = jnp.asarray(groups, dtype=jnp.int32)
        cashflows_yields = yields[groups,...]
    else:
        cashflows_yields = yields


    discounted = cashflows * jnp.exp(-times * cashflows_yields)
    if groups is not None:
        return segment_sum(discounted, groups)
    
    return jnp.sum(discounted, keepdims=True)




def yields_from_pv(cashflows, times, present_values, groups=None, tolerance=1e-8, dtype=None, num_segments=None):

    cashflows = jnp.asarray(cashflows, dtype=dtype)
    times = jnp.asarray(times, dtype=dtype)
    present_values = jnp.asarray(present_values, dtype=dtype)
    if groups is not None:
        groups = jnp.asarray(groups, dtype=jnp.int32)
    else:
        groups = jnp.zeros_like(cashflows, dtype=jnp.int32)
    num_segments = jnp.max(groups) + 1 

    def pv_and_duration(yields):
        cashflows_yields = yields[groups,...]
        # discounted = segment_sum(cashflows * jnp.exp(-times * cashflows_yields), groups, num_segments)
        discounted = cashflows * jnp.exp(-times * cashflows_yields)
        durations = segment_sum(discounted * times, groups, num_segments)
        pvs = segment_sum(discounted, groups, num_segments)
        return pvs, durations
    yields0 = jnp.zeros_like(present_values)


    # cond_func with condition control must be lambda
    _cond = lambda vals : jnp.logical_not(vals[0])


    # _body 
    def _body(vals):
        pvs, durations = pv_and_duration(vals[1])
        delta_yields = (pvs - present_values) / durations
        next_should_stop = (jnp.max(jnp.abs(delta_yields)) <= tolerance)
        return [next_should_stop, vals[1] + delta_yields]       

    loop_vars = [jnp.asarray(False), yields0]

    _, estimated_yields = while_loop( _cond,_body, loop_vars)

    return estimated_yields

__all__ = ['present_value', 'pv_from_yields', 'yields_from_pv']

