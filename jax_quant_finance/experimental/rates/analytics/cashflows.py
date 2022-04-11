import jax.numpy as jnp
from jax.lax import while_loop
from jax import jit
import numpy as np
from jax.ops import segment_sum

""""
TODO: ADD Error Control and document
"""

def present_value(cashflows, discount_factors, dtype=jnp.float64):
    dtype = dtype or cashflows.dtype
    
    cashflows = jnp.asarray(cashflows, dtype=dtype)
    discounted = cashflows * discount_factors
    return jnp.sum(discounted, axis=-1)


def pv_from_yields(cashflows, times, yields, dtype=jnp.float64):
    """
    Need to be fixed
    jax.lax.gather is a little hard to understand
    Jax tracer must be concrete in lax.while_loop ?
    _cond and _body only allow single 1 position value
    """
    cashflows = jnp.asarray(cashflows, dtype=dtype)
    times = jnp.asarray(times, dtype=dtype)
    yields = jnp.asarray(yields, dtype=dtype)

    cashflows_yields = yields
    
    discounted = cashflows * jnp.exp(-times * cashflows_yields)

    return jnp.sum(discounted)




def yields_from_pv(cashflows, times, present_values, groups=None, tolerance=1e-8, max_iterations=10, dtype=None, name=None, num_segments=None):
    """
    how to use jax to exec tf.gather(use numpy style slicing now)
    add argument num_segment to make segment_sum() `num_segments` argument concrete (other solutions?)
    """
    cashflows = jnp.asarray(cashflows, dtype=dtype)
    times = jnp.asarray(times, dtype=dtype)
    present_values = jnp.asarray(present_values, dtype=dtype)
    if groups is not None:
        groups = jnp.asarray(groups, dtype=jnp.int32)
    else:
        groups = jnp.zeros_like(cashflows, dtype=jnp.int32)
    num_segments = jnp.max(groups) + 1 

    def pv_and_duration(yields):
        # should use numpy style slicing instead of jax.lax.gather
        cashflows_yields = yields[groups,...]
        discounted = segment_sum(cashflows * jnp.exp(-times * cashflows_yields), groups, num_segments)
        durations = segment_sum(discounted * times, groups, num_segments)
        pvs = discounted
        return pvs, durations
    yields0 = jnp.zeros_like(present_values)


    # cond_func must be lambda
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

