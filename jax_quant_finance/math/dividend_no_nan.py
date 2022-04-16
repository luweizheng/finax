import jax.numpy as jnp

def divide_no_nan(dividend, divisor, dtype=None):
    dividend = jnp.asarray(dividend, dtype=dtype)
    dtype = dtype or dividend.dtype
    
    divisor = jnp.asarray(divisor, dtype=dtype)
    
    _divisor = jnp.where(divisor==0.0, 1.0, divisor)
    
    res = jnp.where(divisor==0.0, 1.0, dividend/_divisor)
    
    return res