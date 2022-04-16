import jax.numpy as jnp
import numpy as np

def diff(x, order=1, exclusive=False, axis=-1, dtype=None, name=None):
    x = jnp.asarray(x, dtype=dtype)

    #x0 = x[:-order:1]
    #x1 = x[order::1]
    x0 = jnp.take(x, indices=jnp.arange(0,x.shape[axis]-order,1), axis=axis)
    x1 = jnp.take(x, indices=jnp.arange(order,x.shape[axis],1), axis=axis)
    exclusive_diff = x1 - x0
    
    
    

    if exclusive:
        return exclusive_diff
    # slices[axis] = slice(None, order)
    return jnp.concatenate([jnp.take(x, indices=jnp.arange(0,order,1), axis=axis), exclusive_diff], axis=axis)