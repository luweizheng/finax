import jax.numpy as jnp
from numpy import int32


def uniform_grid(minimums, maximums, sizes, dtype=None, validate_args=False):
    minimums = jnp.asarray(minimums, dtype=dtype)
    dtype = minimums.dtype or None
    maximums = jnp.asarray(maximums, dtype=dtype)
    sizes = jnp.asarray(sizes, dtype=jnp.int32)
    
    
    dim = sizes.shape[0]
    locations = [
        jnp.linspace(minimums[i], maximums[i], num=sizes[i])
        for i in range(dim)
    ]
    
    return locations
    



__all__ = [
    'uniform_grid',
]