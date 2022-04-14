import jax.numpy as jnp

def scatter_nd(indices, updates, shape):
    zeros = jnp.zeros(shape, updates.dtype)
    key = tuple(jnp.moveaxis(indices, -1, 0))
    return zeros.at[key].add(updates)