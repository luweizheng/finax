import jax.numpy as jnp

def scatter_nd(indices, updates, shape):
    zeros = jnp.zeros(shape, updates.dtype)
    key = tuple(jnp.moveaxis(indices, -1, 0))
    return zeros.at[key].add(updates)


#def divide_no_nan(x, y):
#    """Computes a safe divide which returns 0 if `y` (denominator) is zero."""
#    x = jnp.asarray(x)
#    y = jnp.asarray(y)

#    _y = jnp.where(jnp.isclose(0.0, y), 0.0, y)
#    res = jnp.where(jnp.isclose(0.0, y), 0.0, x/_y)
#    return res


def divide_no_nan(x, y):
    """Computes a safe divide which returns 0 if `y` (denominator) is zero."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    return jnp.where(~jnp.isclose(0.0, y), x / y, 0)