import jax.numpy as jnp

def diff(x, order=1, exclusive=False, axis=-1, dtype=None, name=None):
    x = jnp.asarray(x, dtype=dtype)

    # jnp no rank attribute for shape
    #slices = x.shape[0] * [slice(None)]
    #slices[axis] = slice(None, -order)

    # after slice it's not jnp but pure list
    x0 = x[:-order:1]
    x1 = x[order::1]
    exclusive_diff = x1 - x0
    exclusive_diff = jnp.asarray(exclusive_diff, dtype=dtype)

    if exclusive:
        return exclusive_diff
    # slices[axis] = slice(None, order)
    return jnp.concatenate([jnp.asarray(x[0:order], dtype=dtype), exclusive_diff], axis=axis)