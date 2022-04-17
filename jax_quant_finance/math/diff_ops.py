import jax.numpy as jnp

def diff(x, order=1, exclusive=False, axis=-1, dtype=None):
    """Computes the difference between elements of an array at a regular interval.

    For a difference along the final axis, if exclusive is True, then computes:

    ```
        result[..., i] = x[..., i+order] - x[..., i] for i < size(x) - order

    ```

    This is the same as doing `x[..., order:] - x[..., :-order]`. Note that in
    this case the result `Tensor` is smaller in size than the input `Tensor`.

    If exclusive is False, then computes:

    ```
        result[..., i] = x[..., i] - x[..., i-order] for i >= order
        result[..., i] = x[..., i]  for 0 <= i < order

    ```
    """
    
    x = jnp.asarray(x, dtype=dtype)

    x0 = jnp.take(x, indices=jnp.arange(0,x.shape[axis]-order,1), axis=axis)
    x1 = jnp.take(x, indices=jnp.arange(order,x.shape[axis],1), axis=axis)
    exclusive_diff = x1 - x0
    
    if exclusive:
        return exclusive_diff
    return jnp.concatenate(
        [jnp.take(x, indices=jnp.arange(0, order, 1), axis=axis), exclusive_diff], 
        axis=axis)