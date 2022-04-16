import jax.numpy as jnp

def simpson(func, lower, upper, num_points=1001, dtype=None, name=None):
    """Evaluates definite integral using composite Simpson's 1/3 rule.

    Integrates `func` using composite Simpson's 1/3 rule [1].

    Evaluates function at points of evenly spaced grid of `num_points` points,
    then uses obtained values to interpolate `func` with quadratic polynomials
    and integrates these polynomials.

    #### References
    [1] Weisstein, Eric W. "Simpson's Rule." From MathWorld - A Wolfram Web
        Resource. http://mathworld.wolfram.com/SimpsonsRule.html

    #### Example
    ```python
        f = lambda x: x*x
        a = tf.constant(0.0)
        b = tf.constant(3.0)
        integrate(f, a, b, num_points=1001) # 9.0
    ```

    Args:
        func: Python callable representing a function to be integrated. It must be a
        callable of a single `Tensor` parameter and return a `Tensor` of the same
        shape and dtype as its input. It will be called with a `Tesnor` of shape
        `lower.shape + [n]` (where n is integer number of points) and of the same
        `dtype` as `lower`.
        lower: `Tensor` or Python float representing the lower limits of
        integration. `func` will be integrated between each pair of points defined
        by `lower` and `upper`.
        upper: `Tensor` of the same shape and dtype as `lower` or Python float
        representing the upper limits of intergation.
        num_points: Scalar int32 `Tensor`. Number of points at which function `func`
        will be evaluated. Must be odd and at least 3.
        Default value: 1001.
        dtype: Optional `tf.Dtype`. If supplied, the dtype for the `lower` and
        `upper`. Result will have the same dtype.
        Default value: None which maps to dtype of `lower`.
        name: Python str. The name to give to the ops created by this function.
        Default value: None which maps to 'integrate_simpson_composite'.

    Returns:
        `Tensor` of shape `func_batch_shape + limits_batch_shape`, containing
        value of the definite integral.
        
    TODO: Add Error control

    """
    lower = jnp.asarray(lower, dtype=dtype)
    dtype = lower.dtype
    upper = jnp.asarray(upper, dtype=dtype)
    num_points = jnp.asarray(num_points, dtype=jnp.int32)

    dx = (upper-lower) /(jnp.asarray(num_points, dtype=dtype)-1)
    dx_expand = jnp.expand_dims(dx, -1)
    lower_exp = jnp.expand_dims(lower, -1)
    grid = lower_exp + dx_expand * jnp.arange(num_points, dtype=dtype)
    weights_first = jnp.asarray([1.0], dtype=dtype)
    weights_mid = jnp.tile(jnp.asarray([4.0, 2.0], dtype=dtype), [(num_points - 3) // 2])
    weights_last = jnp.asarray([4.0, 1.0], dtype=dtype)
    weights = jnp.concatenate([weights_first, weights_mid, weights_last], axis=0)

    return jnp.sum(func(grid) * weights, axis=-1) * dx/3
  
