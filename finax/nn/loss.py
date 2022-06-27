"""Methods for commonly used loss functions."""

import jax.numpy as jnp
from jax import jit

@jit
def sum_square_error(y, y_pred):
    """Computes the sum of square error."""
    return jnp.sum(jnp.square(y - y_pred))


@jit
def mean_square_error(y, y_pred):
    """Computes the mean of square error."""
    return jnp.mean(jnp.square(y - y_pred))