import enum

import numpy as np
import jax.numpy as jnp
from jax import random


@enum.unique
class RandomType(enum.Enum):
    """Types of random number sequences.

    * `PSEUDO`: The standard MindSpore random generator.
    * `PSEUDO_ANTITHETIC`: PSEUDO random numbers along with antithetic variates.
    * `HALTON`: The standard Halton sequence.
    * `HALTON_RANDOMIZED`: The randomized Halton sequence.
    * `SOBOL`: The standard Sobol sequence.

    """

    PSEUDO = 0
    PSEUDO_ANTITHETIC = 1
    STATELESS = 2
    STATELESS_ANTITHETIC = 3
    HALTON = 4
    HALTON_RANDOMIZED = 5
    SOBOL = 6


def normal_pseudo(sample_shape,
                     mean,
                     key,
                     dtype=None):
    """Returns normal draws using the tfp multivariate normal distribution."""
    batch_shape = mean.shape
    sample_shape = tuple(sample_shape)
    output_shape = sample_shape + batch_shape
    key, subkey = random.split(key)
    samples = random.normal(key=subkey, shape=output_shape, dtype=dtype)
    mean = jnp.broadcast_to(mean, samples.shape)
    return mean + samples
    # else:
    #     return mean + tf.linalg.matvec(scale_matrix, samples)