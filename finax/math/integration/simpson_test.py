import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

from finax.math.integration.simpson import simpson

import numpy as np

#TODO add tests

func = lambda x: jnp.exp(2 * x +1)
lower=jnp.asarray([1.0,2.0])
upper=jnp.asarray([3.0,4.0])
dtype = jnp.float64

antiderivate_func = lambda x: jnp.exp(2*x+1)/2

sim = simpson(np.sin, 0, 1, num_points=10001 ,dtype=dtype)


print(sim)



