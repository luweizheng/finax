import jax.numpy as jnp
from jax import config
from jax import random
config.update("jax_enable_x64", True)
from finax.math import random_sampler
from finax.experimental.heston import utils


num_draws = jnp.asarray(2, dtype=jnp.int32)
steps_num = jnp.asarray(3, dtype=jnp.int32)
num_samples = jnp.asarray(4, dtype=jnp.int32)
random_type = random_sampler.RandomType.STATELESS
key = random.PRNGKey(0)
samples = utils.generate_mc_normal_draws(num_draws, steps_num, num_samples,key=key,random_type=random_type, dtype=jnp.float64)
print(samples)

