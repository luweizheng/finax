import numpy as np
from finax.experimental.heston.heston_model import HestonModel
from finax.experimental.math.pde import grids
from jax import config
config.update("jax_enable_x64", True)
config.update("jax_disable_jit", True)
from jax import random
import jax.numpy as jnp

key = random.PRNGKey(0)


theta = 0.5
process = HestonModel(mean_reversion=1.0, theta=theta, volvol=1.0, rho=-0.0, dtype=jnp.float64)
years = 1.0
times = np.linspace(0.0, years, int(30 * years))
num_samples = 2

paths = process.sample_paths(
    times,
    num_samples=num_samples,
    initial_state=np.array([np.log(100), 0.45]),
    key = key
)


    
