from finax.experimental.math.pde.grids import uniform_grid
import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)



dtype = np.float64
min_x, max_x, sizes = [0.1], [3.0], [5]
grid = uniform_grid(min_x, max_x, sizes, dtype=dtype)


print(grid)

