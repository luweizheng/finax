import numpy as np
import jax_quant_finance as jqf
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

dirichlet = jqf.experimental.pde.boundary_conditions.dirichlet

print(dirichlet)