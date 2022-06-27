import finax
from jax import config
import jax.numpy as jnp

config.update("jax_enable_x64", True)

fd_solvers = finax.experimental.pde.fd_solvers
dirichlet = finax.experimental.pde.boundary_conditions.dirichlet
neumann = finax.experimental.pde.boundary_conditions.neumann
grids = finax.experimental.pde.grids
explicit_step = finax.experimental.pde.steppers.explicit.explicit_step


# Heat Equation
def final_cond_fn(x):
    return jnp.e * jnp.sin(x)

def expected_result_fn(x):
    return jnp.sin(x)

@dirichlet
def lower_boundary_fn(t, x):
    del x
    return -jnp.exp(t)

@dirichlet
def upper_boundary_fn(t,x):
    del x
    return jnp.exp(t)


grid = grids.uniform_grid(
    minimums=[-10.5 * jnp.pi],
    maximums=[10.5 * jnp.pi],
    sizes = [100],
    dtype=jnp.float64
)

def second_order_coeff_fn(t, x):
    del t, x
    return [[1]]

xs = grid[0]
final_values = jnp.asarray([final_cond_fn(x) for x in xs], dtype=grid[0].dtype)

# print(final_values)


(a, b) = explicit_step()(1, 0.99, grid, final_values, [(lower_boundary_fn, upper_boundary_fn)], second_order_coeff_fn=second_order_coeff_fn, first_order_coeff_fn=None, zeroth_order_coeff_fn=None, inner_second_order_coeff_fn=None, inner_first_order_coeff_fn=None, num_steps_performed=None,dtype=grid[0].dtype)

print(b)