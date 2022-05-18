from time import time
from jax_quant_finance.experimental.pde.steppers.explicit import explicit_step
import numpy as np
import jax_quant_finance as jqf
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)


fd_solvers = jqf.experimental.pde.fd_solvers
dirichlet = jqf.experimental.pde.boundary_conditions.dirichlet
neumann = jqf.experimental.pde.boundary_conditions.neumann
grids = jqf.experimental.pde.grids
explicit_step = jqf.experimental.pde.steppers.explicit.explicit_step

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

time_step = 0.01

print(f"Grid is constructed")

def testHeatEquation(grid, 
                     final_t, 
                     time_step, 
                     final_cond_fn, 
                     expected_result_fn, 
                     one_step_fn, 
                     lower_boundary_fn, 
                     upper_boundary_fn):
    
    def second_order_coeff_fn(t, x):
        del t, x
        return [[1]]
    
    xs = grid[0]
    
    final_values = jnp.asarray([final_cond_fn(x) for x in xs], dtype=grid[0].dtype)
    
    result = fd_solvers.solve_backward(
        start_time=final_t,
        end_time=0,
        coord_grid=grid,
        values_grid=final_values,
        num_steps=None,
        start_step_count=0,
        time_step=time_step,
        one_step_fn=one_step_fn,
        boundary_conditions=[(lower_boundary_fn, upper_boundary_fn)],
        values_transform_fn=None,
        second_order_coeff_fn=second_order_coeff_fn,
        dtype=grid[0].dtype
    )
    
    actual = result[0]
    
    expected = expected_result_fn(xs)
    
    print(f"Actual Result: {actual}")
    
    print(f"Expected Result: {expected}")
    
one_step_fn = explicit_step()
testHeatEquation(grid=grid, final_t=1, time_step=time_step, final_cond_fn=final_cond_fn, expected_result_fn=expected_result_fn, one_step_fn=one_step_fn, lower_boundary_fn=lower_boundary_fn, upper_boundary_fn=upper_boundary_fn)
    
    
    



