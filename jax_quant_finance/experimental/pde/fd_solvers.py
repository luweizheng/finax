import jax.numpy as jnp
from jax_quant_finance.experimental.pde.steppers.explicit import explicit_step
from jax.lax import while_loop

def solve_backward(start_time,
                   end_time,
                   coord_grid,
                   values_grid,
                   num_steps=None,
                   start_step_count=0,
                   time_step=None,
                   one_step_fn=None,
                   boundary_conditions=None,
                   values_transform_fn=None,
                   second_order_coeff_fn=None,
                   first_order_coeff_fn=None,
                   zeroth_order_coeff_fn=None,
                   inner_second_order_coeff_fn=None,
                   inner_first_order_coeff_fn=None,
                   maximum_steps=None,
                   dtype=None):
    
    values_grid = jnp.asarray(values_grid, dtype=dtype)
    dtype = dtype or values_grid.dtype
    
    start_time = jnp.asarray(start_time ,dtype=dtype)
    
    end_time = jnp.maximum(jnp.minimum(jnp.asarray(end_time, dtype=dtype), start_time),0)
    
    
    
    
    return _solve(
        _time_direction_backward_fn,
        start_time,
        end_time,
        coord_grid,
        values_grid,
        num_steps,
        start_step_count,
        time_step,
        one_step_fn,
        boundary_conditions,
        values_transform_fn,
        second_order_coeff_fn,
        first_order_coeff_fn,
        zeroth_order_coeff_fn,
        inner_second_order_coeff_fn,
        inner_first_order_coeff_fn,
        maximum_steps
    )



def _solve(
    time_direction_fn,
    start_time,
    end_time,
    coord_grid,
    values_grid,
    num_steps=None,
    start_step_count=0,
    time_step=None,
    one_step_fn=None,
    boundary_conditions=None,
    values_transform_fn=None,
    second_order_coeff_fn=None,
    first_order_coeff_fn=None,
    zeroth_order_coeff_fn=None,
    inner_second_order_coeff_fn=None,
    inner_first_order_coeff_fn=None,
    maximum_steps=None):
    
    
    if (num_steps is None) == (time_step is None):
        raise ValueError('Exactly one of num_steps or time_step'
                     ' should be supplied.')
        
    coord_grid = [jnp.asarray(dim_grid, dtype=values_grid.dtype) for dim_grid in coord_grid]
    
    n_dims = len(coord_grid)
    if one_step_fn is None:
        if n_dims == 1:
            one_step_fn = explicit_step()
        else:
            raise ValueError("Only Support dim 1 now")
        
    
    if boundary_conditions is None:
        
        def zero_dirichlet(t, grid):
            del t, grid
            return 1, None, jnp.asarray(0, dtype=values_grid.dtype)
        

        boundary_conditions = [(zero_dirichlet, zero_dirichlet)] * n_dims

    
    time_step_fn, est_max_steps = _get_time_steps_info(start_time, end_time, num_steps, time_step, time_direction_fn)

    if est_max_steps is None and maximum_steps is not None:
        est_max_steps = maximum_steps
    
    def loop_cond(loop_var):
        (should_stop, time, x_grid, f_grid, steps_performed) = loop_var
        del time, x_grid, f_grid, steps_performed
        return jnp.logical_not(should_stop)
    
    def loop_body(loop_var):
        (should_stop, time, x_grid, f_grid, steps_performed) = loop_var
        
        del should_stop
        next_should_stop, t_next = time_step_fn(time)
        next_xs, next_fs = one_step_fn(
          time=time,
          next_time=t_next,
          coord_grid=x_grid,
          value_grid=f_grid,
          boundary_conditions=boundary_conditions,
          second_order_coeff_fn=second_order_coeff_fn,
          first_order_coeff_fn=first_order_coeff_fn,
          zeroth_order_coeff_fn=zeroth_order_coeff_fn,
          inner_second_order_coeff_fn=inner_second_order_coeff_fn,
          inner_first_order_coeff_fn=inner_first_order_coeff_fn,
          num_steps_performed=steps_performed
        )
        
        print(f"next fs: {next_fs}")
        
        if values_transform_fn is not None:
            next_xs, next_fs = values_transform_fn(t_next, next_xs, next_fs)
        
        return next_should_stop, t_next, next_xs, next_fs,steps_performed + 1
    
    should_already_stop = (start_time == end_time)
    loop_var = (should_already_stop, start_time, coord_grid, values_grid,
                    start_step_count)
    
    
    
    print(f"start:{start_time}")
    (_, final_time, final_coords, final_values, steps_performed) = while_loop(loop_cond, loop_body, loop_var)
    print(f"final values: {final_coords}")
    return final_values, final_coords, final_time, steps_performed 
        

def _is_callable(var_or_fn):
    """Returns whether an object is callable or not."""
    # Python 2.7 as well as Python 3.x with x > 2 support 'callable'.
    # In between, callable was removed hence we need to do a more expansive check
    if hasattr(var_or_fn, '__call__'):
        return True
    try:
        return callable(var_or_fn)
    except NameError:
        return False

def _get_time_steps_info(start_time, end_time, num_steps, time_step, time_direction_fn):
    dt = None
    estimated_max_steps = None
    interval = jnp.abs(end_time - start_time)
    if num_steps is not None:
        dt = interval/jnp.asarray(num_steps, dtype = start_time.dtype)
        
        estimated_max_steps = num_steps
    if time_step is not None and not _is_callable(time_step):
        dt = time_step
        estimated_max_steps = jnp.asarray(jnp.ceil(interval / dt), dtype=jnp.int32)
        
    if dt is not None:
        raw_time_step_fn = lambda _:dt
    else:
        raw_time_step_fn = time_step
        
        
    def time_step_fn(t):
        dt = raw_time_step_fn(t)
        should_stop, t_next = time_direction_fn(t, dt, end_time)
        return should_stop, t_next

    
    return time_step_fn, estimated_max_steps


def _time_direction_forward_fn(t, dt, end_time):
    t_next = jnp.minimum(end_time, t + dt)
    return t_next >= end_time, t_next


def _time_direction_backward_fn(t, dt, end_time):
    t_next = jnp.maximum(end_time, t - dt)
    return t_next <= end_time, t_next

__all__ = ['solve_backward']