"""Utility methods for model building."""
import jax.numpy as jnp

from jax_quant_finance.utils import ops


def prepare_grid(*, times, time_step, dtype, tolerance=None,
                 num_time_steps=None, times_grid=None):
    """Prepares grid of times for path generation.

    Args:
        times:  Rank 1 `Tensor` of increasing positive real values. The times at
        which the path points are to be evaluated.
        time_step: Rank 0 real `Tensor`. Maximal distance between points in
        resulting grid.
        dtype: `tf.Dtype` of the input and output `Tensor`s.
        tolerance: A non-negative scalar `Tensor` specifying the minimum tolerance
        for discernible times on the time grid. Times that are closer than the
        tolerance are perceived to be the same.
        Default value: `None` which maps to `1-e6` if the for single precision
            `dtype` and `1e-10` for double precision `dtype`.
        num_time_steps: Number of points on the grid. If suppied, a uniform grid
        is constructed for `[time_step, times[-1] - time_step]` consisting of
        max(0, num_time_steps - len(times)) points that is then concatenated with
        times. This parameter guarantees the number of points on the time grid
        is `max(len(times), num_time_steps)` and that `times` are included to the
        grid.
        Default value: `None`, which means that a uniform grid is created.
        containing all points from 'times` and the uniform grid of points between
        `[0, times[-1]]` with grid size equal to `time_step`.
        times_grid: An optional rank 1 `Tensor` representing time discretization
        grid. If `times` are not on the grid, then the nearest points from the
        grid are used.
        Default value: `None`, which means that times grid is computed using
        `time_step` and `num_time_steps`.

    Returns:
        Tuple `(all_times, mask, time_indices)`.
        `all_times` is a 1-D real `Tensor`. If `num_time_steps` is supplied the
        shape of the output is `max(num_time_steps, len(times))`. Otherwise
        consists of all points from 'times` and the uniform grid of points between
        `[0, times[-1]]` with grid size equal to `time_step`.
        `mask` is a boolean 1-D `Tensor` of the same shape as 'all_times', showing
        which elements of 'all_times' correspond to THE values from `times`.
        Guarantees that times[0]=0 and mask[0]=False.
        `time_indices`. An integer `Tensor` of the same shape as `times` indicating
        `times` indices in `all_times`.
    """
    if tolerance is None:
        tolerance = 1e-10 if dtype == jnp.float64 else 1e-6
    tolerance = jnp.asarray(tolerance, dtype=dtype)
    if times_grid is None:
        if num_time_steps is None:
            all_times, time_indices = _grid_from_time_step(
          times=times, time_step=time_step, dtype=dtype, tolerance=tolerance)
        else:
            all_times, time_indices = _grid_from_num_times(
                times=times, time_step=time_step, num_time_steps=num_time_steps)
    else:
        all_times = times_grid
        time_indices = jnp.searchsorted(times_grid, times)
        # Adjust indices to bring `times` closer to `times_grid`.
        times_diff_1 = times_grid[time_indices] - times
        times_diff_2 = times_grid[jnp.maximum(time_indices-1, 0)] - times
        time_indices = jnp.where(
            jnp.abs(times_diff_2) > jnp.abs(times_diff_1),
            time_indices,
            jnp.maximum(time_indices - 1, 0))
    
    # Create a boolean mask to identify the iterations that have to be recorded.
    # Use `tf.scatter_nd` because it handles duplicates. Also we first create
    # an int64 Tensor and then create a boolean mask because scatter_nd with
    # booleans is currently not supported on GPUs.
    mask = ops.scatter_nd(
        indices=jnp.expand_dims(jnp.asarray(time_indices, dtype=jnp.int64), axis=1),
        updates=jnp.ones(times.shape),
        shape=all_times.shape)
    mask = jnp.where(mask > 0, True, False)

    return all_times, mask, time_indices


def _grid_from_time_step(*, times, time_step, dtype, tolerance):
    """Creates a time grid from an input time step."""
    grid = jnp.arange(0.0, times[-1], time_step, dtype=dtype)
    all_times = jnp.concatenate([times, grid], axis=0)
    all_times = jnp.sort(all_times)

    # Remove duplicate points
    dt = all_times[1:] - all_times[:-1]
    dt = jnp.concatenate([jnp.asarray([1.0], dtype=dtype), dt], axis=-1)
    
    # if two consecutive elements are equal, their difference (dt) would be less
    duplicate_mask = jnp.greater(dt, tolerance)
    all_times = all_times[duplicate_mask]
    time_indices = jnp.searchsorted(all_times, times)
    time_indices = jnp.minimum(time_indices, jnp.asarray(all_times.shape[0] - 1))

    # Move `time_indices` to the left, if the requested `times` are removed from
    # `all_times` during deduplication
    time_indices = jnp.where(
        all_times[time_indices] - times > tolerance,
        time_indices - 1,
        time_indices)

    return all_times, time_indices


def _grid_from_num_times(*, times, time_step, num_time_steps):
    """Creates a time grid for the requeste number of time steps."""
    # Build a uniform grid for the timestep of size
    # max(0, num_time_steps - tf.shape(times)[0])
    uniform_grid = jnp.linspace(
        time_step, times[-1] - time_step,
        jnp.maximum(num_time_steps - times.shape[0], 0))
    grid = jnp.sort(jnp.concatenate([uniform_grid, times], axis=0))
    # Add zero to the time grid
    all_times = jnp.concatenate([jnp.asarray([0]), grid], axis=0)
    time_indices = jnp.searchsorted(all_times, times)
    return all_times, time_indices