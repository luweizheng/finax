import pytest
import numpy as np
import jax.numpy as jnp

from jax.config import config
config.update("jax_enable_x64", True)

from jax_quant_finance.models import utils


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_prepare_grid_num_time_step(dtype):
    num_points = 100
    times = jnp.linspace(0.02, 1.0, 50, dtype=dtype)
    time_step = times[-1] / num_points
    grid, _, time_indices = utils.prepare_grid(
        times=times, time_step=time_step, dtype=dtype,
        num_time_steps=num_points)
    expected_grid = jnp.linspace(0, 1, 101, dtype=dtype)
    expected_indices = jnp.array([2 * i for i in range(1, 51)], dtype=dtype)
    assert jnp.allclose(grid, expected_grid, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(time_indices, expected_indices, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_prepare_grid_time_step(dtype):
    times = jnp.asarray([0.1, 0.5, 1, 2], dtype=dtype)
    time_step = jnp.asarray(0.1, dtype=jnp.float64)
    grid, _, time_indices = utils.prepare_grid(
        times=times, time_step=time_step, dtype=dtype)
    expected_grid = jnp.linspace(0, 2, 21, dtype=dtype)
    recovered_times = grid[time_indices]
    assert jnp.allclose(grid, expected_grid, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(times, recovered_times, rtol=1e-6, atol=1e-6)