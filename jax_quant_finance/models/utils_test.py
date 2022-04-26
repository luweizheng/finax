import numpy as np
import jax.numpy as jnp
from jax import jit, jvp
import jax.scipy.stats as stats

import jax_quant_finance as jqf

from absl.testing import absltest
from absl.testing import parameterized

from jax._src import test_util as jtu

from jax.config import config
config.update("jax_enable_x64", True)

from jax_quant_finance.models import utils

class UtilsTest(jtu.JaxTestCase):

    @parameterized.named_parameters(
      ('SinglePrecision', np.float32),
      ('DoublePrecision', np.float64),
    )
    def test_prepare_grid_num_time_step(self, dtype):
        num_points = 100
        times = jnp.linspace(0.02, 1.0, 50, dtype=dtype)
        time_step = times[-1] / num_points
        grid, keep_mask, time_indices = utils.prepare_grid(
            times=times, time_step=time_step, dtype=dtype,
            num_time_steps=num_points)
        expected_grid = np.linspace(0, 1, 101, dtype=dtype)
        expected_indices = np.array([2 * i for i in range(1, 51)])
        
        np.testing.assert_allclose(grid, expected_grid, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(time_indices, expected_indices, rtol=1e-6, atol=1e-6)


    @parameterized.named_parameters(
        ('SinglePrecision', np.float32),
        ('DoublePrecision', np.float64),
    )
    def test_prepare_grid_time_step(self, dtype):
        times = jnp.asarray([0.1, 0.5, 1, 2], dtype=dtype)
        time_step = jnp.asarray(0.1, dtype=jnp.float64)
        grid, keep_mask, time_indices = utils.prepare_grid(
            times=times, time_step=time_step, dtype=dtype)
        expected_grid = np.linspace(0, 2, 21, dtype=dtype)
        recovered_times = grid[time_indices]

        np.testing.assert_allclose(grid, expected_grid, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(times, recovered_times, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())