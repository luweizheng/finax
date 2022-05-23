import numpy as np
import jax.numpy as jnp

import finax

from absl.testing import absltest
from absl.testing import parameterized

from jax._src import test_util as jtu

from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_numpy_rank_promotion", "allow")

from finax.math import random_sampler


class GenericItoProcessTest(jtu.JaxTestCase):

    def test_sample_paths_wiener(self):
        """Tests paths properties for Wiener process (dX = dW)."""

        def drift_fn(t, x):
            return jnp.zeros_like(x)

        def vol_fn(t, x):
            return jnp.expand_dims(jnp.ones_like(x), -1)

        process = finax.models.GenericItoProcess(
            dim=1, drift_fn=drift_fn, volatility_fn=vol_fn)
        times = jnp.array([0.1, 0.2, 0.3])
        num_samples = 10000

        # (batch_size, num_samples, len(times), dim)
        paths = process.sample_paths(times=times, num_samples=num_samples, seed=42, time_step=0.01)
        
        means = jnp.mean(paths, axis=1).reshape((-1))
        covars = jnp.cov(paths.reshape([num_samples, -1]), rowvar=False)
        expected_means = np.zeros((3,))
        expected_covars = np.minimum(times.reshape([-1, 1]), times.reshape([1, -1]))
        
        np.testing.assert_allclose(means, expected_means, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(covars, expected_covars, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())