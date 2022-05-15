import functools
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, jvp, random
import jax.scipy.stats as stats

import jax_quant_finance as jqf

from absl.testing import absltest
from absl.testing import parameterized

from jax._src import test_util as jtu

from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_numpy_rank_promotion", "allow")

from jax_quant_finance.models import utils
from jax_quant_finance.models import euler_sampling
from jax_quant_finance.math import random_sampler


class EulerSamplingTest(jtu.JaxTestCase):

    @parameterized.named_parameters(
    {
        'testcase_name': 'WhileLoopWithGrid',
        'loop_method': 'while',
        'use_time_step': False,
        'use_time_grid': True,
    }, {
        'testcase_name': 'Scan',
        'loop_method': 'scan',
        'use_time_step': False,
        'use_time_grid': True,
    })
    def test_sample_paths_wiener(self, use_time_step, loop_method, use_time_grid):
        """Tests paths properties for Wiener process (dX = dW)."""
        dtype = jnp.float64

        def drift_fn(_, x) -> jnp.ndarray:
            return jnp.zeros_like(x)

        def vol_fn(_, x) -> jnp.ndarray:
            return jnp.expand_dims(jnp.ones_like(x), -1)

        times = np.array([0.1, 0.2, 0.3])
        num_samples = 4
        if use_time_step:
            time_step = 0.01
            num_time_steps = None
        else:
            time_step = None
            num_time_steps = 30
        if use_time_grid:
            time_step = None
            times_grid = jnp.linspace(start=0.0, stop=0.3, num=31, dtype=dtype)
        else:
            times_grid = None
        
        paths = euler_sampling.sample(
            dim=1, drift_fn=drift_fn, volatility_fn=vol_fn,
            times=times,
            num_samples=num_samples,
            time_step=time_step,
            num_time_steps=num_time_steps,
            times_grid=times_grid,
            random_type=random_sampler.RandomType.STATELESS_ANTITHETIC,
            loop_method=loop_method)
        
        num_samples = 4
        paths = jnp.squeeze(paths, axis=0)

        np.testing.assert_equal(paths.shape, (num_samples, 3, 1))

        means = jnp.mean(paths, axis=0).reshape([-1])
        expected_means = np.zeros((3,))

        covars = np.cov(paths.reshape([num_samples, -1]), rowvar=False)
        expected_covars = np.minimum(times.reshape([-1, 1]), times.reshape([1, -1]))

        np.testing.assert_allclose(means, expected_means, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(covars, expected_covars, rtol=1e-2, atol=1e-2)

    
    # @parameterized.named_parameters(
    # {
    #     'testcase_name': 'NonBatch',
    #     'use_batch': False,
    #     'loop_method': "while",
    #     'random_type': random_sampler.RandomType.STATELESS,
    # }, {
    #     'testcase_name': 'Batch',
    #     'use_batch': True,
    #     'loop_method': "while",
    #     'random_type': random_sampler.RandomType.STATELESS_ANTITHETIC,
    # }, {
    #     'testcase_name': 'BatchWithScan',
    #     'use_batch': True,
    #     'loop_method': "scan",
    #     'random_type': random_sampler.RandomType.STATELESS,
    # })
    # # , {
    # #     'testcase_name': 'BatchAntithetic',
    # #     'use_batch': True,
    # #     'watch_params': None,
    # #     'supply_normal_draws': False,
    # #     'random_type': random_sampler.RandomType.STATELESS_ANTITHETIC,
    # # }, {
    # #     'testcase_name': 'BatchWithNormalDraws',
    # #     'use_batch': True,
    # #     'watch_params': None,
    # #     'supply_normal_draws': True,
    # #     'random_type': random_sampler.RandomType.STATELESS,
    # # })
    # def test_sample_paths_1d(self, use_batch, loop_method, random_type):
    #     """Tests path properties for 1-dimentional Ito process.

    #     We construct the following Ito process.

    #     ````
    #     dX = mu * sqrt(t) * dt + (a * t + b) dW
    #     ````

    #     For this process expected value at time t is x_0 + 2/3 * mu * t^1.5 .
    #     """
    #     dtype = jnp.float64
    #     mu = 0.2
    #     a = 0.4
    #     b = 0.33

    #     def drift_fn(t, x):
    #         drift = mu * jnp.sqrt(t) * jnp.ones_like(x, dtype=t.dtype)
    #         return drift

    #     def vol_fn(t, x):
    #         del x
    #         return (a * t + b) * jnp.ones([1, 1], dtype=t.dtype)

    #     times = np.array([0.0, 0.1, 0.21, 0.32, 0.43, 0.55])
    #     num_samples = 10000

    #     if use_batch:
    #         # x0.shape = [2, 1]
    #         x0 = np.array([[0.1], [0.1]])
    #         paths = euler_sampling.sample(dim=1, drift_fn=drift_fn, volatility_fn=vol_fn,
    #             times=times, num_samples=num_samples,
    #             initial_state=x0,
    #             random_type=random_type,
    #             time_step=0.01,
    #             dtype=dtype,
    #             loop_method=loop_method)
    #     else:
    #         x0 = np.array([0.1])
    #         paths = euler_sampling.sample(
    #             dim=1,
    #             drift_fn=drift_fn, volatility_fn=vol_fn,
    #             times=times, num_samples=num_samples, initial_state=x0,
    #             random_type=random_type,
    #             time_step=0.01,
    #             dtype=dtype)
        
    #     # (len(times), num_samples , dim)
    #     if not use_batch:
    #         paths = paths.transpose((1, 0, 2))
    #         np.testing.assert_equal(paths.shape, (num_samples, 6, 1))
    #     else:
    #         paths = paths.transpose((0, 2, 1, 3))
    #         np.testing.assert_equal(paths.shape, (2, num_samples, 6, 1))
        
    #     if not use_batch:
    #         means = np.mean(paths, axis=0).reshape(-1)
    #     else:
    #         means = np.mean(paths, axis=1).reshape(2, 6)
    #     expected_means = x0 + (2.0 / 3.0) * mu * np.power(times, 1.5)
        
    #     self.assertAllClose(means, expected_means, rtol=1e-2, atol=1e-2)


    # @parameterized.named_parameters(
    # {
    #     'testcase_name': 'STATELESS',
    #     'random_type': random_sampler.RandomType.STATELESS,
    # })
    # def test_sample_paths_2d(self, random_type):
    #     """Tests path properties for 2-dimentional Ito process.

    #     We construct the following Ito processes.

    #     dX_1 = mu_1 sqrt(t) dt + s11 dW_1 + s12 dW_2
    #     dX_2 = mu_2 sqrt(t) dt + s21 dW_1 + s22 dW_2

    #     mu_1, mu_2 are constants.
    #     s_ij = a_ij t + b_ij

    #     For this process expected value at time t is (x_0)_i + 2/3 * mu_i * t^1.5.
    #     """
    #     dtype = jnp.float64
    #     mu = np.array([0.2, 0.7])
    #     a = np.array([[0.4, 0.1], [0.3, 0.2]])
    #     b = np.array([[0.33, -0.03], [0.21, 0.5]])

    #     def drift_fn(t, x):
    #         return jnp.broadcast_to(mu * jnp.sqrt(t), x.shape) * jnp.ones_like(x, dtype=x.dtype)

    #     def vol_fn(t, x):
    #         del x
    #         return (a * t + b) * jnp.ones([2, 2], dtype=t.dtype)

    #     num_samples = 10000
    #     times = np.array([0.1, 0.21, 0.32, 0.43, 0.55])
    #     x0 = np.array([0.1, -1.1])

    #     paths = euler_sampling.sample(
    #             dim=2,
    #             drift_fn=drift_fn, volatility_fn=vol_fn,
    #             times=times,
    #             num_samples=num_samples,
    #             initial_state=x0,
    #             time_step=0.01,
    #             random_type=random_type)

    #     paths = paths.transpose((1, 0, 2))
    #     self.assertAllClose(paths.shape, (num_samples, 5, 2), atol=0)
        
    #     means = np.mean(paths, axis=0)
    #     times = np.reshape(times, [-1, 1])
    #     expected_means = x0 + (2.0 / 3.0) * mu * np.power(times, 1.5)

    #     self.assertAllClose(means, expected_means, rtol=1e-2, atol=1e-2)

    
    # @parameterized.named_parameters({
    #     'testcase_name': '1DBatch',
    #     'batch_rank': 1,
    # })
    # def test_batch_sample_paths_2d(self, batch_rank):
    #     """Tests path properties for a batch of 2-dimentional Ito process.

    #     We construct the following Ito processes.

    #     dX_1 = mu_1 sqrt(t) dt + s11 dW_1 + s12 dW_2
    #     dX_2 = mu_2 sqrt(t) dt + s21 dW_1 + s22 dW_2

    #     mu_1, mu_2 are constants.
    #     s_ij = a_ij t + b_ij

    #     For this process expected value at time t is (x_0)_i + 2/3 * mu_i * t^1.5.

    #     Args:
    #     batch_rank: The rank of the batch of processes being simulated.
    #     """
    #     dtype = jnp.float64

    #     mu = np.array([0.2, 0.7])
    #     a = np.array([[0.4, 0.1], [0.3, 0.2]])
    #     b = np.array([[0.33, -0.03], [0.21, 0.5]])

    #     def drift_fn(t, x):
    #         return jnp.broadcast_to(mu * jnp.sqrt(t), x.shape) * jnp.ones_like(x, dtype=t.dtype)

    #     def vol_fn(t, x):
    #         return (a * t + b) * jnp.ones((2, 2), dtype=x.dtype)

    #     times = np.array([0.1, 0.21, 0.32, 0.43, 0.55])
    #     x0 = np.array([0.1, -1.1]) * np.ones([2, 2])

    #     times_grid = None
    #     time_step = 0.01

    #     num_samples = 10000

    #     random_type = random_sampler.RandomType.STATELESS

    #     paths = euler_sampling.sample(
    #             dim=2,
    #             drift_fn=drift_fn, volatility_fn=vol_fn,
    #             times=times,
    #             num_samples=num_samples,
    #             initial_state=x0,
    #             time_step=0.01,
    #             random_type=random_type)
        
    #     paths = paths.transpose((0, 2, 1, 3))

    #     num_samples = 10000
    #     self.assertAllClose(paths.shape,
    #                         (2, num_samples, 5, 2), atol=0)
    #     means = np.mean(paths, axis=1)
    #     times = np.reshape(times, [1] * 1 + [-1, 1])
    #     expected_means = np.reshape(
    #         x0, (2, 1, 2)) + (2.0 / 3.0) * mu * np.power(times, 1.5)
    #     self.assertAllClose(means, expected_means, rtol=1e-2, atol=1e-2)

if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())