"""Tests for random sampling."""

import numpy as np
import jax.numpy as jnp
from jax import random

from jax import config
config.update("jax_enable_x64", True)

from absl.testing import absltest
from absl.testing import parameterized

from jax._src import test_util as jtu

import jax_quant_finance.math.random_sampler as jqf_rnd

class OpsTest(jtu.JaxTestCase):

    def test_shapes(self):
        """Tests the sample shapes."""
        subkey = random.PRNGKey(0)
        sample_no_batch = jqf_rnd.normal_pseudo([2, 4], mean=jnp.array([0.2, 0.1]), key=subkey)
        print(sample_no_batch)
        print("====")
        self.assertEqual(sample_no_batch.shape, (2, 4, 2))
        new_key, subkey = random.split(subkey)
        sample_no_batch = jqf_rnd.normal_pseudo([2, 4], mean=jnp.array([0.2, 0.1]), key=subkey)
        print(sample_no_batch)
        sample_batch = jqf_rnd.normal_pseudo([2, 4], mean=jnp.array([[0.2, 0.1], [0., -0.1], [0., 0.1]]), key=subkey)
        self.assertEqual(sample_batch.shape, (2, 4, 3, 2))


if __name__ == '__main__':
    absltest.main(testLoader=jtu.JaxTestLoader())