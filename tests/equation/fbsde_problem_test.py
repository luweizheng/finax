"""Tests for FBSDE Equation Problem."""

import jax
import jax.numpy as jnp

from jax import config
config.update("jax_enable_x64", True)

from absl.testing import absltest
from absl.testing import parameterized

from jax._src import test_util as jtu

from finax.equation.fbsde import FBSDEProblem

class FBSDETest(jtu.JaxTestCase):

    def test_construct(self):
        """Tests construct"""
        def g_fn(X):
            return jnp.sum(X ** 2, axis=-1, keepdims=True)

        def mu_fn(t, X):
            del t, Y, Z
            return jnp.zeros_like(X)

        def sigma_fn(t, X, Y):
            del t, Y
            return 0.4 * X

        def phi_fn(t, X, Y, Z):
            del t
            return 0.05 * (Y - jnp.sum(X * Z, axis=1, keepdims=True))
        
        x0 = jnp.array([[1.0, 0.5] * int(100 / 2)])
        x0 = jnp.broadcast_to(x0, (32, 100))

        tspan = (0.0, 1.0)
        num_timesteps = 50

        bsb_problem = FBSDEProblem.create(g_fn=g_fn, 
            mu_fn=mu_fn, 
            sigma_fn=sigma_fn, 
            phi_fn=phi_fn, 
            x0=x0, 
            tspan=tspan, 
            num_timesteps=num_timesteps)

        print(jax.tree_util.tree_structure(bsb_problem))

if __name__ == '__main__':
    absltest.main(testLoader=jtu.JaxTestLoader())



