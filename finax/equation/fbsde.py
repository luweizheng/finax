"""Classes that specify Forward-Backward Stochastic Differential Equation Problem."""

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
from flax import struct

Array = jnp.ndarray

class FBSDEProblem(struct.PyTreeNode):
    """Base class for a Forward-Backward SDE Problem.

    Attributes:
        g_fn: the terminal condition for the equation, 
            input parameter is X at the terminal time.
        dg_fn: the gradient function of `g_fn`.
        mu_fn: the drift function of the forward process X, 
            input parameters are `t`, `Xt`, `Yt` and `Zt`.
        sigma_fn: the diffusion fucntion of the forward process X, 
            input parameters are `t`, `Xt` and `Yt`.
        phi_fn: the function of the backward process Y,
            input parameters are `t`, `Xt`, `Yt`, `Zt`.
        x0: the initial X for the problem.
        tspan: the time span of the problem with the following form
            : (start, end) 
        num_timesteps: number of timesteps.
    """
    g_fn: Callable = struct.field(pytree_node=False)
    dg_fn: Callable = struct.field(pytree_node=False)
    mu_fn: Callable = struct.field(pytree_node=False)
    sigma_fn: Callable = struct.field(pytree_node=False)
    phi_fn: Callable = struct.field(pytree_node=False)
    x0: Array
    tspan: tuple[float, float]
    num_timesteps: int = struct.field(pytree_node=False)
    dim: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, *, g_fn, mu_fn, sigma_fn, phi_fn, x0, tspan, num_timesteps, dim, **kwargs):
        """Creates a new instance with input parameters."""
        
        def dg_fn(X):
            y, vjp_func = jax.vjp(g_fn, X)
            return vjp_func(jnp.ones(y.shape))[0]

        return cls(
            g_fn=g_fn,
            dg_fn=dg_fn,
            mu_fn=mu_fn,
            sigma_fn=sigma_fn,
            phi_fn=phi_fn,
            x0=x0,
            tspan=tspan,
            num_timesteps=num_timesteps,
            dim=dim,
            **kwargs,
        )


