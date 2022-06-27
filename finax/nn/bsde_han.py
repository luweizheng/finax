"""Model to train Neural Forward-Backward Stochastic Differential Equation."""

import time
from typing import Any, Callable, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom

import flax
from flax import linen as nn
from flax.training.train_state import TrainState
from flax import struct
from flax import core

import optax
from flax.core.frozen_dict import freeze

from finax.nn.loss import mean_square_error
from finax.equation.fbsde import FBSDEProblem

ModuleDef = Any
EquProblemDef = Any

class BSDEHanModel(TrainState):
    # equ_problem: EquProblemDef
    batch_size: int                 # batch size or number of trajectories
    # num_timesteps: int              # number of timesteps
    # dim: int                        # dimension of assets
    # batch_stats: core.FrozenDict[str, Any] = None
    
    
    @classmethod
    def create(cls, *, u_mdl, zgrad_mdl, equ_problem, batch_size, tx, rng=jrandom.PRNGKey(42), **kwargs):
        """Creates a new model instance with parameters."""

        class UNet(nn.Module):

            @nn.compact
            def __call__(self, x, train: bool = True):
                x = nn.Dense(features=40)(x)
                x = nn.relu(x)
                x = nn.Dense(features=40)(x)
                x = nn.relu(x)
                x= nn.Dense(features=1)(x)
                return x

        class ZGradNet(nn.Module):

            @nn.compact
            def __call__(self, x, train: bool = True):
                x = nn.Dense(features=40)(x)
                x = nn.relu(x)
                x = nn.Dense(features=40)(x)
                x = nn.relu(x)
                x = nn.Dense(features=30)(x)
                return x

        class BSDEHanCell(nn.Module):
            zgrad_mdl: nn.Module
            # equ_problem: EquProblemDef

            @nn.compact
            def __call__(self, carry, t, W, train: bool = True):
                i, x0, y0, z0 = carry

                t0 = t[:, i, :]
                t1 = t[:, i + 1, :]
                # t0 = i * t
                # t1 = (i + 1) * t

                W0 = W[:, i, :]
                W1 = W[:, i + 1, :]

                x1 = x0 + equ_problem.mu_fn(t0, x0) * (t1 - t0) + \
                        equ_problem.sigma_fn(t0, x0, y0) * (W1 - W0)
                y1 = y0 - equ_problem.phi_fn(t0, x0, y0, z0) * (t1 - t0) + \
                    jnp.sum(z0 * (W1 - W0), axis=1, keepdims=True)

                z1 = self.zgrad_mdl()(x1)

                carry = (i+1, x1, y1, z1)
                output = (x1, y1)

                return carry, output

        class BSDEHan(nn.Module):
            u_mdl: nn.Module
            zgrad_mdl: nn.Module
            # equ_problem: Any
            dt: float

            # def setup(self):
            #     self.u = self.u_mdl()
            #     self.zgrad = self.zgrad_mdl()
            
            def u_fn(self, x0):
                return self.u_mdl()(x0)

            def z_fn(self, x0):
                return self.zgrad_mdl()(x0)

            @nn.compact
            def __call__(self, x0, t, W, length=1, unroll=1):
                y0 = self.u_fn(x0)
                z0 = self.z_fn(x0)

                initial_carry = (0, x0, y0, z0)

                fbsdes = nn.scan(BSDEHanCell,
                        variable_axes={'params': 0},
                        split_rngs={"params": True},
                        in_axes=flax.core.broadcast,
                        out_axes=0,
                        length=length,
                        unroll=unroll)
                y = fbsdes(zgrad_mdl=ZGradNet)(initial_carry, t, W)
            
                return y

        
        dt = (equ_problem.tspan[1] - equ_problem.tspan[0]) / equ_problem.num_timesteps

        bsde_net = BSDEHan(u_mdl=UNet, zgrad_mdl=ZGradNet, dt=dt)

        _, variables = bsde_net.init_with_output(
            rng, 
            x0=equ_problem.x0, 
            t=jnp.zeros((batch_size, equ_problem.num_timesteps, 1)), 
            W=jnp.ones((batch_size, equ_problem.num_timesteps, equ_problem.dim)),
            length=equ_problem.num_timesteps)

        opt_state = tx.init(variables['params'])

        return cls(
            step=0,
            apply_fn=bsde_net.apply,
            params=variables['params'],
            tx=tx,
            opt_state=opt_state,
            # equ_problem=equ_problem,
            batch_size=batch_size,
            **kwargs,
        )

def fetch_minibatch(equ_problem, batch_size, rng):  # Generate time + a Brownian motion
    T = equ_problem.tspan[1]
    M = batch_size
    N = equ_problem.num_timesteps
    D = equ_problem.dim
 
    Dt = jnp.concatenate([jnp.zeros((M, 1), dtype=jnp.float32), jnp.ones((M, N)) * T / N], axis=1).reshape((M, N+1, 1))  # M x (N+1) x 1
    DW = jnp.concatenate([jnp.zeros((M, 1, D), dtype=jnp.float32), jnp.sqrt(T / N) * jrandom.normal(rng, shape=(M, N, D))], axis=1) # M x (N+1) x D

    t = jnp.cumsum(Dt, axis=1)  # M x (N+1) x 1
    W = jnp.cumsum(DW, axis=1)  # M x (N+1) x D
    
    return t, W

# def fetch_minibatch(equ_problem, batch_size, rng):  # Generate time + a Brownian motion
#     T = equ_problem.tspan[1]
#     M = batch_size
#     N = equ_problem.num_timesteps
#     D = equ_problem.dim

#     # Dt = jnp.concatenate(
#     #     [jnp.zeros((M, 1), dtype=jnp.float32), jnp.ones((M, N)) * T / N], axis=1).reshape((M, N+1, 1))  # M x (N+1) x 1
#     # DW = jnp.concatenate([jnp.zeros((M, 1, D), dtype=jnp.float32), jnp.sqrt(T / N) * jrandom.normal(rng, shape=(M, N, D))], axis=1) # M x (N+1) x D

#     # t = jnp.cumsum(Dt, axis=1)  # M x (N+1) x 1
#     # W = jnp.cumsum(DW, axis=1)  # M x (N+1) x D

#     dt = T / N * jnp.ones((M, 1))
#     dW = jnp.sqrt(T / N) * jrandom.normal(rng, shape=(M, N, D))

#     return dt, dW


@jax.jit
def train_step(train_state, data, x0, equ_problem, unroll):
    
    t, W = data

    def loss_fn(params):
        loss = 0.0
    
        out_carry, out_val = train_state.apply_fn({'params': params}, x0=x0, t=t, W=W, length=equ_problem.num_timesteps, unroll=unroll)
        (_, x_final, y_final, z_final) = out_carry
        (x_list, y_list) = out_val

        loss += mean_square_error(y_final, equ_problem.g_fn(x_final))

        return (loss, y_final)
    
    (loss, y), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
    
    train_state = train_state.apply_gradients(grads=grads)

    return train_state, loss, y


def train(train_state, x0, num_iters, equ_problem, unroll=1, rng=jrandom.PRNGKey(42), verbose=True):
    start_time = time.time()
    iter_time = time.time()
    
    for i in range(num_iters):
        rng, _ = jax.random.split(rng)
        data = fetch_minibatch(equ_problem, train_state.batch_size, rng)
        with jax.checking_leaks():
            train_state, loss, result = train_step(train_state, data, x0, equ_problem, unroll)
            if verbose:
                if i % 100 == 0:
                    elapsed = time.time() - iter_time
                    print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f' %
                            (i, loss, result[0], elapsed))
                    iter_time = time.time()

    if verbose:
        print(f"total time: {time.time() - start_time}")
    
    return train_state