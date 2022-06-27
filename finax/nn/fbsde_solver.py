"""Model to train Neural Forward-Backward Stochastic Differential Equation."""

import time
from functools import partial

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrandom

from flax.core import broadcast
from flax import linen as nn
from flax.training.train_state import TrainState
from flax import struct
from flax import core

import optax
from flax.core.frozen_dict import freeze

from finax.nn.loss import sum_square_error
from finax.equation.fbsde import FBSDEProblem
from .model import Model

ModuleDef = Any
EquProblemDef = Any

class FBSDETrainState(TrainState):
    batch_size: int
    
    @classmethod
    def create(cls, *, mdl, equ_problem, batch_size, tx, rng=jrandom.PRNGKey(42), **kwargs):
        """Creates a new model instance with parameters."""
        
        net = mdl()
        variables = net.init(rng, t=jnp.zeros([batch_size, 1]), x=equ_problem.x0)
        opt_state = tx.init(variables['params'])
        # tx = optax.adam(learning_rate=learning_rate)

        def u_du_fn(params, t, x):
            """Forward Function
            """
            # M x 1, M x D
            def u_fn(t, x):
                u = net.apply({'params': params}, t, x)
                return u
            u, vjp_fn = jax.vjp(u_fn, t, x)
            du_dt, du_dx = vjp_fn(jnp.ones_like(u))
            return u, du_dx

        return cls(
            step=0,
            apply_fn=u_du_fn,
            params=variables['params'],
            tx=tx,
            opt_state=opt_state,
            batch_size=batch_size,
            **kwargs,
        )
    

def fetch_minibatch(equ_problem, batch_size, rng):  # Generate time + a Brownian motion
    T = equ_problem.tspan[1]
    M = batch_size
    N = equ_problem.num_timesteps
    D = equ_problem.dim

    # Dt = jnp.concatenate(
    #     [jnp.zeros((M, 1), dtype=jnp.float32), jnp.ones((M, N)) * T / N], axis=1).reshape((M, N+1, 1))  # M x (N+1) x 1
    # DW = jnp.concatenate([jnp.zeros((M, 1, D), dtype=jnp.float32), jnp.sqrt(T / N) * jrandom.normal(rng, shape=(M, N, D))], axis=1) # M x (N+1) x D

    # t = jnp.cumsum(Dt, axis=1)  # M x (N+1) x 1
    # W = jnp.cumsum(DW, axis=1)  # M x (N+1) x D

    dt = T / N * jnp.ones((M, 1))
    dW = jnp.sqrt(T / N) * jrandom.normal(rng, shape=(M, N, D))

    return dt, dW

def sde(params, train_state, samples, equ_problem, unroll=1):

    t, W = samples

    def initialize_state():  
        t0 = t * 0.0
        x0 = equ_problem.x0
        y0, z0 = train_state.apply_fn(params, t0, x0)

        # (loop_counter, x, y, z)
        initial_state = (0, x0, y0, z0)
        return initial_state
    
    def step_fn(carry, xs):
        i, x0, y0, z0 = carry

        t0 = i * t
        t1 = (i+1) * t
        dW = W[:, i, :]

        y0 = jnp.reshape(y0, newshape=(-1, 1))

        x1 = x0 + equ_problem.mu_fn(t0, x0, y0, z0) * t + \
                equ_problem.sigma_fn(t0, x0, y0) * dW
        y1_tilde = y0 + equ_problem.phi_fn(t0, x0, y0, z0) * t + \
            jnp.sum(z0 * equ_problem.sigma_fn(t0, x0, y0) * dW, axis=1, keepdims=True)
        y1, z1 = train_state.apply_fn(params, t1, x1)

        carry = (i+1, x1, y1, z1)
        return carry, (x1, y1, y1_tilde)
            
    initial_state = initialize_state()
    (_, x_final, y_final, z_final), output_list = jax.lax.scan(
        f=step_fn,
        init=initial_state,
        xs=None,
        length=50,
        unroll=unroll)

    return x_final, y_final, z_final, output_list

@partial(jax.jit, static_argnums=3)
def train_step(train_state, data, equ_problem, unroll=1):
    t, W = data

    def loss_fn(params):
        loss = 0.0
    
        x_final, y_final, z_final, output_list = sde(params, train_state, data, equ_problem, unroll)
        (x_list, y_list, y_tilde_list) = output_list
        # print(x_list.shape)
        loss += sum_square_error(y_list, y_tilde_list)
        # print(loss)
        loss += sum_square_error(y_final, equ_problem.g_fn(x_final))
        # print(loss)
        loss += sum_square_error(z_final, equ_problem.dg_fn(x_final))
        # print(loss)

        result = {
            'x': x_list,
            'y': y_list   
        }

        return loss, result

    (loss, result), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)

    train_state = train_state.apply_gradients(grads=grads)

    return train_state, loss, result
        

def train(train_state, num_iters, equ_problem, unroll=1, rng=jrandom.PRNGKey(42), verbose=True):
    start_time = time.time()
    iter_time = time.time()
    
    for i in range(num_iters):
        rng, _ = jax.random.split(rng)
        data = fetch_minibatch(equ_problem, train_state.batch_size, rng)
        train_state, loss, result = train_step(train_state, data, equ_problem, unroll)

        if verbose:
            if i % 100 == 0:
                elapsed = time.time() - iter_time
                print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f' %
                        (i, loss, result['y'][0, 0, 0], elapsed))
                iter_time = time.time()

    if verbose:
        print(f"total time: {time.time() - start_time}")
    
    return train_state

