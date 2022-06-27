"""Methods to train Neural Forward-Backward Stochastic Differential Equation."""

import dataclasses
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

ModuleDef = Any
EquProblemDef = Any

class FBSDEModel(struct.PyTreeNode):
    step: int
    u_net_apply_fn: Callable = struct.field(pytree_node=False)
    # u_params: core.FrozenDict[str, Any]
    fbsde_net_apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any]
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState
    # initial_carry: Tuple
    equ_problem: EquProblemDef
    batch_size: int                 # batch size or number of trajectories
    num_timesteps: int              # number of timesteps
    dim: int                        # dimension of assets
    batch_stats: core.FrozenDict[str, Any] = None

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
            grads: Gradients that have the same pytree structure as `.params`.
            **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
            An updated instance of `self` with `step` incremented by one, `params`
            and `opt_state` updated by applying `grads`, and additional attributes
            replaced as specified by `kwargs`.
        """
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )
    
    
    @classmethod
    def create(cls, *, mdl, equ_problem, batch_size, num_timesteps, dim, tx, rng=jrandom.PRNGKey(42), **kwargs):
        """Creates a new model instance with parameters."""
        
        class UNet(nn.Module):
            u: ModuleDef

            @nn.compact
            def __call__(self, t, x):
                u = self.u()
                (y, bwd) = nn.vjp(lambda mdl, x: mdl(t, x), u, x)
                dudx = bwd(jnp.ones(y.shape))
                return y, dudx[0]

        class FBSDECell(nn.Module):
            u_net: Callable[..., nn.Module]

            @nn.compact
            def __call__(self, carry, t, W):
                # `t` and `W` are (batch_size, num_timestep, dim)
                # it have input data across iterations 
                i, x0, y0, z0 = carry

                # use `i` to index input data
                t0 = t[:, i-1, :]
                t1 = t[:, i, :]
                W0 = W[:, i-1, :]
                W1 = W[:, i, :]

                x1 = x0 + equ_problem.mu_fn(t0, x0, y0, z0) * (t1 - t0) + \
                        equ_problem.sigma_fn(t0, x0, y0) * (W1 - W0)
                y1_tilde = y0 + equ_problem.phi_fn(t0, x0, y0, z0) * (t1 - t0) + \
                    jnp.sum(z0 * equ_problem.sigma_fn(t0, x0, y0) * (W1 - W0), axis=1, keepdims=True)
                
                y1, z1 = self.u_net(t1, x1)

                carry = (i+1, x1, y1, z1)
                outputs = (x1, y1_tilde, y1)
                return carry, outputs


        class FBSDE(nn.Module):
            # u_net: ModuleDef
            # equ_problem: Any

            @nn.compact
            def __call__(self, carry, t, W, length=1):
                # I want t and W share across different iterations, 
                # and use a index variable to index intput data,
                # should I use `flax.core.broadcast`

                # I want the neural network params share across iterations,
                # should I use variable_broadcast="params"
                fbsdes = nn.scan(FBSDECell,
                        variable_broadcast="params",
                        split_rngs={"params": False},
                        in_axes=broadcast,
                        out_axes=0,
                        length=length)
                y = fbsdes(name="fbsde", 
                    u_net=u_net)(carry, t, W)
                return y

        u_net = UNet(u=mdl)
        fbsde_net = FBSDE()

        (y0, z0), u_du_variables = u_net.init_with_output(
            rng, t=jnp.zeros([batch_size, 1]), x=equ_problem.x0)
        
        initial_carry = (1, equ_problem.x0, y0, z0)

        # p = fbsde_net.init(rng, initial_carry, jnp.zeros([batch_size, num_timesteps + 1, 1]), jnp.ones([batch_size, num_timesteps + 1, dim]))
        # print(f"p.params {p['params']}")
        
        fbsde_params = {
            'fbsde':  {
                'u_net': u_du_variables['params']
            }
        }
        fbsde_params = freeze(fbsde_params)

        opt_state = tx.init(fbsde_params)

        return cls(
            step=0,
            u_net_apply_fn=u_net.apply,
            fbsde_net_apply_fn=fbsde_net.apply,
            params=fbsde_params,
            tx=tx,
            opt_state=opt_state,
            equ_problem=equ_problem,
            batch_size=batch_size,
            num_timesteps=num_timesteps,
            dim=dim,
            **kwargs,
        )
    

def fetch_minibatch(model, rng):  # Generate time + a Brownian motion
    T = model.equ_problem.tspan[1]
    M = model.batch_size
    N = model.num_timesteps
    D = model.dim

    Dt = jnp.concatenate(
        [jnp.zeros((M, 1), dtype=jnp.float32), jnp.ones((M, N)) * T / N], axis=1).reshape((M, N+1, 1))  # M x (N+1) x 1
    DW = jnp.concatenate([jnp.zeros((M, 1, D), dtype=jnp.float32), jnp.sqrt(T / N) * jrandom.normal(rng, shape=(M, N, D))], axis=1) # M x (N+1) x D

    t = jnp.cumsum(Dt, axis=1)  # M x (N+1) x 1
    W = jnp.cumsum(DW, axis=1)  # M x (N+1) x D

    return t, W

@jax.jit
def train_step(model, data, batch_size):
    # batch_size = model.batch_size
    t, W = data

    def loss_fn(params):
        loss = 0.0
        
        (y0, z0) = model.u_net_apply_fn(
            {'params': params['fbsde']['u_net']}, 
            # t=t[:, 0], 
            t=jnp.zeros((batch_size, 1)),
            x=model.equ_problem.x0)
        # define initial carry for nn.scan
        x0 = model.equ_problem.x0
        initial_carry = (1, x0, y0, z0)

        out_carry, out_val = model.fbsde_net_apply_fn(
            {'params': params}, 
            carry=initial_carry, 
            t=t, W=W, 
            length=51)

        (_, x_final, y_final, z_final) = out_carry
        (x, y_tilde_list, y_list) = out_val

        loss += sum_square_error(y_tilde_list, y_list)
        loss += sum_square_error(y_final, model.equ_problem.g_fn(x_final))
        loss += sum_square_error(z_final, model.equ_problem.dg_fn(x_final))

        return (loss, y_list)

    (loss, y), grads = jax.value_and_grad(
        loss_fn, has_aux=True)(model.params)
    
    model = model.apply_gradients(grads=grads)

    return model, loss, y

def train(model, num_iters, batch_size, rng=jrandom.PRNGKey(42), verbose=True):
    start_time = time.time()
    iter_time = time.time()
    
    for i in range(num_iters):
        rng, _ = jax.random.split(rng)
        data = fetch_minibatch(model, rng)
        model, loss, y_pred = train_step(model, data, batch_size)

        if verbose:
            if i % 100 == 0:
                elapsed = time.time() - iter_time
                print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f' %
                        (i, loss, y_pred[0, 0, 0], elapsed))
                iter_time = time.time()

    if verbose:
        print(f"total time: {time.time() - start_time}")
    
    return model

