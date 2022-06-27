import jax.numpy as jnp
from jax import lax
from jax.scipy.special import erf
from finax.models import generic_ito_process
from finax.experimental.math.pde import grids
from finax.math import random_sampler
from finax.experimental.heston import utils



__all__ = [
    'HestonModel'
]

_SQRT_2 = jnp.sqrt(2)

class HestonModel(generic_ito_process.GenericItoProcess):
    
    def __init__(self, mean_reversion, theta, volvol, rho, dtype):
        self._dtype = dtype or jnp.float32
        self._mean_reversion = jnp.asarray(mean_reversion, dtype=self._dtype)
        self._theta = jnp.asarray(theta, dtype=self._dtype)
        self._volvol = jnp.asarray(volvol, dtype=self._dtype)
        self._rho = jnp.asarray(rho, dtype=self._dtype)
        
        
        def _vol_fn(t, x):
            vol = jnp.sqrt(jnp.abs(x[..., 1]))
            zeros = jnp.zeros_like(vol)
            rho, volvol = self._rho, self._volvol # scalar  support only
            vol_matrix_2 = jnp.stack([zeros, volvol * jnp.sqrt(1 - rho ** 2) * vol], -1)
            vol_matrix_1 = jnp.stack([vol, volvol * rho * vol], -1)
            vol_matrix = jnp.stack([vol_matrix_1, vol_matrix_2])
            return vol_matrix
        
        
        def _drift_fn(t, x):
            var = x[..., 1]
            mean_reversion, theta = self._mean_reversion, self._theta # scalar support only
            log_spot_drift = -var/2
            var_drift = mean_reversion * (theta - var)
            drift = jnp.stack([log_spot_drift, var_drift], -1)
            return drift
        
    
        super(HestonModel, self).__init__(2, _drift_fn, _vol_fn, self._dtype)
    
    def sample_paths(self, 
                     times,
                     initial_state,
                     num_samples = 1,
                     random_type = None,
                     key = None,
                     time_step = None,
                     tolerance=1e-6,
                     num_time_steps = None,
                     precompute_normal_draws = True,
                     times_grid=None,
                     normal_draws = None):
        times = jnp.asarray(times, dtype=self._dtype)
        
        if normal_draws is not None:
            normal_draws = jnp.asarray(normal_draws, dtype=self._dtype)
            perm = [1, 0, 2]
            normal_draws = lax.transpose(normal_draws, permutation=perm)
            num_samples = normal_draws.shape[-1]
            
        current_log_spot = (
            jnp.asarray(initial_state[..., 0], dtype=self._dtype)
            +jnp.zeros([num_samples], dtype=self._dtype)
        )
        current_vol = (
            jnp.asarray(initial_state[..., 1], dtype=self._dtype)
            +jnp.zeros([num_samples], dtype=self._dtype)
        )
        
        
        num_requested_times = times.shape[0]
        
        if times_grid is None:
            pass
        else:
            times_grid = jnp.asarray(times_grid, dtype=self._dtype)
            times = times_grid
        
        return self._sample_path(times=times,
                                 num_requested_times=num_requested_times,
                                 current_log_spot=current_log_spot,
                                 current_vol=current_vol,
                                 num_samples=num_samples,
                                 random_type=random_type,
                                 key=key,
                                 tolerance=tolerance,
                                 precompute_normal_draws=precompute_normal_draws,
                                 normal_draws=normal_draws)
        
        
        
    def _sample_path(self, 
                     times, 
                     num_requested_times, 
                     current_log_spot, 
                     current_vol, 
                     num_samples, 
                     random_type, 
                     key, 
                     tolerance, 
                     precompute_normal_draws, 
                     normal_draws):
        
        dt = times[1:] - times[:-1]
        mean_reversion, theta, volvol, rho = self._mean_reversion, self._theta, self._volvol, self._rho
        
        steps_num = dt.shape[-1]
        
        if normal_draws is None:
            if precompute_normal_draws or random_type in (
                random_sampler.RandomType.STATELESS,
                random_sampler.RandomType.STATELESS_ANTITHETIC
            ) :
                normal_draws = utils.generate_mc_normal_draws(
                    num_normal_draws=2, num_time_steps=steps_num,
                    num_sample_paths=num_samples, random_type=random_type,
                    dtype=self._dtype, key=key
                )
            else:
                normal_draws = None
                
        log_spot_paths = jnp.zeros(shape=(steps_num, num_samples), dtype=mean_reversion.dtype)
        
        vol_paths = jnp.zeros(shape=[steps_num, num_samples], dtype=mean_reversion.dtype)
        
        def cond_fn(args):
            i, _, _, _, _ = args
            return jnp.logical_and(i < steps_num, True)
        
        
        def body_fn(args):
            i, current_vol, current_log_spot, vol_paths, log_spot_paths = args
            time_step = dt[i]
            
            if normal_draws is None:
                normals = random_sampler.normal_pseudo(
                    (num_samples, ),
                    mean=jnp.zeros([2], dtype=mean_reversion.dtype), key=key
                )
            
            else:
                normals = normal_draws[i]
            
            def _next_vol_fn():
                return self._update_variance(mean_reversion, theta, volvol, rho, current_vol, time_step, normals[..., 0])
            
            next_vol = lax.cond(time_step > tolerance, _next_vol_fn, lambda: current_vol)
            
            def _next_log_spot_fn():
                return self._update_log_spot(
                    mean_reversion, theta, volvol, rho,
                    current_vol, next_vol, current_log_spot, time_step, normals[..., 1]
                )
            
            next_log_spot = lax.cond(time_step > tolerance, _next_log_spot_fn, lambda: current_log_spot)
            
            
            vol_paths = vol_paths.at[i].set(next_vol)
            
            log_spot_paths = log_spot_paths.at[i].set(next_log_spot)
            
            return (i+1, next_vol, next_log_spot, vol_paths, log_spot_paths)

        
        _, _, _, vol_paths, log_spot_paths = lax.while_loop(cond_fn, body_fn, (0, current_vol, current_log_spot, vol_paths, log_spot_paths))
        return jnp.stack([log_spot_paths, vol_paths], axis=-1)
    
    
    def _update_variance(
        self,
        mean_reversion, theta, volvol, rho, current_vol, time_step, normals, psi_c=1.5):
        del rho
        psi_c = jnp.asarray(psi_c, dtype=mean_reversion.dtype)
        scaled_time = jnp.exp(-mean_reversion * time_step)
        volvol_squared = volvol ** 2
        m = theta + (current_vol - theta) * scaled_time
        s_squared = (
            current_vol * volvol_squared * scaled_time / mean_reversion 
            * (1 - scaled_time) + theta * volvol_squared / 2 / mean_reversion
            * (1 - scaled_time)**2
        ) 
        
        psi = s_squared/ m**2
        uniforms = 0.5 * (1 + erf(normals / _SQRT_2))
        cond = psi < psi_c
        
        psi_inv = 2 / psi
        b_squared = psi_inv - 1 + jnp.sqrt(psi_inv * (psi_inv - 1))
        
        a = m / (1 + b_squared)
        next_var_true = a * (jnp.sqrt(b_squared) + jnp.squeeze(normals)) ** 2
        
        p = (psi - 1) / (psi + 1)
        beta = (1 - p) / m
        next_var_false = jnp.where(uniforms > p, jnp.log(1 - p) - jnp.log(1 - uniforms),
                                   jnp.zeros_like(uniforms))/ beta
        
        
        next_vol = jnp.where(cond, next_var_true, next_var_false)
        
        
        return next_vol
    
    def _update_log_spot(self, mean_reversion, theta, volvol, rho,
                         current_vol, next_vol, current_log_spot, time_step, normals, 
                         gamma_1=0.5, gamma_2=0.5):
        k_0 = - rho * mean_reversion * theta / volvol * time_step
        k_1 = (gamma_1 * time_step
               * (mean_reversion * rho / volvol - 0.5)
               - rho / volvol)
        k_2 = (gamma_2 * time_step
               * (mean_reversion * rho / volvol - 0.5)
               + rho / volvol)
        k_3 = gamma_1 * time_step * (1 - rho ** 2)
        k_4 = gamma_2 * time_step * (1 - rho ** 2)
        
        next_log_spot = current_log_spot + k_0 + k_1 * current_vol + k_2 * next_vol + jnp.sqrt(k_3 * current_vol + k_4 * next_vol) * normals
        
        return next_log_spot
        
            
        
        
         
       
    