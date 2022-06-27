import jax.numpy as jnp
from finax.math import random_sampler
from jax import lax

def generate_mc_normal_draws(num_normal_draws,
                             num_time_steps,
                             num_sample_paths,
                             random_type,
                             batch_shape = None,
                             key=None,
                             dtype=None):
    
    dtype = dtype or jnp.float32
    
    if batch_shape is None:
        batch_shape = jnp.asarray([], dtype=jnp.int32)
        
    total_dimensions = jnp.zeros([num_time_steps * num_normal_draws], dtype=dtype)
    
    if random_type in [random_sampler.RandomType.STATELESS_ANTITHETIC]:
        sample_shape = jnp.concatenate([jnp.asarray([num_sample_paths], dtype=jnp.int32), batch_shape], axis=0)
        is_antithetic = True
    
    
    else:
        sample_shape = jnp.concatenate([batch_shape, jnp.asarray([num_sample_paths], dtype=jnp.int32)], axis=0)
        is_antithetic = False
    
    normal_draws = random_sampler.normal_pseudo(
        sample_shape,
        mean = total_dimensions,
        key=key,
        dtype=dtype
    )
    
    normal_draws = jnp.reshape(
        normal_draws,
        jnp.concatenate([sample_shape, jnp.asarray([num_time_steps, num_normal_draws], dtype=jnp.int32)], axis=0))
    
    normal_draws_shape_rank = len(normal_draws.shape)
    if is_antithetic and normal_draws_shape_rank > 3:
        perm = [normal_draws_shape_rank - 2] + list(range(1, normal_draws_shape_rank - 2)) + [0, normal_draws_shape_rank - 1]
    else:
        perm = [normal_draws_shape_rank - 2] +list(range(normal_draws_shape_rank - 2)) + [normal_draws_shape_rank-1]
    
    normal_draws = lax.transpose(normal_draws, permutation=perm)
    return normal_draws