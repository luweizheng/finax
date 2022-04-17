import jax.numpy as jnp
from jax_quant_finance.utils import shape_utils


__all__ = [
    'interpolate',
]


def interpolate(x, 
                x_data, 
                y_data, 
                left_slope=None, 
                right_slope=None, 
                validate_args=False, 
                optimize_for_tpu=False, 
                dtype=None, 
                name=None):
    
    """Performs linear interpolation for supplied points
    
    
    """
    
    x = jnp.asarray(x, dtype=dtype)
    dtype = dtype or x.dtype
    x_data = jnp.asarray(x_data, dtype=dtype)
    y_data = jnp.asarray(y_data, dtype=dtype)
    # Try broadcast batch shapes
    x, x_data, y_data = shape_utils.broadcast_common_batch_shape(x, x_data, y_data)
    
    # Rank of the inputs is known
    # tf.rank -> jnp.ndim
    batch_rank = jnp.ndim(x) - 1
    if batch_rank == 0:
        x = jnp.expand_dims(x, 0)
        x_data = jnp.expand_dims(x_data, 0)
        y_data = jnp.expand_dims(y_data, 0)
    
    if left_slope is None:
        left_slope = jnp.asarray(0.0, dtype=x.dtype)
    else:
        left_slope = jnp.asarray(left_slope, dtype=x.dtype)
    
    if right_slope is None:
        right_slope = jnp.asarray(0.0, dtype=x.dtype)
    else:
        right_slope = jnp.asarray(right_slope, dtype=x.dtype)
        
    control_deps = []
    if validate_args:
        # TODO : ADD Error Controls
        # Check that `x_data` elements is non-decreasing
        diffs = x_data[..., 1:] - x_data[..., :-1]
        pass

    upper_indices = jnp.searchsorted(x_data, x, side='left')
    upper_indices = jnp.asarray(upper_indices, dtype=jnp.int32)
    x_data_size = shape_utils.get_shape(x_data)[-1]
    at_min = jnp.equal(upper_indices, 0)
    at_max = jnp.equal(upper_indices, x_data_size)
    values_min = jnp.expand_dims(y_data[..., 0], -1) + left_slope * (
        x - jnp.broadcast_to(
            jnp.expand_dims(x_data[..., 0], -1),
            shape=shape_utils.get_shape(x)))
    values_max = jnp.expand_dims(y_data[..., -1], -1) + right_slope * (
        x - jnp.broadcast_to(
            jnp.expand_dims(x_data[..., -1], -1),
            shape=shape_utils.get_shape(x)))   
    
    # won't go out of bounds.
    lower_encoding = jnp.maximum(upper_indices - 1 , 0)
    upper_encoding = jnp.minimum(upper_indices, x_data_size - 1)
    
    # TODO : Finish get_slice

    def get_slice(x, encoding):
        # TODO : impl gather func with axis and batch_dims
        if optimize_for_tpu: # not finish
            pass
        return 

            
    
    
