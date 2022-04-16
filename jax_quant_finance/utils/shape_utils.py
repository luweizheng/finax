
"""Utilities for shape manipulation."""

import jax.numpy as jnp

__all__ = [
    'get_shape',
    'broadcast_common_batch_shape',
    'common_shape'
]

def get_shape(x):
    x = jnp.asarray(x)
    is_fully_defined = jnp.all(x)
    if is_fully_defined:
        return x.shape
    return jnp.shape(x)
    

def common_shape(*args):
    """Returns common shape for a sequence of Tensors.

    The common shape is the smallest-rank shape to which all tensors are
    broadcastable.

    #### Example
    ```python
    import tensorflow as tf
    import tf_quant_finance as tff

    args = [tf.ones([1, 2], dtype=tf.float64), tf.constant([[True], [False]])]
    tff.utils.common_shape(*args)
    # Expected: [2, 2]
    ```

    Args:
    *args: A sequence of `Tensor`s of compatible shapes and any `dtype`s.
    name: Python string. The name to give to the ops created by this function.
        Default value: `None` which maps to the default name
        `broadcast_tensor_shapes`.

    Returns:
    A common shape for the input `Tensor`s, which an instance of TensorShape,
    if the input shapes are fully defined, or a `Tensor` for dynamically shaped
    inputs.

    Raises:
    ValueError: If inputs are of incompatible shapes.
    """
    is_fully_defined = True
    if args:
        for arg in args:
            arg = jnp.asarray(arg)
            is_fully_defined &= jnp.all(arg)
        if is_fully_defined:
            output_shape = args[0].shape
            for arg in args[1:]:
                try:
                    output_shape = jnp.broadcast_shapes(output_shape, arg.shape)
                except ValueError:
                    raise ValueError(f'Shapes of {args} are incompatible')
            output_shape = jnp.asarray(output_shape, dtype=jnp.int32)
            return output_shape
        output_shape = jnp.shape(args[0])
        for arg in args[1:]:
            output_shape = jnp.broadcast_shapes(output_shape, jnp.shape(arg))
        output_shape = jnp.asarray(output_shape, dtype=jnp.int32)    
        return output_shape
    

def broadcast_common_batch_shape(
    *args,
    event_ranks=None):
    """Broadcasts argument batch shapes to the common shape.

    Each input `Tensor` is assumed to be of shape `batch_shape_i + event_shape_i`.
    The function finds a common `batch_shape` and broadcasts each `Tensor` to
    `batch_shape + event_shape_i`. The common batch shape is the minimal shape
    such that all `batch_shape_i` can broadcast to it.

    #### Example 1. Batch shape is all dimensions but the last one
    ```python
    import tensorflow as tf
    import tf_quant_finance as tff

    # Two Tensors of shapes [2, 3] and [2]. The batch shape of the 1st Tensor is
    # [2] and for the second is []. The common batch shape is [2]
    args = [tf.ones([2, 3], dtype=tf.float64), tf.constant([True, False])]
    tff.utils.broadcast_common_batch_shape(*args)
    # Expected: (array([[1., 1., 1.], [1., 1., 1.]]),
    #            array([[True, True], [False, False]])
    ```

    #### Example 2. Specify ranks of event shapes
    ```python
    import tensorflow as tf
    import tf_quant_finance as tff

    args = [tf.ones([2, 3], dtype=tf.float64), tf.constant([True, False])]
    tff.utils.broadcast_common_batch_shape(*args,
                                            event_ranks)
    # Expected: (array([[1., 1., 1.], [1., 1., 1.]]),
    #            array([[True, True], [False, False]])
    ```

    Args:
    *args: A sequence of `Tensor`s of compatible shapes and any `dtype`s.
    event_ranks: A sequence of integers of the same length as `args` specifying
        ranks of `event_shape` for each input `Tensor`.
        Default value: `None` which means that all dimensions but the last one
        are treated as batch dimension.
    name: Python string. The name to give to the ops created by this function.
        Default value: `None` which maps to the default name
        `broadcast_tensor_shapes`.

    Returns:
    A tuple of broadcasted `Tensor`s. Each `Tensor` has the same `dtype` as the
    corresponding input `Tensor`.

    Raises:
    ValueError:
        (a) If `event_ranks` is supplied and is of different from `args` length.
        (b) If inputs are of incompatible shapes.
    """
    if event_ranks is None:
        event_ranks = [1] * len(args)
    if len(event_ranks) != len(args):
      raise ValueError(
          '`args` and `event_dims` should be of the same length but are {0} '
          'and {1} elements, respectively'.format(len(event_ranks), len(args)))
    dummies = [jnp.zeros(get_shape(arg)[:-d]) for arg, d in zip(args, event_ranks)]
    common_batch_shape = common_shape(*dummies)
    return tuple(jnp.broadcast_to(x, jnp.concatenate(
        [common_batch_shape, jnp.asarray(get_shape(x), dtype=jnp.int32)[-d:]], axis=0))
                 for x, d in zip(args, event_ranks))
