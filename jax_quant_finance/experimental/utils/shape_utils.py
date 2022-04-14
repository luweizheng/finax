import jax.numpy as jnp


__all__ = [
    'get_shape',
    'broadcast_common_batch_shape',
    'common_shape'
]


def get_shape(x, name=None):
    x = jnp.asarray(x)
    is_fully_defined = jnp.all(x)
    if is_fully_defined:
        return x.shape
    return jnp.shape(x)
    

def common_shape(*args, 
                 name=None):
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
    event_ranks=None,
    name=None):
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
