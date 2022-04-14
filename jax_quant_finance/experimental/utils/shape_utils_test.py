import numpy as np
import jax.numpy as jnp
from jax import config
from jax_quant_finance.experimental.utils.shape_utils import get_shape, common_shape, broadcast_common_batch_shape
config.update("jax_enable_x64", True)

# test get shape
x = [[1],[2]]
s = get_shape(x)
print(s)


# test common shape
args = [jnp.ones([1, 2], dtype=jnp.float64),
        jnp.asarray([[True], [False]], dtype=jnp.bool_),
        jnp.zeros([1], dtype=jnp.float32)]


s = common_shape(*args)
print(s)


# test broadcast common shape batch
x = jnp.zeros([3, 4])
y = jnp.zeros([2, 1, 3, 10])
z = jnp.zeros([])
x, y, z = broadcast_common_batch_shape(x, y, z)

s1 = np.array_equal(x, np.zeros([2, 1, 3, 4]))
      
print(s1)

s2 = np.array_equal(y, np.zeros([2, 1, 3, 10]))
      
print(s2)

s3 = np.array_equal(z, np.zeros([2, 1, 3]))
      
print(s3)










