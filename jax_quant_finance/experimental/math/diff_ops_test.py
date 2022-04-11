from jax_quant_finance.experimental.math.diff_ops import diff
import numpy as np
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
import numpy as np

x = jnp.asarray([1,2,3,4,5])
order = 2
dx = diff(x, order=order, exclusive=True)
# _x = np.array(x[::])
# slices = 1 * [slice(None)]
# slices[-1] = slice(None, 2)
# print(_x[slices])

# print(x[order::order]-x[:-order:order])
# print(x.at[0])

print(dx)