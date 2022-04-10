import jax.numpy as jnp
from jax.config import config
from jax.ops import segment_sum
config.update("jax_enable_x64", True)

"""
no use, just test some ops or lax functions
"""

data = jnp.asarray([1.0,2.0,3.0,4.0], dtype=jnp.float64)
segment_ids = jnp.asarray([0,0,0,1], dtype=jnp.int32)
# res = segment_sum(data, segment_ids)
# print(res)
pv = 995.50315587
pv = jnp.asarray([pv,pv,pv,pv], dtype=jnp.float64)

print(pv[segment_ids,...])