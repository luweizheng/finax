from jax_quant_finance.experimental.math.intergration.integrate import integrate
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

func = lambda x: jnp.exp(2 * x +1)
lower=jnp.asarray([1.0,2.0])
upper=jnp.asarray([3.0,4.0])
dtype = jnp.float64

antiderivate_func = lambda x: jnp.exp(2*x+1)/2

sim = integrate(func, lower, upper, num_points=10001 ,dtype=dtype)
ans = antiderivate_func(upper)-antiderivate_func(lower)

print(sim)

print(ans)