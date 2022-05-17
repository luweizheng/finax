import jax_quant_finance as jqf
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import numpy as np

time_step = 0.0001
u = jnp.asarray([1, 2, -1, -2], dtype=jnp.float64)
matrix = jnp.asarray(
    [[1, -1, 0, 0], [3, 1, 2, 0], [0, -2, 1, 4], [0, 0, 3, 1]],
    dtype=jnp.float64)


def _convert_to_tridiagonal_format(matrix):
    matrix_np = np.asarray(matrix)
    n = matrix_np.shape[0]
    superdiag = [matrix_np[i, i + 1] for i in range(n - 1)] + [0]
    diag = [matrix_np[i, i] for i in range(n)]
    subdiag = [0] + [matrix_np[i + 1, i] for i in range(n - 1)]
    return tuple(
    jnp.asarray(v, dtype=matrix.dtype) for v in (diag, superdiag, subdiag))
    
tridiag_form = _convert_to_tridiagonal_format(matrix)


explicit_scheme = jqf.experimental.pde.steppers.explicit.explicit_scheme


actual = explicit_scheme(u, 0, time_step, lambda t: (tridiag_form, None))

print(actual)

    

