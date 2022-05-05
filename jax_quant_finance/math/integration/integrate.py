import enum
from jax_quant_finance.math.integration.simpson import simpson


@enum.unique
class IntegrationMethod(enum.Enum):
  """Specifies which algorithm to use for the numeric integration.

  * `COMPOSITE_SIMPSONS_RULE`: Composite Simpson's 1/3 rule.
  """
  COMPOSITE_SIMPSONS_RULE = 1


def integrate(func, 
              lower, 
              upper, 
              method=IntegrationMethod.COMPOSITE_SIMPSONS_RULE, 
              dtype=None,
              name=None,
              **kwargs):
    """Evaluates definite integral.

    #### Example
    ```python
        f = lambda x: x*x
        a = jnp.asarray(0.0)
        b = jnp.asarray(3.0)
        integrate(f, a, b) # 9.0
    ```

    Args:
        func: Python callable representing a function to be integrated. It must be a
        callable of a single `ndarray` parameter and return a `ndarray` of the same
        shape and dtype as its input. It will be called with a `Tesnor` of shape
        `lower.shape + [n]` (where n is integer number of points) and of the same
        `dtype` as `lower`.
        lower: `ndarray` or Python float representing the lower limits of
        integration. `func` will be integrated between each pair of points defined
        by `lower` and `upper`.
        upper: `ndarray` of the same shape and dtype as `lower` or Python float
        representing the upper limits of intergation.
        method: Integration method. Instance of IntegrationMethod enum. Default is
        IntegrationMethod.COMPOSITE_SIMPSONS_RULE.
        dtype: Dtype of result. Must be real dtype.

    Returns:
        `ndarray` of the same shape and dtype as `lower`, containing the value of the
        definite integral.

    Raises: ValueError if `method` was not recognized.

    TODO: Add Error control
    """            


    if method == IntegrationMethod.COMPOSITE_SIMPSONS_RULE:
        return simpson(func, lower, upper, dtype=dtype, **kwargs)
    else:
        raise ValueError('Unknown method: %s.' %method)

