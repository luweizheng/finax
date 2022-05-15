"""The Euler sampling method for ito processes."""

from typing import Callable, List, Optional, Union
from functools import partial

import numpy as np

from jax import lax, jit, vmap
import jax.numpy as jnp
from jax import random

from jax_quant_finance.math import random_sampler
from jax_quant_finance.models import utils


def sample(
    dim: int,
    drift_fn: Callable[..., jnp.ndarray],
    volatility_fn: Callable[..., jnp.ndarray],
    times: Union[jnp.ndarray, np.ndarray],
    seed: int = 0,
    time_step = None,
    num_time_steps = None,
    num_samples = 1,
    initial_state: Optional[Union[jnp.ndarray, np.ndarray]] = None,
    random_type: Optional[random_sampler.RandomType] = None,
    times_grid: Optional[Union[jnp.ndarray, np.ndarray]] = None,
    validate_args: bool = False,
    tolerance: Optional[Union[jnp.ndarray, np.ndarray]] = None,
    loop_method: str = "while",
    dtype: Optional[jnp.dtype] = None
    ) -> jnp.ndarray:
    """Returns a sample paths from the process using Euler method.

    For an Ito process,

    ```
        dX = a(t, X_t) dt + b(t, X_t) dW_t
        X(t=0) = x0
    ```
    with given drift `a` and volatility `b` functions Euler method generates a
    sequence {X_n} as

    ```
    X_{n+1} = X_n + a(t_n, X_n) dt + b(t_n, X_n) (N(0, t_{n+1}) - N(0, t_n)),
    X_0 = x0
    ```
    where `dt = t_{n+1} - t_n` and `N` is a sample from the Normal distribution.
    See [1] for details.

    #### Example
    Sampling from 2-dimensional Ito process of the form:

    ```none
    dX_1 = mu_1 * sqrt(t) dt + s11 * dW_1 + s12 * dW_2
    dX_2 = mu_2 * sqrt(t) dt + s21 * dW_1 + s22 * dW_2
    ```

    ```python
    import numpy as np
    import jax.numpy as jnp
    import jax_quant_finance as jqf

    mu = np.array([0.2, 0.7])
    a = np.array([[0.4, 0.1], [0.3, 0.2]])
    b = np.array([[0.33, -0.03], [0.21, 0.5]])
    num_samples = 10000
    dim = 2
    dtype = jnp.float64
    random_type = jqf.math.random_sampler.RandomType.STATELESS

    # Define drift and volatility functions
    def drift_fn(t, x):
        return drift = mu * jnp.sqrt(t) * jnp.ones_like(x, dtype=t.dtype)

    def vol_fn(t, x):
        return (a * t + b) * jnp.ones([2, 2], dtype=t.dtype)

    # Set starting location
    x0 = np.array([0.1, -1.1])
    # Sample `num_samples` paths at specified `times` using Euler scheme.
    times = np.array([0.1, 0.21, 0.32, 0.43, 0.55])
    paths = jqf.models.euler_sampling.sample(
                batch_size=1,
                dim=2,
                drift_fn=drift_fn, volatility_fn=vol_fn,
                times=times,
                num_samples=num_samples,
                initial_state=x0,
                time_step=0.01,
                random_type=random_type)
    
    # Expected: paths.shape = [5, 10000, 2]
    ```

    #### References
    [1]: Wikipedia. Euler-Maruyama method:
    https://en.wikipedia.org/wiki/Euler-Maruyama_method

    Args:
        batch_size: Python int greater than or equal to 1. Batch Size of independent 
            stochastic process we want to model.
        dim: Python int greater than or equal to 1. The dimension of the Ito
            Process.
        drift_fn: A Python callable to compute the drift of the process. The
            callable should accept two real `ndarray` arguments of the same dtype.
            The first argument is the scalar time t.
            The second argument is the value of Ito process X which has shape
            `(num_samples, dim)`. 
            The result is value of a(t, X).
            The return value of the callable is a real `ndarray` of the same dtype 
            as the input arguments and of shape `(num_samples, dim)`.
        volatility_fn: A Python callable to compute the volatility of the process.
            The callable should accept two real `ndarray` arguments of the same dtype
            and shape `times_shape`. 
            The first argument is the scalar time t.
            The second argument is the value of Ito process X which has shape
            `(num_samples, dim)`. 
            The result is value of b(t, X).
            The return value of the callable is a real `ndarray` of the same dtype as
            the input arguments and of shape `(num_samples, dim)`.
        times: Rank 1 `ndarray` of increasing positive real values. The times at
            which the path points are to be evaluated.
        time_step: An optional scalar real `ndarray` - maximal distance between
            points in grid in Euler schema.
            Either this or `num_time_steps` should be supplied.
            Default value: `None`.
        num_time_steps: An optional Scalar integer `ndarray` - a total number of time
            steps performed by the algorithm. The maximal distance betwen points in
            grid is bounded by `times[-1] / (num_time_steps - times.shape[0])`.
            Either this or `time_step` should be supplied.
            Default value: `None`.
        num_samples: Positive scalar `int`. The number of paths to draw.
            Default value: 1.
        initial_state: `Tensor` of shape broadcastable with
            `batch_size + (num_samples, dim)`. The initial state of the process.
            `batch_size` represents the shape of the independent batches of the
            stochastic process. 
            Default value: None which maps to a zero initial state.
        random_type: Enum value of `RandomType`. The type of (quasi)-random
            number generator to use to generate the paths.
            Default value: None which maps to the JAX's stateless pseudo-random numbers.
        seed: Seed for the random number generator. The seed is
        only relevant if `random_type` is one of
            `[STATELESS, STATELESS_ANTITHETIC]`. 
            Default value: `None` which means no seed is set.
        times_grid: An optional rank 1 `ndarray` representing time discretization
            grid. If `times` are not on the grid, then the nearest points from the
            grid are used. When supplied, `num_time_steps` and `time_step` are
            ignored.
            Default value: `None`, which means that times grid is computed using
            `time_step` and `num_time_steps`.
        validate_args: Python `bool`. When `True` performs multiple checks:
            * That `times`  are increasing with the minimum increments of the
                specified tolerance.
            When `False` invalid dimension may silently render incorrect outputs.
            Default value: `False`.
        tolerance: A non-negative scalar `Tensor` specifying the minimum tolerance
            for discernible times on the time grid. Times that are closer than the
            tolerance are perceived to be the same.
            Default value: `None` which maps to `1-e6` if the for single precision
                `dtype` and `1e-10` for double precision `dtype`.
        dtype: Optional `jnp.dtype`. If supplied, the dtype to be used for 
            `ndarray` data type conversion. 
            Default value: `None`. which means that the dtype implied by `times` is used.

    Returns:
        A real `ndarray` of shape (batch_size, len(times), num_samples, dim) 
        where `len(times)` is the size of the `times`.

    Raises:
        ValueError:
            When `times_grid` is not supplied, and neither `num_time_steps` nor
            `time_step` are supplied or if both are supplied.
    """

    times = jnp.asarray(times, dtype=dtype)
    dtype = times.dtype
    
    if tolerance is None:
        tolerance = 1e-10 if dtype == jnp.float64 else 1e-6
    tolerance = jnp.asarray(tolerance, dtype=dtype)
    
    if validate_args:
        assert (jnp.all(times[1:] <= times[:-1] + tolerance)), f"`times` increments should be greater than tolerance {tolerance}"
    
    if initial_state is None:
        initial_state = jnp.zeros(dim, dtype=dtype)
    initial_state = jnp.asarray(initial_state, dtype=dtype)

    # the last axis of `initial_state` is the dim of the process
    batch_shape = initial_state.shape[:-1]
    if len(batch_shape) > 1:
        raise ValueError(
            'We currently do not support multiple batch shape of `initial_state`.'
            'The shape of  `initial_state` should be (batch_size, dim) or (dim, ).'
            )
    elif len(batch_shape) == 0:
        # if there is no batch shape in `initial_state`, add a batch axis in initial_state
        jnp.expand_dims(initial_state, axis=0)
    batch_size = initial_state.shape[0]

    # compute the value of every element in times
    num_requested_times = times.shape[0]
    
    # Create a time grid for the Euler scheme.
    if num_time_steps is not None and time_step is not None:
        raise ValueError(
            'When `times_grid` is not supplied only one of either '
            '`num_time_steps` or `time_step` should be defined but not both.')
    if times_grid is None:
        if time_step is None:
            if num_time_steps is None:
                raise ValueError(
                    'When `times_grid` is not supplied, either `num_time_steps` '
                    'or `time_step` should be defined.')
            num_time_steps = jnp.asarray(num_time_steps, dtype=dtype)
            time_step = times[-1] / num_time_steps
        else:
            time_step = jnp.asarray(time_step, dtype=dtype)
    else:
        times_grid = jnp.asarray(times_grid, dtype=dtype)
        if validate_args:
            assert (jnp.all(times_grid[1:] > times_grid[:-1] + tolerance)), f"`times_grid` increments should be greater than tolerance {tolerance}"

    # `times` is the times path, which is a 1-D ndarray, 
    #     ranging from 0 to times[-1]] with `num_time_steps` elements
    # `keep_masks` has same shape with `times`, it is the mask 
    #     indicating which time step should be evaluated as a result, 
    #     0 is for false, 1 is for true
    times, keep_mask, time_indices = utils.prepare_grid(
        times=times,
        time_step=time_step,
        num_time_steps=num_time_steps,
        times_grid=times_grid,
        tolerance=tolerance,
        dtype=dtype)

    keys = random.PRNGKey(seed)
    print(batch_size)
    print(f"initial_state shape: {initial_state.shape}")
    keys = random.split(keys, num=batch_size)
    sample_fn = vmap(_sample, in_axes=(0, None, None, None, None, None, None, None, 0, None, None, None))

    paths = sample_fn(keys, 
        dim, 
        drift_fn, 
        volatility_fn, 
        times, 
        keep_mask, 
        num_requested_times, 
        num_samples, 
        initial_state, 
        random_type, 
        dtype, 
        loop_method)
    
    # (batch_size, len(times), num_samples, dim) -> (batch_size, num_samples, len(times), dim)
    paths = paths.transpose((0, 2, 1, 3))
    return paths

    # if batch_size > 1:
    #     keys = random.split(keys, num=batch_size)
    #     sample_fn = vmap(_sample, in_axes=(0, None, None, None, None, None, None, None, 0, None, None, None))

    #     return sample_fn(keys, dim, drift_fn, volatility_fn, times, keep_mask, num_requested_times, num_samples, initial_state, random_type, dtype, loop_method)
    # else:
    #     sample_fn = _sample
    #     return sample_fn(
    #         seed=keys,
    #         dim=dim,
    #         drift_fn=drift_fn,
    #         volatility_fn=volatility_fn,
    #         times=times,
    #         keep_mask=keep_mask,
    #         num_requested_times=num_requested_times,
    #         num_samples=num_samples,
    #         initial_state=initial_state,
    #         random_type=random_type,
    #         dtype=dtype,
    #         loop_method=loop_method)


def _sample(seed,
            dim,
            drift_fn,
            volatility_fn,
            times,
            keep_mask,
            num_requested_times,
            num_samples,
            initial_state,
            random_type,
            dtype,
            loop_method):
    """Returns a sample of paths from the process using Euler method."""
    dt = times[1:] - times[:-1]
    sqrt_dt = jnp.sqrt(dt)
    current_state = jnp.broadcast_to(initial_state, shape=(num_samples, dim)) + jnp.zeros((num_samples, dim), dtype=dtype)
    steps_num = dt.shape[-1]
    wiener_mean = None
    
    # TODO: multivariate
    # STATELESS random normal
    if random_type in [None, random_sampler.RandomType.STATELESS_ANTITHETIC]:
        normal_draws_shape = (steps_num, num_samples // 2, dim)
        normal_draws = random.normal(key=seed,
                shape=normal_draws_shape,
                dtype=dtype)
        normal_draws = jnp.concatenate([normal_draws, -normal_draws], axis=1)
    elif random_type == random_sampler.RandomType.STATELESS:
        normal_draws_shape = (steps_num, num_samples, dim)
        normal_draws = random.normal(key=seed,
                shape=normal_draws_shape,
                dtype=dtype)
    # low-discrepancy random_type
    
    # create a ndarray to store all these results
    element_shape = current_state.shape
    result = jnp.zeros((num_requested_times, ) + element_shape, dtype=dtype)

    if loop_method == 'while':
        while_loop_fn = jit(_while_loop, static_argnames=['drift_fn', 'volatility_fn', 'dtype', 'random_type'])
        return while_loop_fn(steps_num=steps_num,
            current_state=current_state,
            drift_fn=drift_fn, volatility_fn=volatility_fn, wiener_mean=wiener_mean,
            num_samples=num_samples, times=times,
            dt=dt, sqrt_dt=sqrt_dt, keep_mask=keep_mask,
            num_requested_times=num_requested_times,
            random_type=random_type, result=result, normal_draws=normal_draws)
    elif loop_method == 'scan':
        scan_fn = jit(_scan, static_argnames=['drift_fn', 'volatility_fn', 'dtype', 'random_type'])
        return scan_fn(steps_num=steps_num,
            current_state=current_state,
            drift_fn=drift_fn, volatility_fn=volatility_fn, wiener_mean=wiener_mean,
            num_samples=num_samples, times=times,
            dt=dt, sqrt_dt=sqrt_dt, keep_mask=keep_mask,
            num_requested_times=num_requested_times,
            random_type=random_type, result=result, normal_draws=normal_draws)


def _while_loop(*, steps_num, current_state,
                drift_fn, volatility_fn, wiener_mean,
                num_samples, times, dt, sqrt_dt, num_requested_times,
                keep_mask, random_type, result, normal_draws):
    """Sample paths using `lax.while_loop`."""
    written_count = 0
    result = result.at[written_count].set(current_state)
    written_count = lax.add(written_count, jnp.asarray(keep_mask[0]))
    
    # Define sampling while_loop body function
    def cond_fn(state):
        i, written_count, _, _ = state
        return jnp.logical_and(i < steps_num, written_count < num_requested_times)

    def step_fn(state):
        """Performs one step of Euler scheme.
        X_{n+1} = X_{n} + drift_fn() * dt + volatility_fn() * dW
        """
        i, written_count, current_state, result = state
        
        # the first element of `times` is 0 
        current_time = times[i + 1]

        dw = normal_draws[i]
        dw = dw * sqrt_dt[i]

        dt_inc = dt[i] * drift_fn(current_time, current_state)
        dw_inc = jnp.einsum('...ij,...j->...i', volatility_fn(current_time, current_state), dw)
        
        next_state = current_state + dt_inc + dw_inc
        result = result.at[written_count].set(next_state)
        written_count = lax.add(written_count, keep_mask[i + 1])

        return i+1, written_count, next_state, result
    
    # result of sample paths
    # result shape: (num_requested_times, num_samples, dim)
    _, _, _, result = lax.while_loop(cond_fun=cond_fn, 
            body_fun=step_fn, 
            init_val=(0, written_count, current_state, result))
   
    return result


def _scan(*, steps_num, current_state,
                drift_fn, volatility_fn, wiener_mean,
                num_samples, times, dt, sqrt_dt, num_requested_times,
                keep_mask, random_type, result, normal_draws):
    """Sample paths using `lax.scan`."""
    written_count = 0
    result = result.at[written_count].set(current_state)
    written_count = lax.add(written_count, jnp.asarray(keep_mask[0]))

    def step_fn(carry, normal_draw):
        """Performs one step of Euler scheme.
        X_{n+1} = X_{n} + drift_fn() * dt + volatility_fn() * dW
        """
        
        i, written_count, current_state, result = carry
        current_time = times[i + 1]
        dw = normal_draw * sqrt_dt[i]

        dt_inc = dt[i] * drift_fn(current_time, current_state)
        dw_inc = jnp.einsum('...ij,...j->...i', volatility_fn(current_time, current_state), dw)
        
        next_state = current_state + dt_inc + dw_inc
        result = result.at[written_count].set(next_state)
        written_count = lax.add(written_count, keep_mask[i + 1])
        return (i + 1, written_count, next_state, result), None
    
    # result of sample paths
    # result shape: (num_requested_times, num_samples, dim)
    (_, _, _, result), _ = lax.scan(f=step_fn,
            init=(0, written_count, current_state, result),
            xs=normal_draws
        )
   
    return result


def _euler_step(*, i, written_count, current_state,
                drift_fn, volatility_fn, wiener_mean,
                num_samples, times, dt, sqrt_dt, keep_mask,
                normal_draws, result):
    """Performs one step of Euler scheme.
    X_{n+1} = X_{n} + drift_fn() * dt + volatility_fn() * dW
    """
    # the first element of `times` is 0 
    current_time = times[i + 1]

    dw = normal_draws[i]
    dw = dw * sqrt_dt[i]

    dt_inc = dt[i] * drift_fn(current_time, current_state)
    dw_inc = jnp.einsum('...ij,...j->...i', volatility_fn(current_time, current_state), dw)
    
    next_state = current_state + dt_inc + dw_inc
    result = result.at[written_count].set(next_state)
    written_count = lax.add(written_count, keep_mask[i + 1])

    return i + 1, written_count, next_state, result
