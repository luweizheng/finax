"""Defines class to describe any Ito processes.

Uses Euler scheme for sampling
"""
from finax.models import ito_process, euler_sampling
from finax.utils import ParallelType, LoopType


class GenericItoProcess(ito_process.ItoProcess):
    """Generic Ito process defined from a drift and volatility function."""

    def __init__(self, dim, drift_fn, volatility_fn, dtype=None):
        """Initializes the Ito process with given drift and volatility functions.

        Represents a general Ito process:

        ```None
        dX_i = a_i(t, X) dt + Sum(S_{ij}(t, X) dW_j for 1 <= j <= n), 1 <= i <= n
        ```

        The vector coefficient `a_i` is referred to as the drift of the process and
        the matrix `b_{ij}` as the volatility of the process. For the process to be
        well defined, these coefficients need to satisfy certain technical
        conditions which may be found in Ref. [1]. The vector `dW_j` represents
        independent Brownian increments.

        #### Example. Sampling from 2-dimensional Ito process of the form:

        ```none
        dX_1 = mu_1 * sqrt(t) dt + s11 * dW_1 + s12 * dW_2
        dX_2 = mu_2 * sqrt(t) dt + s21 * dW_1 + s22 * dW_2
        ```

        ```python
        mu = np.array([0.2, 0.7])
        s = np.array([[0.3, 0.1], [0.1, 0.3]])
        num_samples = 10000
        dim = 2
        dtype=tf.float64

        # Define drift and volatility functions
        def drift_fn(t, x):
        return mu * tf.sqrt(t) * tf.ones([num_samples, dim], dtype=dtype)

        def vol_fn(t, x):
        return s * tf.ones([num_samples, dim, dim], dtype=dtype)

        # Initialize `GenericItoProcess`
        process = GenericItoProcess(dim=2, drift_fn=drift_fn, volatility_fn=vol_fn,
                                    dtype=dtype)
        # Set starting location
        x0 = np.array([0.1, -1.1])
        # Sample `num_samples` paths at specified `times` locations using built-in
        # Euler scheme.
        times = [0.1, 1.0, 2.0]
        paths = process.sample_paths(
                times,
                num_samples=num_samples,
                initial_state=x0,
                time_step=0.01,
                seed=42)
        ```

        #### References
        [1]: Brent Oksendal. Stochastic Differential Equations: An Introduction with
        Applications. Springer. 2010.

        Args:
        dim: Python int greater than or equal to 1. The dimension of the Ito
            process.
        drift_fn: A Python callable to compute the drift of the process. The
            callable should accept two real `ndarray` arguments of the same dtype.
            The first argument is the scalar time t, the second argument is the
            value of Ito process X - `ndarray` of shape
            `batch_shape + sample_shape + [dim]`, where `batch_shape` represents
            a batch of models and `sample_shape` represents samples for each of the
            models. The result is value of drift a(t, X). The return value of the
            callable is a real `ndarray` of the same dtype as the input arguments and
            of shape `batch_shape + sample_shape + [dim]`. For example,
            `sample_shape` can stand for `[num_samples]` for Monte Carlo sampling,
            or `[num_grid_points_1, ..., num_grid_points_dim]` for Finite Difference
            solvers.
        volatility_fn: A Python callable to compute the volatility of the process.
            The callable should accept two real `ndarray` arguments of the same dtype
            and shape `times_shape`. The first argument is the scalar time t, the
            second argument is the value of Ito process X - `ndarray` of shape
            `batch_shape + sample_shape + [dim]`, where `batch_shape` represents
            a batch of models and `sample_shape` represents samples for each of the
            models. The result is value of volatility S_{ij}(t, X). The return value
            of the callable is a real `ndarray` of the same dtype as the input
            arguments and of shape `batch_shape + sample_shape + [dim, dim]`. For
            example, `sample_shape` can stand for `[num_samples]` for Monte Carlo
            sampling, or `[num_grid_points_1, ..., num_grid_points_dim]` for Finite
            Difference solvers.
        dtype: The default dtype to use when converting values to `ndarray`s.
            Default value: None which means that default dtypes inferred.
        name: str. The name scope under which ops created by the methods of this
            class are nested.
            Default value: None which maps to the default name
            `generic_ito_process`.

        Raises:
            ValueError if the dimension is less than 1, or if either `drift_fn`
                or `volatility_fn` is not supplied.
        """
        if dim < 1:
            raise ValueError('Dimension must be 1 or greater.')
        if drift_fn is None or volatility_fn is None:
            raise ValueError('Both drift and volatility functions must be supplied.')
        self._dim = dim
        self._drift_fn = drift_fn
        self._volatility_fn = volatility_fn
        self._dtype = dtype

    def dim(self):
        """The dimension of the process."""
        return self._dim

    def dtype(self):
        """The data type of process realizations."""
        return self._dtype

    def drift_fn(self):
        """Python callable calculating instantaneous drift.

        The callable should accept two real `Tensor` arguments of the same dtype.
        The first argument is the scalar time t, the second argument is the value of
        Ito process X - `Tensor` of shape
        `batch_shape + sample_shape + [dim]`, where `batch_shape` represents a batch
        of models and `sample_shape` represents samples for each of the models. The
        result is value of drift a(t, X). The return value of the callable is a real
        `Tensor` of the same dtype as the input arguments and of shape
        `batch_shape + sample_shape + [dim]`. For example, `sample_shape` can stand
        for `[num_samples]` for Monte Carlo sampling, or
        `[num_grid_points_1, ..., num_grid_points_dim]` for Finite Difference
        solvers.

        Returns:
        The instantaneous drift rate callable.
        """
        return self._drift_fn

    def volatility_fn(self):
        """Python callable calculating the instantaneous volatility.

        The callable should accept two real `Tensor` arguments of the same dtype and
        shape `times_shape`. The first argument is the scalar time t, the second
        argument is the value of Ito process X - `Tensor` of shape
        `batch_shape + sample_shape + [dim]`, where `batch_shape` represents a batch
        of models and `sample_shape` represents samples for each of the models. The
        result is value of volatility S_{ij}(t, X). The return value of the callable
        is a real `Tensor` of the same dtype as the input arguments and of shape
        `batch_shape + sample_shape + [dim, dim]`. For example, `sample_shape` can
        stand for `[num_samples]` for Monte Carlo sampling, or
        `[num_grid_points_1, ..., num_grid_points_dim]` for Finite Difference
        solvers.

        Returns:
        The instantaneous volatility callable.
        """
        return self._volatility_fn

    def sample_paths(self,
                    times,
                    num_samples=1,
                    initial_state=None,
                    random_type=None,
                    seed=0,
                    loop_method=LoopType.WHILE,
                    parallel_mode=ParallelType.SINGLE_DEVICE,
                    time_step=None,
                    num_time_steps=None,
                    times_grid=None,
                    validate_args=False,
                ):
        """Returns a sample of paths from the process using Euler sampling.

        The default implementation uses the Euler scheme. However, for particular
        types of Ito processes more efficient schemes can be used.

        Args:
        times: Rank 1 `Tensor` of increasing positive real values. The times at
            which the path points are to be evaluated.
        num_samples: Positive scalar `int`. The number of paths to draw.
            Default value: 1.
        initial_state: `Tensor` of shape broadcastable
            `batch_shape + [num_samples, dim]`. The initial state of the process.
            `batch_shape` represents the shape of the independent batches of the
            stochastic process as in the `drift_fn` and `volatility_fn` of the
            underlying class. Note that the `batch_shape` is inferred from
            the `initial_state` and hence when sampling is requested for a batch of
            stochastic processes, the shape of `initial_state` should be as least
            `batch_shape + [1, 1]`.
            Default value: None which maps to a zero initial state.
        random_type: Enum value of `RandomType`. The type of (quasi)-random number
            generator to use to generate the paths.
            Default value: None which maps to the standard pseudo-random numbers.
        seed: Seed for the random number generator. The seed is
            only relevant if `random_type` is one of
            `[STATELESS, STATELESS_ANTITHETIC]`.
            Default value: 0.
        loop_method: An enum to specify which loop method to use: 
            * `finax.utils.LoopType.WHILE` for `jax.lax.while_loop`
            `finax.utils.LoopType.SCAN` for `jax.lax.scan`
            Default value: `LoopType.WHILE`.
        parallel_mode: An enum to specify which parallelization mode to use: 
            * `finax.utils.ParallelType.SINGLE_DEIVCE` for single device evaluation.
            * `finax.utils.ParallelType.MULTI_DEIVCE` for single host multiple devices evaluation.
            Default value: `ParallelType.SINGLE_DEIVCE`.
        time_step: An optional scalar real - maximal distance between
            points in the time grid.
            Either this or `num_time_steps` should be supplied.
            Default value: `None`.
        num_time_steps: An optional Scalar integer - a total number of
            time steps performed by the algorithm. The maximal distance between
            points in grid is bounded by
            `times[-1] / (num_time_steps - times.shape[0])`.
            Either this or `time_step` should be supplied.
            Default value: `None`.
        times_grid: An optional rank 1 `ndarray` representing time discretization
            grid. If `times` are not on the grid, then the nearest points from the
            grid are used.
            Default value: `None`, which means that times grid is computed using
            `time_step` and `num_time_steps`.
        validate_args: Python `bool`. When `True` performs multiple checks:
            * That `times`  are increasing with the minimum increments of the
                specified tolerance.
            When `False` invalid dimension may silently render incorrect outputs.
            Default value: `False`.

        Returns:
            A real `ndarray` of shape (batch_size + num_samples, len(times), dim).

        Raises:
            ValueError:
                (1) When `times_grid` is not supplied, and neither `num_time_steps` nor
                `time_step` are supplied or if both are supplied.
        """
        return euler_sampling.sample(
            dim=self._dim,
            drift_fn=self._drift_fn,
            volatility_fn=self._volatility_fn,
            times=times,
            seed=seed,
            time_step=time_step,
            num_time_steps=num_time_steps,
            num_samples=num_samples,
            initial_state=initial_state,
            random_type=random_type,
            times_grid=times_grid,
            validate_args=validate_args,
            loop_method=loop_method,
            parallel_mode=parallel_mode,
            dtype=self._dtype
        )