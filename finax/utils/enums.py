import enum

@enum.unique
class LoopType(enum.Enum):
    """Types of Loop.

    * `WHILE`: `jax.lax.while_loop`.
    * `SCAN`: `jax.lax.scan`.

    """

    WHILE = 0
    SCAN = 1


@enum.unique
class ParallelType(enum.Enum):
    """Types of Parallel Computing.

    * `SINGLE_DEVICE`: Single Device.
    * `MULTI_DEVICE`:  Multiple Devices on single host.

    """

    SINGLE_DEVICE = 0
    MULTI_DEVICE = 1