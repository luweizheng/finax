import enum

@enum.unique
class LoopType(enum.Enum):
    """Types of Loop.

    * `WHILE`: `jax.lax.while_loop`.
    * `SCAN`: `jax.lax.scan`.

    """

    WHILE = 0
    SCAN = 1