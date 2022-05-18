import enum

@enum.unique
class ParallelType(enum.Enum):
    """Types of Parallel Computing.

    * `SINGLE_DEVICE`: Single Device.
    * `MULTI_DEVICE`:  Multiple Devices on single host.

    """

    SINGLE_DEVICE = 0
    MULTI_DEVICE = 1