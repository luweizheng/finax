import enum

import numpy as np


_SQRT_2 = np.sqrt(2.)

@enum.unique
class RandomType(enum.Enum):
    """Types of random number sequences.

    * `PSEUDO`: The standard MindSpore random generator.
    * `PSEUDO_ANTITHETIC`: PSEUDO random numbers along with antithetic variates.
    * `HALTON`: The standard Halton sequence.
    * `HALTON_RANDOMIZED`: The randomized Halton sequence.
    * `SOBOL`: The standard Sobol sequence.

    """

    PSEUDO = 0
    PSEUDO_ANTITHETIC = 1
    HALTON = 2
    HALTON_RANDOMIZED = 3
    SOBOL = 4