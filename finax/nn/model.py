

import time
from functools import partial

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrandom

from flax.training.train_state import TrainState
from flax import struct

EquationProblemDef = Any

class Model(struct.PyTreeNode):
    train_state: TrainState
    equ_problem: EquationProblemDef
    batch_size: int = struct.field(pytree_node=False) # batch size or number of trajectories