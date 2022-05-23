# Finax: High Performance Quantative Finance on JAX

## Introduction

We developed a quantative finance library based on [JAX](https://github.com/google/jax). This library provides high-performance financial computation leveraging the hardware
acceleration, parallel scientific computing and automatic differentiation of JAX.

Finax is still a research project that is currently under development. Some APIs may change in the future. We welcome any suggestions and contributions.

We can:

* run financial workloads on CPU/GPU/TPU with XLA acceleration

* calculate mathematical derivative of financial models, i.e. Greeks

* distribute workloads on multiple devices

`examples` directory contains several demonstrations of using our Finax library.

## Install

### JAX installation

You must first follow [JAX's installation guide](https://github.com/google/jax/#installation) to install JAX based on your device architecture (CPU/GPU/TPU).

### Finax

```
pip install finax --upgrade
```

## Usage

### Getting-started

Here is an example for option pricing using [Black-Scholes model](https://en.wikipedia.org/wiki/Black-Scholes_model).

```
import numpy as np
from jax import jit

from jax.config import config
config.update("jax_enable_x64", True)

from finax.black_sholes.vanilla_prices import option_price

option_price_fn = jit(option_price)

dtype = jnp.float64
forwards = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
strikes = jnp.array([3.0, 3.0, 3.0, 3.0, 3.0], dtype=dtype)
volatilities = jnp.array([0.0001, 102.0, 2.0, 0.1, 0.4], dtype=dtype)
expiries = jnp.array(1.0, dtype=dtype)

computed_prices = option_price_fn(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards)
```

### 64-bit precision

To enable 64-bit precision, set the respective JAX flag _before_ importing `finax` (see the JAX [guide](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision)), for example:

```python
from jax.config import config
config.update("jax_enable_x64", True)
```

