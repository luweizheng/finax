import numpy as np
import jax.numpy as jnp
from jax import jit

import jax_quant_finance as jqf

from absl.testing import absltest
from absl.testing import parameterized

from jax._src import test_util as jtu

from jax.config import config
config.update("jax_enable_x64", True)

class VanillaPrice(jtu.JaxTestCase):
    def test_option_prices(self):
        """Tests that the BS prices are correct."""
        forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        strikes = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])
        expiries = 1.0
        computed_prices = jqf.black_scholes.option_price(
                volatilities=volatilities,
                strikes=strikes,
                expiries=expiries,
                forwards=forwards)
        expected_prices = np.array(
            [0.0, 2.0, 2.0480684764112578, 1.0002029716043364, 2.0730313058959933])
        self.assertAllClose(expected_prices, computed_prices, atol=1e-10)

    def test_option_prices_normal(self):
        """Tests that the prices using normal model are correct."""
        forwards = np.array([0.01, 0.02, 0.03, 0.03, 0.05])
        strikes = np.array([0.03, 0.03, 0.03, 0.03, 0.03])
        volatilities = np.array([0.0001, 0.001, 0.01, 0.005, 0.02])
        expiries = 1.0
        computed_prices = jqf.black_scholes.option_price(
                volatilities=volatilities,
                strikes=strikes,
                expiries=expiries,
                forwards=forwards,
                is_normal_volatility=True)

        expected_prices = np.array(
            [0.0, 0.0, 0.0039894228040143, 0.0019947114020072, 0.0216663094117537])
        self.assertAllClose(expected_prices, computed_prices, atol=1e-10)

    def test_price_zero_vol(self):
        """Tests that zero volatility is handled correctly."""
        # If the volatility is zero, the option's value should be correct.
        forwards = np.array([1.0, 1.0, 1.0, 1.0])
        strikes = np.array([1.1, 0.9, 1.1, 0.9])
        volatilities = np.array([0.0, 0.0, 0.0, 0.0])
        expiries = 1.0
        is_call_options = np.array([True, True, False, False])
        expected_prices = np.array([0.0, 0.1, 0.1, 0.0])
        computed_prices = jqf.black_scholes.option_price(
                volatilities=volatilities,
                strikes=strikes,
                expiries=expiries,
                forwards=forwards,
                is_call_options=is_call_options)
        self.assertAllClose(expected_prices, computed_prices, atol=1e-10)

    def test_price_zero_expiry(self):
        """Tests that zero expiry is correctly handled."""
        # If the expiry is zero, the option's value should be correct.
        forwards = np.array([1.0, 1.0, 1.0, 1.0])
        strikes = np.array([1.1, 0.9, 1.1, 0.9])
        volatilities = np.array([0.1, 0.2, 0.5, 0.9])
        expiries = 0.0
        is_call_options = np.array([True, True, False, False])
        expected_prices = np.array([0.0, 0.1, 0.1, 0.0])
        computed_prices = jqf.black_scholes.option_price(
                volatilities=volatilities,
                strikes=strikes,
                expiries=expiries,
                forwards=forwards,
                is_call_options=is_call_options)
        self.assertAllClose(expected_prices, computed_prices, atol=1e-10)

    def test_price_long_expiry_calls(self):
        """Tests that very long expiry call option behaves like the asset."""
        forwards = np.array([1.0, 1.0, 1.0, 1.0])
        strikes = np.array([1.1, 0.9, 1.1, 0.9])
        volatilities = np.array([0.1, 0.2, 0.5, 0.9])
        expiries = 1e10
        expected_prices = forwards
        computed_prices = jqf.black_scholes.option_price(
                volatilities=volatilities,
                strikes=strikes,
                expiries=expiries,
                forwards=forwards)
        self.assertAllClose(expected_prices, computed_prices, atol=1e-10)

    def test_price_long_expiry_puts(self):
        """Tests that very long expiry put option is worth the strike."""
        forwards = np.array([1.0, 1.0, 1.0, 1.0])
        strikes = np.array([0.1, 10.0, 3.0, 0.0001])
        volatilities = np.array([0.1, 0.2, 0.5, 0.9])
        expiries = 1e10
        expected_prices = strikes
        computed_prices = jqf.black_scholes.option_price(
                volatilities=volatilities,
                strikes=strikes,
                expiries=expiries,
                forwards=forwards,
                is_call_options=False)
        self.assertAllClose(expected_prices, computed_prices, atol=1e-10)

    def test_price_vol_and_expiry_scaling(self):
        """Tests that the price is invariant under vol->k vol, T->T/k**2."""
        np.random.seed(1234)
        n = 20
        forwards = np.exp(np.random.randn(n))
        volatilities = np.exp(np.random.randn(n) / 2)
        strikes = np.exp(np.random.randn(n))
        expiries = np.exp(np.random.randn(n))
        scaling = 5.0
        base_prices = jqf.black_scholes.option_price(
                volatilities=volatilities,
                strikes=strikes,
                expiries=expiries,
                forwards=forwards)
        scaled_prices = jqf.black_scholes.option_price(
                volatilities=volatilities * scaling,
                strikes=strikes,
                expiries=expiries / scaling / scaling,
                forwards=forwards)
        self.assertAllClose(base_prices, scaled_prices, atol=1e-10)

    @parameterized.named_parameters(
        {
            'testcase_name': 'SinglePrecision',
            'dtype': np.float32
        },
        {
            'testcase_name': 'DoublePrecision',
            'dtype': np.float64
        },
    )
    def test_option_prices_detailed_discount(self, dtype):
        """Tests the prices with discount_rates."""
        spots = np.array([80.0, 90.0, 100.0, 110.0, 120.0] * 2)
        strikes = np.array([100.0] * 10)
        discount_rates = 0.08
        volatilities = 0.2
        expiries = 0.25

        is_call_options = np.array([True] * 5 + [False] * 5)
        dividend_rates = 0.12
        computed_prices = jqf.black_scholes.option_price(
                volatilities=volatilities,
                strikes=strikes,
                expiries=expiries,
                spots=spots,
                discount_rates=discount_rates,
                dividend_rates=dividend_rates,
                is_call_options=is_call_options,
                dtype=dtype)
        expected_prices = jnp.array(
            [0.03, 0.57, 3.42, 9.85, 18.62, 20.41, 11.25, 4.40, 1.12, 0.18], dtype)
        self.assertAllClose(expected_prices, computed_prices, atol=5e-3)


    @parameterized.named_parameters(
    {
        'testcase_name': 'SinglePrecision',
        'dtype': jnp.float32
    }, {
        'testcase_name': 'DoublePrecision',
        'dtype': jnp.float64
    })
    def test_barrier_option_dtype(self, dtype):
        """Function tests barrier option pricing for with given data type."""
        spots = 100.0
        rebates = 3.0
        expiries = 0.5
        discount_rates = 0.08
        dividend_rates = 0.04
        strikes = 90.0
        barriers = 95.0
        expected_price = 9.0246
        is_call_options = True
        is_barrier_down = True
        is_knock_out = True
        volatilities = 0.25
        price = jqf.black_scholes.barrier_price(
            volatilities=volatilities,
            strikes=strikes,
            expiries=expiries,
            spots=spots,
            discount_rates=discount_rates,
            dividend_rates=dividend_rates,
            barriers=barriers,
            rebates=rebates,
            is_barrier_down=is_barrier_down,
            is_knock_out=is_knock_out,
            is_call_options=is_call_options,
            dtype=dtype)

        self.assertAllClose(price, jnp.array(expected_price, dtype), rtol=10e-3)
        self.assertEqual(price.dtype, jnp.dtype(dtype))

    def barrier_option_call_xla(self):
        """Tests barrier option price with XLA."""
        dtype = jnp.float64
        spots = jnp.asarray(100.0, dtype=dtype)
        rebates = jnp.asarray(3.0, dtype=dtype)
        expiries = jnp.asarray(0.5, dtype=dtype)
        discount_rates = jnp.asarray(0.08, dtype=dtype)
        dividend_rates = jnp.asarray(0.04, dtype=dtype)
        strikes = jnp.asarray(90.0, dtype=dtype)
        barriers = jnp.asarray(95.0, dtype=dtype)
        expected_price = jnp.asarray(9.0246, dtype=dtype)
        is_call_options = jnp.asarrayr(True)
        is_barrier_down = jnp.asarray(True)
        is_knock_out = jnp.asarray(True)
        volatilities = jnp.asarray(0.25, dtype=dtype)

        def price_barriers_option(samples):
            return jqf.black_scholes.barrier_price(
                volatilities=samples[0],
                strikes=samples[1],
                expiries=samples[2],
                spots=samples[3],
                discount_rates=samples[3],
                dividend_rates=samples[4],
                barriers=samples[5],
                rebates=samples[6],
                is_barrier_down=samples[7],
                is_knock_out=samples[8],
                is_call_options=samples[9])[0]

        def xla_compiled_op(samples):
            return jit(price_barriers_option)(samples)

        price = xla_compiled_op([
            volatilities, strikes, expiries, spots, discount_rates,
            dividend_rates, barriers, rebates, is_barrier_down, is_knock_out,
            is_call_options
        ])
        self.assertAllClose(price, expected_price, atol=10e-3)


    @parameterized.named_parameters(
        {
            'testcase_name': 'NormalModel',
            'is_normal_model': True,
            'volatilities': [0.01, 0.005],
            'expected_price': [0.3458467885511461, 0.3014786656395892],
        }, {
            'testcase_name': 'LognormalModel',
            'is_normal_model': False,
            'volatilities': [1.0, 0.5],
            'expected_price': [0.34885593, 0.31643427],
        })
    def test_swaption_price(self, is_normal_model, volatilities, expected_price):
        """Function tests swaption pricing."""
        dtype = jnp.float64

        expiries = [1.0, 1.0]
        float_leg_start_times = [[1.0, 1.25, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0],
                                [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]]
        float_leg_end_times = [[1.25, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0],
                            [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]]
        fixed_leg_payment_times = [[1.25, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0],
                                [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]]
        float_leg_daycount_fractions = [[
            0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0
        ], [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]]
        fixed_leg_daycount_fractions = [[
            0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0
        ], [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]]
        fixed_leg_coupon = [0.011, 0.011]
        discount_fn = lambda x: np.exp(-0.01 * np.array(x))
        price = jqf.black_scholes.swaption_price(
                volatilities=volatilities,
                expiries=expiries,
                floating_leg_start_times=float_leg_start_times,
                floating_leg_end_times=float_leg_end_times,
                fixed_leg_payment_times=fixed_leg_payment_times,
                floating_leg_daycount_fractions=float_leg_daycount_fractions,
                fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,
                fixed_leg_coupon=fixed_leg_coupon,
                floating_leg_start_times_discount_factors=discount_fn(
                    float_leg_start_times),
                floating_leg_end_times_discount_factors=discount_fn(
                    float_leg_end_times),
                fixed_leg_payment_times_discount_factors=discount_fn(
                    fixed_leg_payment_times),
                is_normal_volatility=is_normal_model,
                notional=100.,
                dtype=dtype)

        self.assertAllClose(price, np.array(expected_price), atol=1e-6)

if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())