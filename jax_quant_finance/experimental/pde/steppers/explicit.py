from jax_quant_finance.experimental.pde.steppers.parabolic_equation_stepper import parabolic_equation_step
from jax_quant_finance.experimental.pde.steppers.weighted_implicit_explicit import weighted_implicit_explicit_scheme


def explicit_step():
    
    def step_fn(time,
                next_time,
                coord_grid,
                value_grid,
                boundary_conditions,
                second_order_coeff_fn,
                first_order_coeff_fn,
                zeroth_order_coeff_fn,
                inner_second_order_coeff_fn,
                inner_first_order_coeff_fn,
                num_steps_performed,
                dtype=None):
        """Performs the step."""
        del num_steps_performed
        return parabolic_equation_step(time,
                                    next_time,
                                    coord_grid,
                                    value_grid,
                                    boundary_conditions,
                                    second_order_coeff_fn,
                                    first_order_coeff_fn,
                                    zeroth_order_coeff_fn,
                                    inner_second_order_coeff_fn,
                                    inner_first_order_coeff_fn,
                                    time_marching_scheme=explicit_scheme,
                                    dtype=dtype)
    
    return step_fn



explicit_scheme = weighted_implicit_explicit_scheme(theta=1)


__all__ = ['explicit_step', 'explicit_scheme']

