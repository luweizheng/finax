import jax.numpy as jnp


def parabolic_equation_step(
    time, 
    next_time,
    coord_grid,
    value_grid,
    boundary_conditions,
    second_order_coeff_fn,
    first_order_coeff_fn,
    zeroth_order_coeff_fn,
    inner_second_order_coeff_fn,
    inner_first_order_coeff_fn,
    time_marching_scheme,
    dtype=None):
    
    time = jnp.asarray(time, dtype=dtype)
    next_time = jnp.asarray(next_time, dtype=dtype)
    coord_grid = [jnp.asarray(x, dtype=dtype) for _, x in enumerate(coord_grid)]
    value_grid = jnp.asarray(value_grid, dtype=dtype)
    
    
    
    if boundary_conditions[0][0] is None:
        has_default_lower_boundary = True
        lower_index = 0
    else:
        has_default_lower_boundary = False
        lower_index = 1
    
    if boundary_conditions[0][1] is None:
        has_default_upper_boundary = True
        upper_index = None
    else:
        upper_index = -1
        has_default_upper_boundary = False
    
    inner_grid_in = value_grid[..., lower_index:upper_index]
    coord_grid_deltas = coord_grid[0][1:] - coord_grid[0][:-1]
    
    
    def equation_params_fn(t):
        return _construct_space_discretized_eqn_params(
                coord_grid, coord_grid_deltas, value_grid, boundary_conditions,
                has_default_lower_boundary, has_default_upper_boundary,
                second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn,
                inner_second_order_coeff_fn, inner_first_order_coeff_fn, t)
    
    inner_grid_out = time_marching_scheme(
        value_grid=inner_grid_in,
        t1=time,
        t2=next_time,
        equation_params_fn=equation_params_fn)
    
    updated_value_grid = _apply_boundary_conditions_after_step(
        inner_grid_out, boundary_conditions,
        has_default_lower_boundary, has_default_upper_boundary,
        coord_grid, coord_grid_deltas, next_time)
    return coord_grid, updated_value_grid


def _construct_space_discretized_eqn_params(
    coord_grid,
    coord_grid_deltas,
    value_grid,
    boundary_conditions,
    has_default_lower_boundary,
    has_default_upper_boundary,
    second_order_coeff_fn,
    first_order_coeff_fn,
    zeroth_order_coeff_fn,
    inner_second_order_coeff_fn,
    inner_first_order_coeff_fn,              
    t):
    """Construct tridiagnol matrix

    Args:
        coord_grid (_type_): _description_
        coord_grid_deltas (_type_): _description_
        value_grid (_type_): _description_
        boundary_conditions (_type_): _description_
        has_default_lower_boundary (bool): _description_
        has_default_upper_boundary (bool): _description_
        second_order_coeff_fn (_type_): _description_
        first_order_coeff_fn (_type_): _description_
        zeroth_order_coeff_fn (_type_): _description_
        t (_type_): _description_
    """
    
    forward_deltas = coord_grid_deltas[1:]
    backward_deltas = coord_grid_deltas[:-1]
    
    sum_deltas = forward_deltas + backward_deltas
    
    second_order_coeff_fn = second_order_coeff_fn or (lambda *args: [[None]])
    first_order_coeff_fn = first_order_coeff_fn or (lambda *args: [None])
    zeroth_order_coeff_fn  = zeroth_order_coeff_fn or (lambda *args: None)
    inner_second_order_coeff_fn = inner_second_order_coeff_fn or (lambda *args: [[None]])
    inner_first_order_coeff_fn = inner_first_order_coeff_fn or (lambda *args: [None])
    
    second_order_coeff = _prepare_pde_coeffs(
        second_order_coeff_fn(t, coord_grid)[0][0], value_grid)
    first_order_coeff = _prepare_pde_coeffs(
        first_order_coeff_fn(t, coord_grid)[0], value_grid)
    zeroth_order_coeff = _prepare_pde_coeffs(
        zeroth_order_coeff_fn(t, coord_grid), value_grid)
    inner_second_order_coeff = _prepare_pde_coeffs(
        inner_second_order_coeff_fn(t, coord_grid)[0][0], value_grid)
    inner_first_order_coeff = _prepare_pde_coeffs(
        inner_first_order_coeff_fn(t, coord_grid)[0], value_grid)
    
    zeros = jnp.zeros_like(value_grid[...,1:-1])
  # Discretize zeroth-order term.
    if zeroth_order_coeff is None:
        diag_zeroth_order = zeros
    else:
        # Minus is due to moving to rhs.
        diag_zeroth_order = -zeroth_order_coeff[..., 1:-1]
    # Discretize first-order term.
    if first_order_coeff is None and inner_first_order_coeff is None:
        # No first-order term.
        superdiag_first_order = zeros
        diag_first_order = zeros
        subdiag_first_order = zeros
    else:
        superdiag_first_order = -backward_deltas / (sum_deltas * forward_deltas)
        subdiag_first_order = forward_deltas / (sum_deltas * backward_deltas)
        diag_first_order = -superdiag_first_order - subdiag_first_order
        if first_order_coeff is not None:
            superdiag_first_order *= first_order_coeff[..., 1:-1]
            subdiag_first_order *= first_order_coeff[..., 1:-1]
            diag_first_order *= first_order_coeff[..., 1:-1]
        if inner_first_order_coeff is not None:
            superdiag_first_order *= inner_first_order_coeff[..., 2:]
            subdiag_first_order *= inner_first_order_coeff[..., :-2]
            diag_first_order *= inner_first_order_coeff[..., 1:-1]
    # Discretize second-order term.
    if second_order_coeff is None and inner_second_order_coeff is None:
        # No second-order term.
        superdiag_second_order = zeros
        diag_second_order = zeros
        subdiag_second_order = zeros
    else:
        superdiag_second_order = -2 / (sum_deltas * forward_deltas)
        subdiag_second_order = -2 / (sum_deltas * backward_deltas)
        diag_second_order = -superdiag_second_order - subdiag_second_order
        if second_order_coeff is not None:
            superdiag_second_order *= second_order_coeff[..., 1:-1]
            subdiag_second_order *= second_order_coeff[..., 1:-1]
            diag_second_order *= second_order_coeff[..., 1:-1]
        if inner_second_order_coeff is not None:
            superdiag_second_order *= inner_second_order_coeff[..., 2:]
            subdiag_second_order *= inner_second_order_coeff[..., :-2]
            diag_second_order *= inner_second_order_coeff[..., 1:-1]
    superdiag = superdiag_first_order + superdiag_second_order
    subdiag = subdiag_first_order + subdiag_second_order
    diag = diag_zeroth_order + diag_first_order + diag_second_order    
    (subdiag, diag, superdiag) = _apply_default_boundary(subdiag, diag, superdiag,
                                                         zeroth_order_coeff,
                                                         inner_first_order_coeff,
                                                         first_order_coeff,
                                                         forward_deltas,
                                                         backward_deltas,
                                                         has_default_lower_boundary,
                                                         has_default_upper_boundary)
    
    return _apply_robin_boundary_conditions(
        value_grid, boundary_conditions,
        has_default_lower_boundary, has_default_upper_boundary,
        coord_grid, coord_grid_deltas, diag, superdiag, subdiag, t
    )
    
    
    
    


def _apply_default_boundary(subdiag, diag, superdiag,
                            zeroth_order_coeff,
                            inner_first_order_coeff,
                            first_order_coeff,
                            forward_deltas,
                            backward_deltas,
                            has_default_lower_boundary,
                            has_default_upper_boundary):

    batch_shape = diag.shape[:-1]
    if zeroth_order_coeff is None:
        zeroth_order_coeff = jnp.zeros([1], dtype=diag.dtype)
    
    if has_default_lower_boundary:
        (subdiag, diag, superdiag) = _apply_default_lower_boundary(subdiag, 
                                                                   diag,
                                                                   superdiag,
                                                                   zeroth_order_coeff,
                                                                   inner_first_order_coeff,
                                                                   first_order_coeff,
                                                                   forward_deltas,
                                                                   batch_shape)
        
    if has_default_upper_boundary:
        (subdiag, diag, superdiag) = _apply_default_upper_boundary(subdiag, 
                                                                   diag,
                                                                   superdiag,
                                                                   zeroth_order_coeff,
                                                                   inner_first_order_coeff,
                                                                   first_order_coeff,
                                                                   backward_deltas,
                                                                   batch_shape)
    
       
    return subdiag ,diag, superdiag        
        
        


def _apply_default_lower_boundary(subdiag, diag, superdiag,
                                  zeroth_order_coeff,
                                  inner_first_order_coeff,
                                  first_order_coeff,
                                  forward_deltas,
                                  batch_shape):
    
    if inner_first_order_coeff is None:
        inner_coeff = jnp.asarray([1,1], dtype=diag.dtype)
    else:
        inner_coeff = inner_first_order_coeff
    
    if first_order_coeff is None:
        if inner_first_order_coeff is None:
            extra_first_order_coeff = jnp.zeros(batch_shape, dtype=diag.dtype)
        else:
            extra_first_order_coeff = jnp.ones(batch_shape, dtype=diag.dtype)
    else:
        extra_first_order_coeff = first_order_coeff[...,0]
    extra_superdiag_coeff = (inner_coeff[...,0] * extra_first_order_coeff 
                             / forward_deltas[..., 0] 
                             + zeroth_order_coeff[..., 0])
    
    superdiag = _append_first(-extra_superdiag_coeff, superdiag)
    
    extra_diag_coeff = (-inner_coeff[..., 0] * extra_first_order_coeff
                      / forward_deltas[..., 0]
                      + zeroth_order_coeff[..., 0])
    
    diag = _append_first(-extra_diag_coeff, diag)
    subdiag = _append_first(jnp.zeros_like(extra_diag_coeff), subdiag)

    
    return subdiag, diag, superdiag


def _apply_default_upper_boundary(subdiag, diag, superdiag,
                                  zeroth_order_coeff,
                                  inner_first_order_coeff,
                                  first_order_coeff,
                                  backward_deltas,
                                  batch_shape):
    
    if inner_first_order_coeff is None:
        inner_coeff = jnp.asarray([1, 1], dtype=diag.dtype)
    else:
        inner_coeff = inner_first_order_coeff
    if first_order_coeff is None:
        if inner_first_order_coeff is None:
        # Corresponds to B(t, x_max)
            extra_first_order_coeff = jnp.zeros(batch_shape, dtype=diag.dtype)
        else:
            extra_first_order_coeff = jnp.ones(batch_shape, dtype=diag.dtype)
    else:
        extra_first_order_coeff = first_order_coeff[..., -1]
    extra_diag_coeff = (inner_coeff[..., -1] * extra_first_order_coeff
                        / backward_deltas[..., -1]
                        + zeroth_order_coeff[..., -1])
    # Minus is due to moving to rhs.
    diag = _append_last(diag, -extra_diag_coeff)
    # Update subdiagonal
    extra_sub_coeff = (-inner_coeff[..., -2] * extra_first_order_coeff
                        / backward_deltas[..., -1])
    # Minus is due to moving to rhs.
    subdiag = _append_last(subdiag, -extra_sub_coeff)
    # Update superdiag
    superdiag = _append_last(superdiag, -jnp.zeros_like(extra_diag_coeff))
    return subdiag, diag, superdiag


def _apply_robin_boundary_conditions(value_grid,
    boundary_conditions,
    has_default_lower_boundary,
    has_default_upper_boundary,
    coord_grid, coord_grid_deltas,
    diagonal,
    upper_diagonal,
    lower_diagonal, t):

    if (has_default_lower_boundary and
        has_default_upper_boundary):
        return (diagonal, upper_diagonal, lower_diagonal), jnp.zeros_like(diagonal)

    batch_shape = value_grid.shape[:-1]
    # Retrieve the boundary conditions in the form alpha V + beta V' = gamma.
    if has_default_lower_boundary:
        # No need for the BC as default BC was applied
        alpha_l, beta_l, gamma_l = None, None, None
    else:
        alpha_l, beta_l, gamma_l = boundary_conditions[0][0](t, coord_grid)
    if has_default_upper_boundary:
        # No need for the BC as default BC was applied
        alpha_u, beta_u, gamma_u = None, None, None
    else:
        alpha_u, beta_u, gamma_u = boundary_conditions[0][1](t, coord_grid)

    alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = (
        _prepare_boundary_conditions(b, value_grid)
        for b in (alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u))

    if beta_l is None and beta_u is None:
        # Dirichlet or default conditions on both boundaries. In this case there are
        # no corrections to the tridiagonal matrix, so we can take a shortcut.
        if has_default_lower_boundary:
        # Inhomogeneous term is zero for default BC
            first_inhomog_element = jnp.zeros(batch_shape, dtype=value_grid.dtype)
        else:
            first_inhomog_element = lower_diagonal[..., 0] * gamma_l / alpha_l
        if has_default_upper_boundary:
        # Inhomogeneous term is zero for default BC
            last_inhomog_element = jnp.zeros(batch_shape, dtype=value_grid.dtype)
        else:
            last_inhomog_element = upper_diagonal[..., -1] * gamma_u / alpha_u
        inhomog_term = _append_first_and_last(first_inhomog_element,
                                            jnp.zeros_like(diagonal[..., 1:-1]),
                                            last_inhomog_element)
        return (diagonal, upper_diagonal, lower_diagonal), inhomog_term

    # Convert the boundary conditions into the form v0 = xi1 v1 + xi2 v2 + eta,
    # and calculate corrections to the tridiagonal matrix and the inhomogeneous
    # term.
    if has_default_lower_boundary:
        # No update for the default BC
        first_inhomog_element = jnp.zeros(batch_shape, dtype=value_grid.dtype)
        diag_first_correction = 0
        upper_diag_correction = 0
    else:
        # Robin BC case for the lower bound
        xi1, xi2, eta = _discretize_boundary_conditions(coord_grid_deltas[0],
                                                        coord_grid_deltas[1],
                                                        alpha_l,
                                                        beta_l, gamma_l)
        diag_first_correction = lower_diagonal[..., 0] * xi1
        upper_diag_correction = lower_diagonal[..., 0] * xi2
        first_inhomog_element = lower_diagonal[..., 0] * eta

    if has_default_upper_boundary:
        # No update for the default BC
        last_inhomog_element = jnp.zeros(batch_shape, dtype=value_grid.dtype)
        diag_last_correction = 0
        lower_diag_correction = 0
    else:
        # Robin BC case for the upper bound
        xi1, xi2, eta = _discretize_boundary_conditions(coord_grid_deltas[-1],
                                                        coord_grid_deltas[-2],
                                                        alpha_u,
                                                        beta_u, gamma_u)
        diag_last_correction = upper_diagonal[..., -1] * xi1
        lower_diag_correction = upper_diagonal[..., -1] * xi2
        last_inhomog_element = upper_diagonal[..., -1] * eta

    # Update spatial discretization matrix, where appropriate
    diagonal = _append_first_and_last(diagonal[..., 0] + diag_first_correction,
                                        diagonal[..., 1:-1],
                                        diagonal[..., -1] + diag_last_correction)
    upper_diagonal = _append_first(
        upper_diagonal[..., 0] + upper_diag_correction, upper_diagonal[..., 1:])
    lower_diagonal = _append_last(
        lower_diagonal[..., :-1],
        lower_diagonal[..., -1] + lower_diag_correction)
    inhomog_term = _append_first_and_last(first_inhomog_element,
                                            jnp.zeros_like(diagonal[..., 1:-1]),
                                            last_inhomog_element)
    return (diagonal, upper_diagonal, lower_diagonal), inhomog_term


def _apply_boundary_conditions_after_step(    
    inner_grid_out,
    boundary_conditions,
    has_default_lower_boundary,
    has_default_upper_boundary,
    coord_grid, coord_grid_deltas,
    time_after_step):
    
    if has_default_lower_boundary:
        # No update for the default BC
        first_value = None
    else:
        # Robin BC case
        alpha, beta, gamma = boundary_conditions[0][0](time_after_step,
                                                    coord_grid)
        alpha, beta, gamma = (
            _prepare_boundary_conditions(b, inner_grid_out)
            for b in (alpha, beta, gamma))
        xi1, xi2, eta = _discretize_boundary_conditions(coord_grid_deltas[0],
                                                        coord_grid_deltas[1],
                                                        alpha, beta, gamma)
        first_value = (
            xi1 * inner_grid_out[..., 0] + xi2 * inner_grid_out[..., 1] + eta)

    if has_default_upper_boundary:
        # No update for the default BC
        last_value = None
    else:
        # Robin BC case
        alpha, beta, gamma = boundary_conditions[0][1](time_after_step,
                                                    coord_grid)
        alpha, beta, gamma = (
            _prepare_boundary_conditions(b, inner_grid_out)
            for b in (alpha, beta, gamma))
        xi1, xi2, eta = _discretize_boundary_conditions(coord_grid_deltas[-1],
                                                        coord_grid_deltas[-2],
                                                        alpha, beta, gamma)
        last_value = (
            xi1 * inner_grid_out[..., -1] + xi2 * inner_grid_out[..., -2] + eta)

    return _append_first_and_last(first_value, inner_grid_out, last_value)

def _prepare_pde_coeffs(raw_coeffs, value_grid):
    if raw_coeffs is None:
        return None
    dtype = value_grid.dtype
    coeffs = jnp.asarray(raw_coeffs, dtype=dtype)
    broadcast_shape = value_grid.shape
    coeffs = jnp.broadcast_to(coeffs, broadcast_shape)
    return coeffs
    


def _prepare_boundary_conditions(boundary, value_grid):
    if boundary is None:
        return None
    boundary = jnp.asarray(boundary, dtype=value_grid.dtype)
    boundary_shape = boundary.shape[:-1]
    return jnp.broadcast_to(boundary, boundary_shape)


def _discretize_boundary_conditions(dx0, dx1, alpha, beta, gamma):
    if beta is None:
        if alpha is None:
            raise ValueError("Invalid boundary conditions: alpha and beta can't both be None")
        zeros = jnp.zeros_like(gamma)
        return zeros, zeros, gamma/alpha
    
    denom = beta * dx1 * (2 * dx0 + dx1)
    if alpha is not None:
        denom += alpha * dx0 * dx1 * (dx0 + dx1)
    
    xi1 = beta * (dx0 + dx1) * (dx0 + dx1) / denom
    xi2 = -beta * dx0 * dx0 /denom
    eta = gamma * dx0 * dx1 * (dx0 + dx1) /denom
    
    return xi1, xi2, eta


def _append_first_and_last(first, inner, last):
    if first is None:
        return _append_last(inner, last)
    if last is None:
        return _append_first(first, inner)
    return jnp.concatenate((jnp.expand_dims(first, axis=-1), inner, jnp.expand_dims(last, axis=-1)), axis=-1)


def _append_first(first, rest):
    if first is None:
        return rest
    return jnp.concatenate((jnp.expand_dims(first, axis=-1), rest), axis=-1)

def _append_last(rest, last):
    if last is None:
        return rest
    return jnp.concatenate((rest,jnp.expand_dims(last, axis=-1)), axis=-1)    


__all__ = ['parabolic_equation_step']

    
    
    