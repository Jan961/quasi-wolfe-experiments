import numpy as np


# here constraints is a 2 element vector


def create_phi(f, g, x, starting_gradient, bounds):
    def phi(alpha):
        return f(np.clip(x - alpha * starting_gradient, bounds[0], bounds[1]))

    def dphi_minus(alpha):
        indices = check_is_k_step(x, alpha, starting_gradient, bounds)

        # print(f" length of indices inside dphi minus: {len(indices)} ")

        new_x = np.clip(x - alpha * starting_gradient, bounds[0], bounds[1])

        G = g(new_x)
        projected_gradient = project_gradient(starting_gradient, bounds, new_x)
        # print(f" indices {indices}")
        # print(f" new x {x - alpha*starting_gradient}")
        # print(f" projected_gradient at 'indices' before {projected_gradient[indices]}")
        if len(indices) > 0:
            projected_gradient[indices] = starting_gradient[indices]

        # print(f" starting gradient at indices {starting_gradient[indices]} ")
        # print(f"projected_gradient at 'indices' after {projected_gradient[indices]}")
        # print(f" projected direction non-zeros minus {np.count_nonzero(projected_gradient)}")
        return np.dot(-projected_gradient, G)

    def dphi_plus(alpha):
        new_x = np.clip(x - alpha * starting_gradient, bounds[0], bounds[1])
        G = g(new_x)
        projected_gradient = project_gradient(starting_gradient, bounds, new_x)
        # print(f" projected direction non-zeros plus {np.count_nonzero(projected_gradient)}")
        return np.dot(-projected_gradient, G)

    return phi, dphi_minus, dphi_plus


# eta is eta_a
def create_omega(eta, phi, phi_prime_minus, phi_prime_plus):
    def omega(alpha):
        return phi(alpha) - phi(0) - eta * alpha * phi_prime_plus(0)

    def omega_prime_plus(alpha):
        return phi_prime_plus(alpha) - eta * phi_prime_plus(0)

    def omega_prime_minus(alpha):
        return phi_prime_minus(alpha) - eta * phi_prime_plus(0)

    return omega, omega_prime_minus, omega_prime_plus,


def project_gradient(gradient, constraints, x):
    pgradient_lower = np.where((gradient > 0.0) & np.isclose(x, constraints[0], rtol=0, atol=1e-12), 0.0, gradient)
    pgradient_upper = np.where((pgradient_lower < 0.0) & np.isclose(x, constraints[1], rtol=0, atol=1e-12), 0.0, pgradient_lower)
    return pgradient_upper


def check_is_k_step(x, alpha, starting_gradient, constraints):
    indices = []
    for i in range(len(x)):
        if np.isclose(x[i] - starting_gradient[i] * alpha, constraints[0], rtol=0, atol=1e-12):
            indices.append(i)
        elif np.isclose(x[i] - starting_gradient[i] * alpha, constraints[1], rtol=0, atol=1e-12):
            indices.append(i)
    return np.array(indices)


def find_kink_step_indices(x_start, x_end):
    projected_at_start = np.nonzero(np.logical_or(np.isclose(x_start, -1, rtol=0, atol=1e-12), np.isclose(x_start, 1,rtol=0, atol=1e-12)))
    projected_at_end = np.nonzero(np.logical_or(np.isclose(x_end, -1, rtol=0, atol=1e-12), np.isclose(x_end, 1,rtol=0, atol=1e-12)))

    indices_in_between = np.setdiff1d(projected_at_end, projected_at_start)
    return indices_in_between


def get_kink_step_alphas(x, starting_gradient, constraints):
    # print(f" x  - has to be the same a start - what is going on???????: {x}")
    # print(f"starting direction {starting_gradient}")
    # print(f" constraints check {constraints[0]}")
    negative = np.where(starting_gradient > 0, (x - constraints[0]) / starting_gradient, x)
    positive = np.where(starting_gradient < 0, (negative - constraints[1]) / starting_gradient, negative)

    return positive


# first returned boolean says whether wolfe step was found
# second whether the returned alphas are in the order of: a_low, a_high


# the function assumes we know which is a low and a high
def eliminate_kinks(x, starting_gradient, a_low, a_high, constraints,
                    eta_a, eta_w, phi_0, dphi_plus_0, phi, dphi_minus, dphi_plus,
                    omega, omega_prime_minus, omega_prime_plus,
                    interpolate=False):

    el_kinks_f_calls_count = 0
    el_kinks_g_calls_count = 0

    normal_direction = a_low < a_high

    max_iterations = 100

    if normal_direction:
        x_start = np.clip(x - a_low * starting_gradient, constraints[0], constraints[1])
        x_end = np.clip(x - a_high * starting_gradient, constraints[0], constraints[1])
    else:
        x_start = np.clip(x - a_high * starting_gradient, constraints[0], constraints[1])
        x_end = np.clip(x - a_low * starting_gradient, constraints[0], constraints[1])

    kink_step_indices = find_kink_step_indices(x_start, x_end)
    if len(kink_step_indices) > 0:
        # print(f" kink step indices {kink_step_indices}")
        alphas = get_kink_step_alphas(x[kink_step_indices], starting_gradient[kink_step_indices], constraints)
        bigger_a = a_high if normal_direction else a_low
        alphas = alphas[np.invert( np.isclose(alphas, bigger_a, rtol=0, atol=1e-12))]
    else:
        return False, True, a_low, a_high, el_kinks_f_calls_count, el_kinks_g_calls_count

    if len(alphas) > 0:

        alphas.sort()
        iteration = 0
        alpha_low = a_low
        alpha_high = a_high

        lingering_in_3_count = 0
        current_alpha_index = 0

        while current_alpha_index < len(alphas):
            if normal_direction:
                current_alpha = alphas[current_alpha_index]
            else:
                current_alpha = alphas[-1 - current_alpha_index]
            phi_a_i = phi(current_alpha)
            el_kinks_f_calls_count += 1

            dphi_minus_a_i = dphi_minus(current_alpha)
            dphi_plus_a_i = dphi_plus(current_alpha)
            el_kinks_g_calls_count += 2

            if check_step_is_quasi_wolfe(eta_a, eta_w, x, current_alpha,
                                         phi_0, dphi_plus_0, phi_a_i, dphi_minus_a_i, dphi_plus_a_i, starting_gradient,
                                         constraints):
                return True, False, current_alpha, None, el_kinks_f_calls_count, el_kinks_g_calls_count
            # three cases where we can return the current alpha and alpha low - which bound an interval that will not
            # contain kink steps
            if omega(current_alpha) >= omega(a_low) \
                    or (normal_direction and omega_prime_minus(current_alpha) > 0
                        or (not normal_direction and omega_prime_plus(current_alpha) < 0)):
                el_kinks_f_calls_count += 1
                el_kinks_g_calls_count += 1
                return False, False, alpha_low, current_alpha, el_kinks_f_calls_count, el_kinks_g_calls_count

            # else we move up (if normal direction) or down (if not) the alpha_low
            elif lingering_in_3_count < max_iterations:
                alpha_low = current_alpha
                lingering_in_3_count += 1
                current_alpha_index += 1
            elif current_alpha_index != len(alphas) - 1 and current_alpha_index != 0:
                current_alpha_index = (current_alpha_index + len(alphas)) / 2 if normal_direction \
                    else (current_alpha_index - len(alphas)) / 2
                lingering_in_3_count = 0

            else:
                return False, False, current_alpha, alpha_high, el_kinks_f_calls_count, el_kinks_g_calls_count

        return False, False, current_alpha, alpha_high, el_kinks_f_calls_count, el_kinks_g_calls_count

    else:
        return False, True, a_low, a_high, el_kinks_f_calls_count, el_kinks_g_calls_count


def check_step_is_quasi_wolfe(eta_a, eta_w, x, a_i, phi_0, dphi_plus_0, phi_a_i, dphi_minus_a_i, dphi_plus_a_i,
                              starting_gradient,
                              constraints):
    return (phi_a_i <= phi_0 + a_i * eta_a * dphi_plus_0 and
            (abs(dphi_minus_a_i) <= eta_w * abs(dphi_plus_0) or
             abs(dphi_minus_a_i) <= eta_w * abs(dphi_plus_0) or
             (len(check_is_k_step(x, a_i, starting_gradient, constraints)) > 0 and (
                     dphi_minus_a_i <= 0 <= dphi_plus_a_i))))
