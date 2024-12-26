# TODO: Implement safeguards
from line_searches.utils import create_phi, create_omega, find_kink_step_indices, project_gradient, \
    get_kink_step_alphas, \
    check_is_k_step, check_step_is_quasi_wolfe, eliminate_kinks
import numpy as np
import math

"""
`StrongWolfe`: This linesearch algorithm guarantees that the step length
satisfies the (strong) Wolfe conditions.
See Nocedal and Wright - Algorithms 3.5 and 3.6

This algorithm is mostly of theoretical interest, users should most likely
use `MoreThuente`, `HagerZhang` or `BackTracking`.

## Parameters:  (and defaults)
* `c_1 = 1e-4`: Armijo condition
* `c_2 = 0.9` : second (strong) Wolfe condition
* `ρ = 2.0` : bracket growth


adapted from
"""

import math


def interpolate(a_low, a_high, phi_a_low, phi_a_high, dphi_a_low, dphi_a_high):
    try:
        d1 = dphi_a_low + dphi_a_high - 3 * (phi_a_low - phi_a_high) / (a_low - a_high)
        d2 = math.sqrt(d1 * d1 - dphi_a_low * dphi_a_high)
        return a_high - (a_high - a_low) * ((dphi_a_high + d2 - d1) / (dphi_a_high - dphi_a_low + 2 * d2))

    except Exception as e:
        print(f"interpolate failed {e}")
        return a_low + 0.5 * (a_high - a_low)





class QuasiWolfe:
    def __init__(self, alphamax=10, eta_a=1e-4, eta_w=0.1, rho=2.0, max_iterations=10, constraints=(-1, 1),
                 alpha0=.01, is_test=False):
        self.eta_a = eta_a
        self.eta_w = eta_w
        self.rho = rho
        self.max_iterations = max_iterations
        self.alpha_max = alphamax
        self.constraints = constraints
        self.alpha0 = alpha0
        self.is_test = is_test

    def search(self, f, g, x, starting_gradient):

        f_calls_count = 0
        g_calls_count = 0


        phi, dphi_minus, dphi_plus = create_phi(f, g, x, starting_gradient, self.constraints)

        zeroT = 0
        phi_0 = phi(0)

        # Step-sizes
        a_0 = zeroT
        a_old = a_0
        a_curr = self.alpha0

        # ϕ(alpha) = df.f(x + alpha * p)
        phi_a_old = phi_0

        # Iteration counter
        i = 1

        while a_curr < self.alpha_max:

            phi_a_curr = phi(a_curr)
            f_calls_count += 1
            dphi_plus_0 = dphi_plus(0)

            # Test Wolfe conditions - if it does not satisfy the sufficient decrease the objective value is higher than
            # in previous iteration there must be a wolfe step in the interval [a_old, a_curr]
            if (phi_a_curr > phi(0) + self.eta_a * a_curr * dphi_plus_0) or \
                    (phi_a_curr >= phi_a_old and i > 1):
                a_star, st_f_calls_count, st_g_calls_count = self.stage_two(a_old, a_curr, phi, dphi_minus, dphi_plus,
                                        x, starting_gradient, phi_0, dphi_plus_0)
                f_calls_count += st_f_calls_count
                g_calls_count += st_g_calls_count
                return a_star, phi(a_star), f_calls_count, g_calls_count

            dphi_minus_0 = dphi_minus(0)
            dphi_plus_curr = dphi_plus(a_curr)
            dphi_minus_curr = dphi_minus(a_curr)
            g_calls_count += 2

            # Test the three curvature conditions for the quasi-Wolfe step if satisfied return the step
            # previous conditional checked the first condition
            if abs(dphi_minus_curr) <= self.eta_w * abs(dphi_minus_0) or \
                    abs(dphi_plus_curr) <= self.eta_w * abs(dphi_plus_0) or \
                    (len(check_is_k_step(x, a_curr, starting_gradient, self.constraints)) > 0 and
                     (dphi_minus_curr < 0 < dphi_plus_curr)):
                return a_curr, phi_a_curr, f_calls_count, g_calls_count

            # If not a wolfe step but satisfies the sufficient decrease condition and gradient of phi at a_curr
            # is positive
            # there must be a Wolfe step between [a_old,a_curr]
            if dphi_minus_curr >= 0:  # FIXME untested!
                a_star, st_f_calls_count, st_g_calls_count = self.stage_two(a_curr, a_old, phi, dphi_minus, dphi_plus,
                                        x, starting_gradient, phi_0, dphi_plus_0)
                f_calls_count += st_f_calls_count
                g_calls_count += st_g_calls_count
                return a_star, phi(a_star), f_calls_count, g_calls_count

            # Else increase the next x from the interval (a_curr, a_max
            # Choose a_iplus1 from the interval (a_i, a_max)
            a_old = a_curr
            a_curr *= self.rho

            # Update ϕ_a_iminus1
            phi_a_old = phi_a_curr

            # Update iteration count
            i += 1

        return self.alpha_max, phi(self.alpha_max), f_calls_count, g_calls_count

    def stage_two(self, a_low_init, a_high_init, phi, dphi_minus, dphi_plus,
                  x, starting_gradient, phi_0, dphi_plus_0):

        st_f_calls_count = 0
        st_g_calls_count = 0

        zeroT = 0
        # Step-size
        a_j = None

        # Count iterations
        iteration = 0
        max_iterations = 10

        # Generate helper functions
        omega, omega_prime_minus, omega_prime_plus = create_omega(self.eta_a, phi,
                                                                  phi_prime_minus=dphi_minus,
                                                                  phi_prime_plus=dphi_plus)

        # Eliminate kinks
        found_wolfe, need_recheck, a1, a2, el_f_count, el_g_count = eliminate_kinks(x, starting_gradient, a_low_init, a_high_init,
                                                            self.constraints,
                                                            self.eta_a, self.eta_w, phi_0, dphi_plus_0, phi, dphi_minus,
                                                            dphi_plus,
                                                            omega, omega_prime_minus, omega_prime_plus)
        st_f_calls_count += el_f_count
        st_g_calls_count += el_g_count

        if found_wolfe:
            return a1, st_f_calls_count, st_g_calls_count

        if need_recheck:
            a_low, a_high = (a1, a2) if phi(a1) <= phi(a2) else (a2, a1)
        else:
            a_low, a_high = a1, a2

        # Shrink bracket
        while iteration < max_iterations:

            iteration += 1

            phi_a_low, dphi_minus_a_low, dphi_plus_a_low = phi(a_low), dphi_minus(a_low), dphi_plus(a_low)
            phi_a_high, dphi_minus_a_high, dphi_plus_a_high = phi(a_high), dphi_minus(a_high), dphi_plus(a_high)
            st_f_calls_count += 2
            st_g_calls_count += 4


            # Interpolate a_j
            if a_low < a_high:
                a_j = interpolate(a_low, a_high,
                                  phi_a_low, phi_a_high,
                                  dphi_plus_a_low, dphi_minus_a_high)
            else:
                # TODO: Check if this is needed
                a_j = interpolate(a_high, a_low,
                                  phi_a_high, phi_a_low,
                                  dphi_minus_a_high, dphi_plus_a_low)

            # Evaluate ϕ(a_j)
            phi_a_j = phi(a_j)
            st_f_calls_count += 1

            # Check Armijo
            if (phi_a_j > phi_0 + self.eta_a * a_j * dphi_plus_0) or \
                    (phi_a_j > phi_a_low):
                a_high = a_j
            else:
                # It doesn't matter which one we use as they are both the same - kinks have been eliminated
                dphi_minus_a_j = dphi_minus(a_j)
                st_g_calls_count += 1
                # dphi_plus_a_j = dphi_plus(a_j)

                if abs(dphi_minus_a_j) <= self.eta_w * abs(dphi_plus_0):
                    return a_j, st_f_calls_count, st_g_calls_count

                if dphi_minus_a_j * (a_high - a_low) >= 0:
                    a_high = a_low

                a_low = a_j

        # Quasi-error response
        return a_j, st_f_calls_count, st_g_calls_count
