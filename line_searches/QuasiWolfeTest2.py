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





class QuasiWolfeTest2:
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


        stage_one_tries = []
        stage_two_currents = []
        stage_two_brackets = []


        stage_one_tries_alphas = []
        stage_two_currents_alphas = []
        stage_two_brackets_alphas = []


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


            stage_one_tries.append(np.clip(x - a_curr * starting_gradient, -1,1))
            stage_one_tries_alphas.append(a_curr)

            phi_a_curr = phi(a_curr)
            dphi_plus_0 = dphi_plus(0)

            # Test Wolfe conditions - if it does not satisfy the sufficient decrease the objective value is higher than
            # in previous iteration there must be a wolfe step in the interval [a_old, a_curr]
            if (phi_a_curr > phi(0) + self.eta_a * a_curr * dphi_plus_0) or \
                    (phi_a_curr >= phi_a_old and i > 1):
                a_star = self.stage_two(a_old, a_curr, phi, dphi_minus, dphi_plus,
                                        x, starting_gradient, phi_0, dphi_plus_0,
                                        stage_two_currents, stage_two_brackets,
                                        stage_two_currents_alphas, stage_two_brackets_alphas)

                return a_star, phi(a_star), np.array(stage_one_tries), np.array(stage_two_currents), np.array(
                    stage_two_brackets), stage_one_tries_alphas, stage_two_currents_alphas, stage_two_brackets_alphas

            dphi_minus_0 = dphi_minus(0)
            dphi_plus_curr = dphi_plus(a_curr)
            dphi_minus_curr = dphi_minus(a_curr)

            # Test the three curvature conditions for the quasi-Wolfe step if satisfied return the step
            # previous conditional checked the first condition
            if abs(dphi_minus_curr) <= self.eta_w * abs(dphi_minus_0) or \
                    abs(dphi_plus_curr) <= self.eta_w * abs(dphi_plus_0) or \
                    (len(check_is_k_step(x, a_curr, starting_gradient, self.constraints)) > 0 and
                     (dphi_minus_curr < 0 < dphi_plus_curr)):
                print(f"found wolfe")
                return a_curr, phi_a_curr,  np.array(stage_one_tries), np.array(stage_two_currents), np.array(
                    stage_two_brackets), stage_one_tries_alphas, stage_two_currents_alphas, stage_two_brackets_alphas

            # If not a wolfe step but satisfies the sufficient decrease condition and gradient of phi at a_curr
            # is positive
            # there must be a Wolfe step between [a_old,a_curr]
            if dphi_minus_curr >= 0:  # FIXME untested!
                a_star = self.stage_two(a_curr, a_old, phi, dphi_minus, dphi_plus,
                                        x, starting_gradient, phi_0, dphi_plus_0,
                                        stage_two_currents, stage_two_brackets,
                                        stage_two_currents_alphas, stage_two_brackets_alphas
                                        )




                return a_star, phi(a_star),  np.array(stage_one_tries), np.array(stage_two_currents), np.array(
                    stage_two_brackets), stage_one_tries_alphas, stage_two_currents_alphas, stage_two_brackets_alphas
            # Else increase the next x from the interval (a_curr, a_max
            # Choose a_iplus1 from the interval (a_i, a_max)
            a_old = a_curr
            a_curr *= self.rho

            # Update ϕ_a_iminus1
            phi_a_old = phi_a_curr

            # Update iteration count
            i += 1

        return self.alpha_max, phi(self.alpha_max), np.array(stage_one_tries), np.array(stage_two_currents), np.array(
                    stage_two_brackets), stage_one_tries_alphas, stage_two_currents_alphas, stage_two_brackets_alphas

    def stage_two(self, a_low_init, a_high_init, phi, dphi_minus, dphi_plus,
                  x, starting_gradient, phi_0, dphi_plus_0,
                  stage_two_currents, stage_two_brackets,
                  stage_two_currents_alphas, stage_two_brackets_alphas):


        init_bracket_xs = np.clip([x - a_low_init * starting_gradient, x - a_high_init * starting_gradient], -1,1)
        stage_two_brackets.append(init_bracket_xs)
        stage_two_brackets_alphas.append((a_low_init, a_high_init))

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
        found_wolfe, need_recheck, a1, a2 = eliminate_kinks(x, starting_gradient, a_low_init, a_high_init,
                                                            self.constraints,
                                                            self.eta_a, self.eta_w, phi_0, dphi_plus_0, phi, dphi_minus,
                                                            dphi_plus,
                                                            omega, omega_prime_minus, omega_prime_plus)

        if found_wolfe:
            print(f"found wolfe in stage 2")
            return a1

        if need_recheck:
            a_low, a_high = (a1, a2) if phi(a1) <= phi(a2) else (a2, a1)
        else:
            a_low, a_high = a1, a2

        # Shrink bracket
        while iteration < max_iterations:

            iteration += 1

            phi_a_low, dphi_minus_a_low, dphi_plus_a_low = phi(a_low), dphi_minus(a_low), dphi_plus(a_low)
            phi_a_high, dphi_minus_a_high, dphi_plus_a_high = phi(a_high), dphi_minus(a_high), dphi_plus(a_high)

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

            stage_two_currents.append(np.clip(x - a_j * starting_gradient, -1,1))
            stage_two_currents_alphas.append(a_j)

            # Evaluate ϕ(a_j)
            phi_a_j = phi(a_j)

            # Check Armijo
            if (phi_a_j > phi_0 + self.eta_a * a_j * dphi_plus_0) or \
                    (phi_a_j > phi_a_low):
                a_high = a_j
            else:
                # It doesn't matter which one we use as they are both the same - kinks have been eliminated
                dphi_minus_a_j = dphi_minus(a_j)
                # dphi_plus_a_j = dphi_plus(a_j)

                if abs(dphi_minus_a_j) <= self.eta_w * abs(dphi_plus_0):
                    return a_j

                if dphi_minus_a_j * (a_high - a_low) >= 0:
                    a_high = a_low

                a_low = a_j

        # Quasi-error response
        print(f" error in stage two ")
        return a_j
