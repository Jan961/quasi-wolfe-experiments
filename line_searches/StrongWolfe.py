import numpy as np
from line_searches.utils import create_phi, create_omega, find_kink_step_indices, project_gradient

class StrongWolfe:
    def __init__(self, eta_a=1e-4, eta_w=0.1, rho=2.0, alpha_max= 10, max_iterations=100, alpha0=.1):
        self.eta_a = eta_a
        self.eta_w = eta_w
        self.rho = rho
        self.max_iterations = max_iterations
        self.alpha_max = alpha_max
        self.alpha0 = alpha0



    def search(self,  f, g, x, starting_gradient):

        stage_one_tries = []
        stage_two_currents = []
        stage_two_brackets = []

        constraints = (-1e12, 1e12)
        phi, dphi_minus, dphi_plus = create_phi(f, g, x, starting_gradient, constraints)

        phi_0 = phi(0)
        dphi_0 = dphi_plus(0)

        c_1, c_2, rho = self.eta_a, self.eta_w, self.rho

        zeroT = 0.0

        # Step-sizes
        a_0 = zeroT
        a_iminus1 = a_0
        a_i = self.alpha0
        a_max = self.alpha_max

        # phi(alpha) = df.f(x + alpha * p)
        phi_a_iminus1 = phi_0
        phi_a_i = np.nan

        # phi'(alpha) = dot(g(x + alpha * p), p)
        dphi_a_i = np.nan

        # Iteration counter
        i = 1

        while a_i < a_max:
            phi_a_i = phi(a_i)

            # Test Wolfe conditions
            if (phi_a_i > phi_0 + c_1 * a_i * dphi_0) or (phi_a_i >= phi_a_iminus1 and i > 1):
                a_star = self.zoom(a_iminus1, a_i, dphi_0, phi_0, phi, dphi_plus)
                return a_star, phi(a_star), stage_one_tries, stage_two_currents, stage_two_brackets

            dphi_a_i = dphi_plus(a_i)

            # Check condition 2
            if abs(dphi_a_i) <= -c_2 * dphi_0:
                return a_i, phi_a_i, stage_one_tries, stage_two_currents, stage_two_brackets

            # Check condition 3
            if dphi_a_i >= zeroT:
                a_star = self.zoom(a_i, a_iminus1, dphi_0, phi_0, phi, dphi_plus)
                return a_star, phi(a_star), stage_one_tries, stage_two_currents, stage_two_brackets

            # Choose a_iplus1 from the interval (a_i, a_max)
            a_iminus1 = a_i
            a_i *= rho

            # Update phi_a_iminus1
            phi_a_iminus1 = phi_a_i

            # Update iteration count
            i += 1

        # Quasi-error response
        return a_max, phi(a_max), stage_one_tries, stage_two_currents, stage_two_brackets

    def zoom(self, a_lo, a_hi, dphi_0, phi_0, phi, dphi, c_1=1e-4, c_2=0.9):
        zeroT = 0.0
        # Step-size
        a_j = np.nan

        # Count iterations
        iteration = 0
        max_iterations = 10

        # Shrink bracket
        while iteration < max_iterations:
            iteration += 1

            phi_a_lo, phi_prime_a_lo = phi(a_lo), dphi(a_lo)
            phi_a_hi, phi_prime_a_hi = phi(a_hi), dphi(a_hi)

            # Interpolate a_j
            if a_lo < a_hi:
                a_j = self.interpolate(a_lo, a_hi, phi_a_lo, phi_a_hi, phi_prime_a_lo, phi_prime_a_hi)
            else:
                a_j = self.interpolate(a_hi, a_lo, phi_a_hi, phi_a_lo, phi_prime_a_hi, phi_prime_a_lo)

            # Evaluate phi(a_j)
            phi_a_j = phi(a_j)

            # Check Armijo
            if (phi_a_j > phi_0 + c_1 * a_j * dphi_0) or (phi_a_j > phi_a_lo):
                a_hi = a_j
            else:
                # Evaluate phi'(a_j)
                phi_prime_a_j = dphi(a_j)

                if abs(phi_prime_a_j) <= -c_2 * dphi_0:
                    return a_j

                if phi_prime_a_j * (a_hi - a_lo) >= zeroT:
                    a_hi = a_lo

                a_lo = a_j

        # Quasi-error response
        return a_j

    def interpolate(self, a_i1, a_i, phi_a_i1, phi_a_i, dphi_a_i1, dphi_a_i):
        d1 = dphi_a_i1 + dphi_a_i - 3 * (phi_a_i1 - phi_a_i) / (a_i1 - a_i)
        d2 = np.sqrt(d1 * d1 - dphi_a_i1 * dphi_a_i)
        return a_i - (a_i - a_i1) * ((dphi_a_i + d2 - d1) / (dphi_a_i - dphi_a_i1 + 2 * d2))
