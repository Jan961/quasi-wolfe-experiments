"""
`BackTracking` specifies a backtracking line-search that uses
a quadratic or cubic interpolant to determine the reduction in step-size.
E.g.,
if f(α) > f(0) + c₁ α f'(0), then the quadratic interpolant of
f(0), f'(0), f(α) has a minimiser α' in the open interval (0, α). More strongly,
there exists a factor ρ = ρ(c₁) such that α' ≦ ρ α.

This is a modification of the algorithm described in Nocedal Wright (2nd ed), Sec. 3.5.



Also do simple heuristic where start alpha is gets all xi closer than some empirically determined constant to the boundary
"""

from line_searches.utils import create_phi, find_kink_step_indices,project_gradient


class QuasiBacktracking:
    def __init__(self, alphamax, c_1=1e-4, rho_hi=0.5, rho_lo=0.1, max_iterations=1000, order=3, constraints=(-1, 1)):
        self.c_1 = c_1
        self.max_iterations = max_iterations
        self.order = order
        self.alphamax = alphamax
        self.constraints = constraints

    # f, g, x, starting_direction, bounds
    def search(self, f, g, x, starting_direction):
        phi, dphi_minus, dphi_plus = create_phi(f, g, x, starting_direction, self.constraints)
        all_kink_step_indices = find_kink_step_indices(x, x-starting_direction*self.alphamax)

        phi_0 = phi(0)
        dphi_0 = dphi_plus(0)
        phi_x0, phi_x1 = phi_0, phi_0
        alpha_1, alpha_2 = self.alphamax, self.alphamax

        phi_x1 = phi(alpha_1)

        iterations = 0
        while phi_x1 > phi_0 + self.c_1 * alpha_2 * dphi_0:
            iterations += 1

            if iterations > self.max_iterations:
                return alpha_2















