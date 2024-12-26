# ## Parameters:  (and defaults)
# # * `c_1 = 1e-4`: Armijo condition
# # * `c_2 = 0.9` : second (strong) Wolfe condition
# import numpy as np
#
# from line_searches.utils import create_phi
#
# def interpolate(a_left, a_right, phi_a_left, phi_a_right, dphi_a_left, dphi_a_right):
#     try:
#
#         A = np.array([
#             [a_left ** 3, a_left ** 2, a_left, 1],
#             [a_right ** 3, a_right ** 2, a_right, 1],
#             [3 * a_left ** 2, 2 * a_left, 1, 0],
#             [3 * x2 ** 2, 2 * x2, 1, 0]
#         ])
#
#
#     except Exception as e:
#         print(f"interpolate failed {e}")
#         return a_left + 0.5 * (a_right - a_left)
#
#
#
#
#
# class QuasiWolfe:
#     def __init__(self, alphamax=10, eta_a=1e-4, eta_w=0.1, rho=2.0, max_iterations=10, constraints=(-1, 1),
#                  alpha0=.01, is_test=False):
#         self.eta_a = eta_a
#         self.eta_w = eta_w
#         self.rho = rho
#         self.max_iterations = max_iterations
#         self.alpha_max = alphamax
#         self.constraints = constraints
#         self.alpha0 = alpha0
#         self.is_test = is_test
#
#     def search(self, f, g, x, starting_gradient):
#         f_calls_count = 0
#         g_calls_count = 0
#
#         phi, dphi_minus, dphi_plus = create_phi(f, g, x, starting_gradient, self.constraints)