# failing test
import numpy as np

from function_generation.PolynomialGenerator import PolynomialGenerator
from line_searches.utils import find_kink_step_indices, eliminate_kinks, get_kink_step_alphas, create_phi, \
    check_is_k_step, check_step_is_quasi_wolfe, create_omega
from tests.important_pytests.misc import create_test_function


def test_line_search_utils():
    repeats = 100
    degree = np.random.randint(2, 5)
    constraints = np.array([-1, 1])

    for i in range(repeats):
        f, g, h = PolynomialGenerator().generate(n_vars=10, degree=degree)
        start = np.zeros(10) + 0.1

        eta_w = 0.9
        eta_a = 1e-4

        phi, dphi_minus, dphi_plus = create_phi(f, g, start, g(start), constraints)
        omega, domega_minus, domega_plus = create_omega(eta_a, phi, dphi_minus, dphi_plus)

        assert np.allclose(dphi_minus(0), np.dot(-g(start), g(start)), rtol=0, atol=1e-12) and np.allclose(
            -dphi_plus(0),
            np.dot(g(start), g(start)), rtol=0, atol=1e-12)
        # print(f"original start {start}")
        alphas = get_kink_step_alphas(start, g(start), constraints)
        # print(f"kink step alphas {alphas}")
        sorted_alphas = np.sort(alphas)

        max_x = np.clip(start - g(start) * max(alphas), constraints[-1], constraints[1])
        # print(f" max x {max_x}")
        # print(f" what happened to x - check 1 {start}")
        assert len(find_kink_step_indices(start, max_x)) == 10
        for i in range(10):
            assert max_x[i] <= -1 or max_x[i] >= 1
            kink_step_x = start - g(start) * sorted_alphas[i]

            assert np.any(kink_step_x) == 1 or np.any(kink_step_x) == -1
            clipped = np.clip(kink_step_x, -1, 1)
            # print(f" clipped {clipped}")
            assert np.count_nonzero(np.logical_or(np.isclose(clipped, -1, rtol=0, atol=1e-12),
                                                  np.isclose(clipped, 1, rtol=0, atol=1e-12))) == i + 1

        is_wolfe, need_recheck, a_l, a_h = eliminate_kinks(start, g(start), min(alphas), max(alphas), constraints,
                                                           eta_a, eta_w, phi(start), dphi_plus(start), phi, dphi_minus,
                                                           dphi_plus,
                                                           omega, domega_minus, domega_plus)
        # print(f" what happened to x - check 2 {start}")
        if not is_wolfe:
            print(f" a_l {a_l}")
            print(f"a_h {a_h}")
            print(f"Kink step elimination assertion passed ")
        else:
            print(f" found Wolfe step at alpha {a_l} ")


def test_phi_dphi():
    repeats = 300
    start = np.zeros(10) + 0.2
    constraints = [-1, 1]
    degree = 2

    for j in range(repeats):
        f, g, h = PolynomialGenerator().generate(n_vars=10, degree=degree)

        alphas = get_kink_step_alphas(start, g(start), constraints)
        # print(f"kink step alphas {alphas}")
        sorted_alphas = np.sort(alphas)
        phi, dphi_minus, dphi_plus = create_phi(f, g, start, g(start), constraints)
        for i in range(10):
            kink_step_x = np.clip(start - g(start) * sorted_alphas[i], -1, 1)
            clipped = np.clip(kink_step_x, -1, 1)
            # print(f" clipped {clipped}")
            assert len(check_is_k_step(start, sorted_alphas[i], g(start), constraints)) == 1
            assert len(find_kink_step_indices(start, clipped)) == i + 1
            # print(f" K step verification passed")
            assert dphi_minus(sorted_alphas[i]) != dphi_plus(sorted_alphas[i])
        # print(f" first  batch of tests passed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        test_alphas = np.concatenate(([0], sorted_alphas, [1000000]))
        # print(f"test alphas {test_alphas}")
        for k in range(0, 11):
            lower = test_alphas[k]
            upper = test_alphas[k + 1]

            alpha_in_between = np.random.uniform(lower, upper)

            x_in_between = np.clip(start - alpha_in_between * g(start), -1, 1)
            assert phi(alpha_in_between) == f(x_in_between)
            assert dphi_minus(alpha_in_between) == dphi_plus(alpha_in_between), print(
                f"lower {lower}, between: {alpha_in_between}, upper: {upper}  ,lower == between: {np.allclose(lower, alpha_in_between, rtol=0, atol=1e-12)}, upper == between: {np.allclose(upper, alpha_in_between, rtol=0, atol=1e-12)}")
            assert len(check_is_k_step(start, alpha_in_between, g(start), constraints)) == 0

            assert len(find_kink_step_indices(start, x_in_between)) == k

# failing test
def test_eliminate_kinks():
    dimension = 10
    repeats = 10

    f, g = create_test_function(dimension)

    start = np.zeros(dimension) + 0.5
    for i in range(repeats):
        constraints = [-1, 1]
        eta_w = 0.9
        eta_a = 1e-4

        phi, dphi_minus, dphi_plus = create_phi(f, g, start, g(start), constraints)
        omega, omega_prime_plus, omega_prime_minus = create_omega(eta_a, phi, dphi_minus, dphi_plus)
        alphas = get_kink_step_alphas(start, g(start), constraints)
        # print(f" alphas: {alphas}")

        a_low = alphas.min() - 0.01
        a_high = alphas.max() + 0.01

        found_wolfe, _, a_l, a_h = eliminate_kinks(start, g(start), a_low, a_high, constraints,
                                                   eta_a, eta_w, f(start), dphi_plus(start), phi, dphi_minus, dphi_plus,
                                                   omega, omega_prime_plus, omega_prime_minus)
        is_wolfe = check_step_is_quasi_wolfe(eta_a, eta_w, start, a_l, phi(start), dphi_plus(start), phi(a_l),
                                             dphi_minus(a_l), dphi_plus(a_l), g(start), constraints)

        if not is_wolfe:
            assert not found_wolfe
            assert a_l == a_low
            assert a_h == alphas.min()