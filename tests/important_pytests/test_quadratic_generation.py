import numpy as np
import sympy as sp

from experiments.utils.generate_quadratics import generate_quadratic
from function_generation.HessF import HessF


# generate a manual check functions

def generate_check_functions_2D(coeffs: list):

    def f(x:np.ndarray):
        return coeffs[0]*x[0]**2 + coeffs[1]*x[0]*x[1] + coeffs[2]*x[1]**2 + coeffs[3]*x[0]+ coeffs[4]*x[1]

    def g(x:np.ndarray):
        return np.array([2*coeffs[0]*x[0] + coeffs[1]*x[1] + coeffs[3],
                         coeffs[1]*x[0] + 2*coeffs[2]*x[1] + coeffs[4]])

    return f,g

def generate_check_functions_3D(coeffs: list):
    def f(x: np.ndarray):
        return (coeffs[0] * x[0] ** 2 + coeffs[1] * x[0] * x[1] + coeffs[2] * x[1] ** 2 +
                coeffs[3] * x[0] * x[2] + coeffs[4] * x[1] * x[2] + coeffs[5] * x[2] ** 2 +
                coeffs[6] * x[0] + coeffs[7] * x[1] + coeffs[8] * x[2])

    def g(x: np.ndarray):
        return np.array([2 * coeffs[0] * x[0] + coeffs[1] * x[1] + coeffs[3]*x[2] + coeffs[6],
                         coeffs[1] * x[0] + 2 * coeffs[2] * x[1] + coeffs[4]*x[2] + coeffs[7],
                         coeffs[3] * x[0] + coeffs[4] * x[1] + 2 * coeffs[5] * x[2] + coeffs[8]])


    return f, g




# check generated quadratics against manual for all variations in 2D
def test_1():

    number_of_points_to_check = 10
    number_of_functions_to_check = 10

    for i in range(number_of_functions_to_check):

        f,g,h, A,b = generate_quadratic(2, squared=True, linear=True, mixed=True, return_coeffs=True)

    #               square1   2 vars        square2  lin1  lin2
        coeffs_m = [A[0,0], A[0,1] + A[1,0], A[1,1], b[0], b[1]]

        f_m, g_m = generate_check_functions_2D(coeffs_m)

        # now check if the values of f and its grad agree for the two methods for a few random points
        for i in range(number_of_points_to_check):
            x = np.random.uniform(-1,1,2)
            assert np.isclose(f(x), f_m(x))
            assert np.allclose(g(x), g_m(x))


# now do the same but in 3D
def test_2():

    number_of_points_to_check = 10
    number_of_functions_to_check = 10

    for i in range(number_of_functions_to_check):

        f,g,h, A,b = generate_quadratic(3, squared=True, linear=True, mixed=True, return_coeffs=True)

    #               square1   2 vars        square2  lin1  lin2                         square 3
        coeffs_m = [A[0,0], A[0,1] + A[1,0], A[1,1], A[0,2] + A[2,0], A[1,2] + A[2,1], A[2,2], b[0], b[1], b[2]]

        f_m, g_m = generate_check_functions_3D(coeffs_m)

        # now check if the values of f and its grad agree for the two methods for a few random points
        for i in range(number_of_points_to_check):
            x = np.random.uniform(-1,1,3)
            assert np.isclose(f(x), f_m(x))
            assert np.allclose(g(x), g_m(x))


def test_hessian_calculation():
    x = sp.Symbol('x', real=True)
    y = sp.Symbol('y', real=True)
    z = sp.Symbol('z', real=True)

    f = x ** 2 + y ** 2 + z ** 2 + 3 * x * y + 4 * x * z + 5 * y * z
    f_prime = [f.diff(var) for var in [x, y, z]]
    f_prime_prime = [[f_prime[i].diff(var) for var in [x, y, z]] for i in range(3)]

    hessian = HessF(f_prime, [x, y, z], get_hessian=True, is_test=True)
    # print(f" hessian {hessian(np.array([1, 1, 1]))}")
    # print(f" f_prime_prime {f_prime_prime}")
    assert np.allclose(hessian(np.array([1, 1, 1])), np.array(f_prime_prime, dtype=np.float64), rtol=0, atol=1e-12)