import random

import numpy as np
from scipy.interpolate import CubicSpline
from experiments.utils.generate_quadratics import generate_quadratic
from experiments.utils.get_lipschitz import get_lipschitz
from function_generation.PolynomialGenerator import PolynomialGenerator
from function_generation.HessF import HessF
from grad_descent import grad_descent
from grad_descent_test import grad_descent_test
from line_searches.QuasiWolfe import QuasiWolfe
from line_searches.StrongWolfe import StrongWolfe
import sympy as sp
from line_searches.Constant import Constant
from line_searches.QuasiWolfeTest import QuasiWolfeTest
from line_searches.utils import eliminate_kinks, get_kink_step_alphas, find_kink_step_indices, check_is_k_step, \
    create_phi, create_omega, check_step_is_quasi_wolfe
import matplotlib.pyplot as plt


def flatten(xss):
    return [x for xs in xss for x in xs]


def create_test_function(dimension):
    coefficients = np.random.uniform(1.0, 2.0, (dimension))

    def f(x):
        return np.sum(coefficients * x ** 2)

    def g(x):
        return 2 * coefficients * x

    return f, g










def test_lipschitz_1d():
    boundary = 1
    repeats = 300
    for i in range(repeats):
        for d in [2, 4]:

            f, f_prime, _ = PolynomialGenerator().generate(n_vars=1, degree=d)
            # print(f" expression {f.expression}")
            # print(f"coeffs 1 {f.coeffs}")
            symbol = f.expression.free_symbols.pop()
            lipschitz = get_lipschitz(f.expression, -boundary, boundary)
            coeffs = np.array(f.coeffs).flatten()

            # print(f" coeffs flattened {coeffs}")

            if d == 4:
                # print(f"\n checking 4 \n")
                coeffs = coeffs * (np.array([1, 2, 6, 12]))
                # print(f"coeffs multiplied:  {coeffs}")
                x_min = -coeffs[-2] / (2 * coeffs[-1])

                if -1 <= x_min <= 1:
                    # print(f"\n x min: {x_min} \n")
                    y = coeffs[-1] * x_min ** 2 + coeffs[-2] * x_min + coeffs[-3]
                else:
                    y = 0

                b1 = np.sum(np.array([0, 1, -1, 1]) * coeffs)
                b2 = np.sum(coeffs[1:])
                # print(f'Lipschitz {lipschitz}')
                # print(f"y {y}")
                # print(f"b1 {b1}")
                # print(f" b2 {b2}")
                l_manual = np.max(np.abs(np.array([y, b2, b1])))
            else:
                # print(f"\n checking 2 \n")
                l_manual = np.abs(2 * coeffs[-1])
            # print(type(lipschitz))
            # print(f"type manual {l_manual}")
            assert np.allclose(lipschitz, l_manual, rtol=0, atol=1e-12)


# testing polynomial generation with sympy against manual implementation with the same coefficients
# testing both the derivative and the function
def test_polynomial_generator_1d():
    for degree in range(1, 4):
        for j in range(20):
            f, f_prime, _ = PolynomialGenerator().generate(n_vars=1, degree=degree)
            coeff_list = list(np.array(f.coeffs).flatten())
            coeff_list.insert(0, f.constant_factor)
            coefficients = np.array(coeff_list)
            der = np.arange(1, degree + 1)
            new_coefficients = der * coefficients[1:]
            # print(f"coefficients {coefficients}")
            # print(f"real coefficients {np.array(flatten(f.coeffs))}")
            # print(f" constant {f.constant_factor}")

            for i in range(50):
                x = 2 * np.random.rand(1) - 1
                y = f(x)
                y_prime = f_prime(x)
                # print(f"type y {type(y)}")
                # print(f" polyval type {type(np.float64(np.polyval(np.flip(coefficients), x)))}")
                # print(f" y prime type {y_prime.dtype}")

                assert np.allclose(y, np.float64(np.polyval(np.flip(coefficients), x)), rtol=0, atol=1e-12)
                assert np.allclose(y_prime, np.polyval(np.flip(new_coefficients), x), rtol=0, atol=1e-12)


def test_polynomial_generator_2d():
    for degree in range(1, 3):
        for j in range(20):
            f, f_prime, _ = PolynomialGenerator().generate(n_vars=2, degree=degree)
            coeff_list = list(flatten(f.coeffs))
            coeff_list.insert(0, f.constant_factor)

            for i in range(50):
                if degree == 1:
                    x = np.random.uniform(-1, 1, 2)
                    y = f(x)
                    y_manual = np.float64(x[0] * coeff_list[1] + x[1] * coeff_list[2] + coeff_list[0])
                    # print(f"y generated {y}")
                    # print(f"y manual {y_manual}")
                    # print(f" type y {type(y)}")
                    # print(f" type y manual {type(y_manual)}")
                    assert np.allclose(y, y_manual, rtol=0, atol=1e-12)

                if degree == 2:
                    x = np.random.uniform(-1, 1, 2)
                    y = f(x)
                    y_manual = np.float64(
                        x[0] ** 2 * coeff_list[3] + x[0] * x[1] * coeff_list[4] + coeff_list[5] * x[1] ** 2 +
                        coeff_list[0] + x[0] * coeff_list[1] + x[1] * coeff_list[2])

                    y_prime_manual = np.array([2 * x[0] * coeff_list[3] + x[1] * coeff_list[4] + coeff_list[1],
                                               2 * x[1] * coeff_list[5] + x[0] * coeff_list[4] + coeff_list[2]])

                    # print(f"y generated {y}")
                    # print(f"y manual {y_manual}")
                    assert np.allclose(y, y_manual, rtol=0, atol=1e-12)
                    assert np.allclose(f_prime(x), y_prime_manual, rtol=0, atol=1e-12)









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






def test_quasi_wolfe_produces_same_results_as_strong_wolfe_from_optim():
    repeats = 10

    for i in range(repeats):
        f, g, h = PolynomialGenerator().generate(2, 3)
        constraints = [-1e12, 1e12]
        x0 = np.zeros(2)
        # alphamax=10, eta_a=1e-4, eta_w=0.1, rho=2.0, max_iterations=100, ,
        #                  alpha0=.1, is_test=False)
        qw_ls = QuasiWolfe(constraints=constraints, alpha0=0.1, max_iterations=100, eta_a=1e-4, eta_w=0.1, rho=2.0, alphamax=10)

        res1 = qw_ls.search(f, g, x0, g(x0))

        # eta_a = 1e-4, eta_w = 0.1, rho = 2.0, alpha_max = 10, max_iterations = 100, alpha0 = .1
        st_wolfe = StrongWolfe()
        res2 = st_wolfe.search(f, g, x0, g(x0))

        print(f" res1 {res1[0]}")
        print(f" res2 {res2[0]}")

        assert np.allclose(res1[0], res2[0], rtol=0, atol=1e-12)
        assert np.allclose(res1[1], res2[1], rtol=0, atol=1e-12)

# failing test
# these tests are adapted from Julia Optim - checking the unconstrained case
def test_alpha_calculation_quadratic():
    def f(x):
        return sum(x ** 2)

    def g(x):
        return 2 * x

    lsalphas = [1.0, 0.5, 0.5, 0.49995, 0.5, 0.5]  # types
    #          Stat  #HZ  wolfe   mt    bt3   bt2
    x = np.array([-1., -1.])
    constraints = [-10e6, 10e6]
    phi, dphi_minus, dphi_plus = create_phi(f, g, x, g(x), constraints)

    # alpha_init, f, g, x, starting_gradient, phi_0, dphi_plus_0
    strong_wolfe = StrongWolfe(alpha0=1.0)
    lr, _, _, _, _ = strong_wolfe.search(f, g, x, g(x))

    quasi_wolfe = QuasiWolfe(alpha0=1.0)
    lr2, _ = quasi_wolfe.search(f, g, x, g(x))

    assert np.allclose(lr, 0.5, rtol=0, atol=1e-12)
    assert np.allclose(lr, 0.5, rtol=0, atol=1e-12)

    # more_thuente = MoreThuenteTest(alpha0=1.0, constraints=constraints)
    # lr3, _ = more_thuente.search(f, g, x, g(x))
    #
    # assert np.isclose(lr3, 0.49995, rtol=0, atol=1e-12)



# failing test
# Another test adapted from Julia Optim
def test_alpha_calculation_Himmelblau():
    known_results = [0.020646100006834013, 0.0175892025844326]

    # wolfe
    def himmelblau(x):
        return np.sum((x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2)

    def himmelblau_grad(x):
        return np.array([4 * x[0] * (x[0] ** 2 + x[1] - 11) + 2 * (x[0] + x[1] ** 2 - 7),
                         2 * (x[0] ** 2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1] ** 2 - 7)])

    # load x0 from a file with in npz format
    x0 = np.load(r'/tests/test_data/x0.npz')
    alpha0 = 1.0
    starting_gradient = -np.array([42.0, 18.0])

    constraints = [-10e6, 10e6]

    strong_wolfe = StrongWolfe(alpha0=alpha0)
    lr, _, _, _, _ = strong_wolfe.search(himmelblau, himmelblau_grad, x0, starting_gradient)

    quasi_wolfe = QuasiWolfe(alpha0=alpha0, constraints=constraints)
    lr2, _ = quasi_wolfe.search(himmelblau, himmelblau_grad, x0, starting_gradient)

    # mt = MoreThuenteTest(alpha0=alpha0, constraints=constraints)
    # lr3, _ = mt.search(himmelblau, himmelblau_grad, x0, starting_gradient)

    assert np.allclose(lr, known_results[0], rtol=0, atol=1e-12)
    assert np.allclose(lr2, known_results[0], rtol=0, atol=1e-12)
    # assert np.allclose(lr3, known_results[1], rtol=0, atol=1e-12)


# failing test
def test_quasi_wolfe():
    repeats = 200

    params1 = [6, 2]
    params2 = [2, 3]

    for i in range(2):
        function_params = params1 if i == 0 else params2
        constraints = [-1, 1]

        for j in range(repeats):
            f, g, h = PolynomialGenerator().generate(function_params[0], function_params[1])
            x0 = np.random.uniform(-1, 1, function_params[0])

            # line_search_params - [x, grad, lr]
            qw = QuasiWolfeTest(constraints=constraints, alpha0=5)
            step_count, error, steps, _, _, _, stage_one_tries, stage_two_currents, stage_two_brackets, line_search_params = \
                grad_descent_test(f, g, x0, learning_rate=qw, epsilon=0.000001, constraints=constraints,
                                  record_trace=True, test_wolfe=True)
            assert not error

            if len(steps) > 2:
                for k in range(0, len(steps) - 1):

                    params = line_search_params[k]
                    assert np.allclose(params[0], steps[k])
                    print(f" params {params}")
                    phi, dphi_minus, dphi_plus = create_phi(f, g, params[0], params[1], constraints)

                    print(f" params {len(params)}")

                    if params[3]:
                        assert check_step_is_quasi_wolfe(1e-4, 0.9, params[0], params[2], phi(0), dphi_plus(0),
                                                         phi(params[2]),
                                                         dphi_minus(params[2]), dphi_plus(params[2]), params[1],
                                                         constraints)

                    new_x = np.clip(params[0] - params[2] * params[1], constraints[0], constraints[1])
                    assert np.allclose(steps[k + 1], new_x, rtol=0, atol=1e-12)

# failing test
def test_quasi_wolfe_2():
    repeats = 200
    dimension = 3
    degree = 12

    for i in range(repeats):
        f, g, h = PolynomialGenerator().generate(dimension, degree)

        x0 = np.random.uniform(-1, 1, dimension)
        alpha0 = 0.1

        constraints = [-1, 1]
        qw = QuasiWolfeTest(constraints=constraints, alpha0=alpha0, max_iterations=3, alphamax=1)

        step_count, error, steps, _, _, _, stage_one_tries, stage_two_currents, stage_two_brackets, line_search_params = \
            grad_descent_test(f, g, x0, learning_rate=qw, epsilon=0.000001, constraints=constraints,
                              record_trace=True, test_wolfe=True, max_steps=3)

        assert not error


# 0,       1           2               3          4   5            6
# steps, step_count, f_calls_count, g_calls_count, x, gradients, objective_values
def test_gd_steps_counted_correctly():
    repeats = 100
    epsilons = np.logspace(0.7, -2, 100)
    dimension = 5
    degree = 3
    constraints = [-1, 1]
    qw = QuasiWolfe(alpha0=.1, alphamax=10, constraints=constraints)

    for i in range(repeats):
        f, g, h = PolynomialGenerator().generate(dimension, degree)
        start = np.random.uniform(-1, 1, dimension)
        x0 = start

        for e in epsilons:
            result = grad_descent(f, g, x0, epsilon=e, learning_rate=qw, record_trace=True, constraints=constraints)
            if result[1] >= 1:
                if result[1] == 1:
                    print(f" Found the actual edge case")  # if just one step
                if result[1] == 2:
                    print(f" steps : {result[0]}")
                    print(f" steps length {len(result[0])}")
                assert len(result[0]) >= 2  # then at least 2 steps in array
                assert not np.allclose(result[0][0], result[0][1], rtol=0,
                                       atol=1e-12)  # and the first two steps are different

            end = result[4]
            x0 = end

def test_lipschitz_calculated_correctly_for_experiments():

    repeats = 100
    for r in range(repeats):
        f, g, h = generate_quadratic(20, mixed=False, linear=True)

        l1 = np.max(np.abs(np.linalg.eigvals(h())))
        max_element = np.max(np.abs(h()))
        assert np.isclose(l1, max_element, rtol=0, atol=1e-12)














# BELOW TESTS ONLY RELEVANT TO THE MY ACTUAL MASTERS PROJECT - IGNORE

















def test_data_generation():
    # check if data gen works

    # run with exact lipschitz for d up to 20

    # run with exact lipschitz for d up to 20

    max_steps = 10000
    epsilon = 1e-6
    repeats = 2
    repeats_one_function = 3
    dimensions = np.unique(np.floor(np.linspace(2, 20, 3, dtype=int))).astype(int)
    qw = QuasiWolfe()

    number_of_conditions = 2

    bounds = [-1, 1]

    # Main loop
    for i in dimensions:
        all_counts = np.zeros((repeats * repeats_one_function, number_of_conditions))
        all_y_values = np.zeros((repeats * repeats_one_function, number_of_conditions, max_steps))
        lipschitz_values = np.zeros((repeats * repeats_one_function, number_of_conditions))

        for j in range(repeats):

            f_1, g_1, h_1 = generate_quadratic(i, linear=True, mixed=True, )
            l1 = np.max(np.abs(np.linalg.eigvals(h_1())))


            for k in range(repeats_one_function):

                iteration_index = j*repeats_one_function + k
                initial_x = np.random.uniform(-1, 1, i)

                steps1, step_count_1, _, _, end1, gradients1, y_values1 = grad_descent(f_1, g_1, initial_x,
                                                                                      learning_rate=Constant(1 / l1),
                                                                                      epsilon=epsilon,
                                                                                      termination_criterion="grad",
                                                                                      verbose=False, constraints=bounds,
                                                                                      record_trace=False,
                                                                                      record_gradients=False,
                                                                                      record_objective_values=True,
                                                                                      max_steps=max_steps)

                steps2, step_count_2, _, _, end2, gradients2, y_values2 = grad_descent(f_1, g_1, initial_x,
                                                                                      learning_rate=Constant(1 / l1),
                                                                                      epsilon=epsilon,
                                                                                      termination_criterion="grad",
                                                                                      verbose=False, constraints=bounds,
                                                                                      record_trace=False,
                                                                                      record_gradients=False,
                                                                                      record_objective_values=True,
                                                                                      max_steps=max_steps)

                assert len(y_values1) == step_count_1 or len(y_values1) == step_count_1 + 1
                all_counts[iteration_index]= step_count_1
                if len(y_values1) < max_steps:
                    to_insert = np.pad(y_values1, (0, max_steps-len(y_values1)),'constant', constant_values=0)
                else:
                    to_insert = y_values1
                all_y_values[iteration_index] = to_insert

                assert len(y_values2) == step_count_2 or len(y_values2) == step_count_2 + 1
                all_counts[iteration_index]= step_count_2
                if len(y_values2) <= max_steps:
                    if len(y_values2) < max_steps:
                        to_insert = np.pad(y_values2, (0, max_steps-len(y_values2)),'constant', constant_values=0)
                    else:
                        to_insert = y_values2
                    all_y_values[iteration_index] = to_insert

                lipschitz_values[iteration_index, 0] = l1
                lipschitz_values[iteration_index, 1] = l1



        # np.savez(data_file_path_base + fr"\dimension_{i}_counts.npz", counts=counts)
        # np.savez(data_file_path_base + fr"\dimension_{i}_distances_x.npz", distances_x=distances_x)
        # np.savez(data_file_path_base + fr"\dimension_{i}_distances_y.npz", f_values=values_y)
        # np.savez(data_file_path_base + fr"\dimension_{i}_lipschitz_values.npz", lipschitz_values=lispschitz_values)
        # np.savez(data_file_path_base + fr"\dimension_{i}_gradients.npz", gradients=gradients)
        assert np.all(np.any(all_y_values, axis=2))
        assert np.all(lipschitz_values)
        assert np.all(all_counts)
    print("All dimensions completed")


def test_i2_calculation():
    # now a single run
    repeats = 100
    for _ in range(repeats):
        max_steps = 100000
        single_epsilon = 1e-6
        # epsilons = [1e-6]
        bounds = [-1, 1]

        qw = QuasiWolfe()

        stopping_conditions = ["grad", "x", "y"]

        dim = 20
        degree = 2
        x0 = np.random.uniform(-1, 1, dim)
        # f,g,h = PolynomialGenerator().generate(dim, degree, all_positive=False)
        f, g, h, = generate_quadratic(dim)

        stopping_condition = stopping_conditions[2]
        l1 = np.max(np.abs(np.linalg.eigvals(h())))
        record_trace = True
        const = Constant(1 / l1)
        # const = Constant(1e-3)
        # 0,       1           2               3          4          5            6          7
        # steps, step_count, f_calls_count, g_calls_count, end, gradients, objective_values, success

        start = x0
        res = grad_descent(f, g, x0, epsilon=single_epsilon, record_trace=record_trace, record_gradients=record_trace,
                           record_objective_values=record_trace, learning_rate=const, constraints=bounds,
                           termination_criterion=stopping_condition, max_steps=max_steps, nesterov=False, verbose=False)

        if res[7]:
            print(f"Success")
        else:
            print("Failure")

        errors = np.abs(res[6][:-1] - res[6][-1])

        forget_log_x = np.log(1 / errors)

        # now try to use rotation
        from scipy.interpolate import splrep, PPoly
        from experiments.utils.miscellaneous import calculate_I2

        # and some y distances


        errors = np.abs(res[6][:-1] - res[6][-1])

        forget_log_x = np.log(1 / errors)

        # ax.plot(forget_log_x, slope * forget_log_x + intercept, c='b', alpha=0.5)

        points = np.array([forget_log_x, np.arange(len(errors))]).T

        # ax.scatter(forget_log_x, np.arange(len(errors)), c='r' )

        # rotate, rotate back and plot

        angle = -np.arctan((len(forget_log_x) - 1) / (forget_log_x[-1] - forget_log_x[0]))

        # rotation matrix
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        translated_points = points - np.array([forget_log_x[0], 0])
        # ax.scatter(translated_points[:,0],translated_points[:,1] , c='r' )

        rotated = np.array([rotation_matrix @ point for point in translated_points])
        print(f" rotated points manual {rotated}")

        # ax.scatter(rotated[:,0], rotated[:,1], c='black', zorder=20)
        #
        # rotation_matrix2 = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
        # rotated_back = np.array([rotation_matrix2 @ point for point in rotated])
        # translated_back = rotated_back + np.array([forget_log_x[0], 0])

        # ax.scatter(translated_back[:,0], translated_back[:,1], c='black', zorder=20)


        # ax.scatter(translated_linse_points[:,0], translated_linse_points[:,1], c='g')

        # plot rotated line
        # rotated_line_points = np.array([rotation_matrix @ point for point in translated_linse_points])
        # ax.scatter(rotated_line_points[:,0], rotated_line_points[:,1], c='b')

        # fit a spline

        cubic_spline = CubicSpline(rotated[:, 0], rotated[:, 1])

        x = np.linspace(rotated[0, 0], rotated[-1, 0], 1000)
        y = cubic_spline(x)
        # ax.hlines(0, rotated[0,0], rotated[-1,0], color='black', linestyles='--')

        # print(f" roots {cubic_spline.roots()}")
        # print(f"rotated first {rotated[0]}")
        # print(f"rotated last {rotated[-1]}")
        # ax.plot(x,y, c='b')

        tck = splrep(rotated[:, 0], rotated[:, 1], s=0)
        ppoly = PPoly.from_spline(tck)
        roots = ppoly.roots(extrapolate=False)

        if len(roots) == 0:
            all_roots = np.array([0, rotated[-1, 0]])
        else:
            all_roots = roots if np.allclose(roots[-1], rotated[-1, 0]) else np.append(roots,
                                                                                              rotated[-1, 0])
            all_roots = all_roots if np.isclose(all_roots[0], 0) else np.insert(all_roots, 0, [0, 0])
        # print(f" safer roots, no extrapolation {all_roots}")

        areas = np.array([cubic_spline.integrate(all_roots[i], all_roots[i + 1]) for i in range(len(all_roots) - 1)])
        triangle_area = 0.5 * np.abs((len(forget_log_x) - 1) * (forget_log_x[-1] - forget_log_x[0]))

        # print(f" areas {areas}")
        # print(f" triangle area {triangle_area}")
        print(f" manual I2 {np.sum(areas) / triangle_area}")
        I2_manual = np.sum(areas)/ triangle_area

        I2 = calculate_I2(input_step_counts=np.arange(len(errors)), input_errors=errors)
        print(f" I2 auto {I2}")
        assert np.isclose(I2, I2_manual, rtol=0, atol=1e-12)






def test_i2_calculation2():
    repeats = 200

    # now a single run

    max_steps = 100000
    single_epsilon = 1e-6
    # epsilons = [1e-6]
    bounds = [-1, 1]

    qw = QuasiWolfe()

    from scipy.interpolate import splrep, PPoly
    from experiments.utils.miscellaneous import calculate_I2, calculate_I1

    stopping_conditions = ["grad", "x", "y"]

    dim = 20
    degree = 5

    stopping_condition = stopping_conditions[2]
    # l1 = np.max(np.abs(np.linalg.eigvals(h())))
    record_trace = True
    # const = Constant(1/l1)
    # const = Constant(1e-3)
    # 0,       1           2               3          4          5            6          7
    # steps, step_count, f_calls_count, g_calls_count, end, gradients, objective_values, success


    indicator_array = np.zeros((repeats, 2))

    for r in range(repeats):

        x0 = np.random.uniform(-1, 1, dim)
        # f,g,h = PolynomialGenerator().generate(dim, degree, all_positive=False)
        f, g, h, = generate_quadratic(dim)
        res_2 = grad_descent(f, g, x0, epsilon=single_epsilon, record_trace=record_trace, record_gradients=record_trace,
                             record_objective_values=record_trace, learning_rate=qw, constraints=bounds,
                             termination_criterion=stopping_condition, max_steps=max_steps, nesterov=False,
                             verbose=False)



        errors2 = np.abs(res_2[6][:-1] - res_2[6][-1])



        i2 = calculate_I2(input_step_counts=np.arange(len(errors2)), input_errors=errors2)
        if not np.isnan(i2):
            assert i2 < 1 and i2 > -1




