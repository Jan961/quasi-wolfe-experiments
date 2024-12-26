import numpy as np
from sympy.codegen.fnodes import dimension
from sympy.matrices.expressions.blockmatrix import bounds

from experiments.utils.generate_quadratics import generate_quadratic
from function_generation.PolynomialGenerator import PolynomialGenerator
from grad_descent import grad_descent, project_gradient
from line_searches.Constant import (Constant)
from tests.test_utils.scipy_grad_descent import gradient_descent, gradient_descent_chatGPT
from tests.test_utils.test_functions import create_convex_function_1, create_convex_function_2


#  0        1           2               3          4        5           6             7
# steps, step_count, f_calls_count, g_calls_count, x, gradients, objective_values, success




# my gradient descent against manual for 1d functions
def test_gradient_descent_with_custom_function_1d():
    constraints = [-1, 1]
    for j in range(50):

        f, f_prime, _ = PolynomialGenerator().generate(n_vars=1, degree=2, all_positive=True)
        coefficients = f.coeffs
        constant_factor = f.constant_factor

        print(f" coefficients {coefficients}")

        f_alt = lambda x: np.sum(
            np.array([coefficients[-1][0] * x[0] ** 2 + coefficients[-2][0] * x[0] + constant_factor],
                     dtype=np.float64))
        f_prime_alt = lambda x: 2 * coefficients[-1][0] * x[0] + coefficients[-2][0]

        for i in range(50):
            start = np.random.rand(1)
            epsilon = np.random.uniform(0.0001, 0.01)
            LR = 1 / (2 * coefficients[-1])

            res1 = grad_descent(fn=f, fn_grad=f_prime, start=start, epsilon=epsilon, max_steps=1000,
                                constraints=constraints, learning_rate=Constant(LR),
                                termination_criterion="x", record_trace=True, verbose=True)
            # print(f" auto steps {steps}")
            res2 = grad_descent(fn=f_alt, fn_grad=f_prime_alt, start=start, epsilon=epsilon,
                                max_steps=1000,
                                constraints=constraints, learning_rate=Constant(LR),
                                termination_criterion="x", record_trace=True, verbose=True)
            # print(f" alt steps {steps_alt}")
            assert np.allclose(res1[0], res2[0], rtol=0, atol=1e-12)


# a sanity check to again confirm that the projected gradient descent is implemented correctly
def test_objective_always_decreases_with_step_1_over_L():

    repeats = 200
    dimensions = [2,10,20]

    for _ in range(repeats):
        for dim in dimensions:
            f, g, h = generate_quadratic(dim)
            x0 = np.random.uniform(-1, 1, dim)
            l1 = np.max(np.abs(np.linalg.eigvals(h())))
            const = Constant(1 / l1)
            max_steps = 10000
            epsilon = 1e-6
            bounds = [-1, 1]

            res = grad_descent(f, g, x0, epsilon=epsilon, record_trace=False, record_gradients=False,
                           record_objective_values=True, learning_rate=const, constraints=bounds,
                           termination_criterion="grad", max_steps=max_steps, nesterov=False, verbose=False)
            y_values = res[6]
            for i in range(1, len(y_values)):
                assert np.all(np.diff(y_values) < 0)




#  test my non-projected gradient descent against an external code for some quadratic functions
def test_gradient_descent_constant_lr():

    max_steps = 100

    for j in range(20):
        f, g, h = generate_quadratic(2, convex=True)
        x0 = np.random.uniform(-1, 1, 2)
        l1 = np.max(np.abs(np.linalg.eigvals(h())))
        const = Constant(1 / l1)
        epsilon = 1e-6



        #  my GD
        res = grad_descent(f, g, x0, epsilon=epsilon, record_trace=False, record_gradients=False,
                       record_objective_values=True, learning_rate=const,
                       termination_criterion="grad", max_steps=max_steps, nesterov=False, verbose=False)
        y_values = res[6]
        gradients = res[5]
        steps = res[0]


        # external GD
        all_x_i, all_y_i, all_f_i = gradient_descent(x0, f, g, wolfe=False, step_size=1/l1, max_iter=max_steps)

        # check that the objective values are the same; -1 bc of an extra step in my implementation
        for i in range(len(y_values)-1):
            assert np.isclose(y_values[i], all_f_i[i], atol=1e-6)

        # check that the gradients are the same
        for i in range(len(gradients)-1):
            assert np.isclose(gradients[i], np.linalg.norm(g(np.array([all_x_i[i], all_y_i[i]]))), atol=1e-6)

        # check that the steps are the same
        for i in range(len(steps)-1):
            assert np.allclose(steps[i], np.array([all_x_i[i], all_y_i[i]]), atol=1e-6)


#  test my non-projected gradient descent against an external code for some different convex function in 2D
def test_gradient_quadratics_2D():
    max_steps = 100
    step_size = 1e-3


    f, g = create_convex_function_1()
    x0 = np.random.uniform(-1, 1, 2)
    const = Constant(step_size)
    epsilon = 1e-6

    #  my GD
    res = grad_descent(f, g, x0, epsilon=epsilon, record_trace=False, record_gradients=False,
                       record_objective_values=True, learning_rate=const,
                       termination_criterion="grad", max_steps=max_steps, nesterov=False, verbose=False)
    y_values = res[6]
    gradients = res[5]
    steps = res[0]

    # external GD
    all_x_i, all_y_i, all_f_i = gradient_descent(x0, f, g, wolfe=False, step_size=step_size, max_iter=max_steps)

    # check that the objective values are the same; -1 bc of an extra step in my implementation
    for i in range(len(y_values) - 1):
        assert np.isclose(y_values[i], all_f_i[i], atol=1e-6)

    # check that the gradients are the same
    for i in range(len(gradients) - 1):
        assert np.isclose(gradients[i], np.linalg.norm(g(np.array([all_x_i[i], all_y_i[i]]))), atol=1e-6)

    # check that the steps are the same
    for i in range(len(steps) - 1):
        assert np.allclose(steps[i], np.array([all_x_i[i], all_y_i[i]]), atol=1e-6)



#  test my non-projected gradient descent against external implementation (chatGPT) for a  convex function in 3D
def test_gradient_quadratics_3D():
    max_steps = 100
    step_size = 1e-3


    f, g = create_convex_function_2()
    x0 = np.random.uniform(-1, 1, 3)
    const = Constant(step_size)
    epsilon = 1e-6

    #  my GD
    res = grad_descent(f, g, x0, epsilon=epsilon, record_trace=False, record_gradients=False,
                       record_objective_values=True, learning_rate=const,
                       termination_criterion="grad", max_steps=max_steps, nesterov=False, verbose=False)
    y_values = res[6]
    gradients = res[5]
    steps = res[0]

    # chatGPT GD
    result, history  = gradient_descent_chatGPT(f, g, x0, learning_rate=step_size, max_iterations=max_steps, tolerance=epsilon)

    print(f"history {history}")


    all_x_i = [x[1][0] for x in history]
    all_y_i = [x[1][1] for x in history]
    all_z_i = [x[1][2] for x in history]
    all_f_i = [f(x[1]) for x in history]

    print(f"all_x_i {all_x_i}")
    print(f"all_y_i {all_y_i}")
    print(f"all_z_i {all_z_i}")
    print(f"all_f_i {all_f_i}")

    print(f"y values - auto clacl : {y_values}")
    # check that the objective values are the same; -1 bc of an extra step in my implementation
    for i in range(len(y_values) - 1):
        assert np.isclose(y_values[i], all_f_i[i], atol=1e-6)

    # check that the gradients are the same
    for i in range(len(gradients) - 1):
        assert np.isclose(gradients[i], np.linalg.norm(g(np.array([all_x_i[i], all_y_i[i], all_z_i[i]]))), atol=1e-6)

    # check that the steps are the same
    for i in range(len(steps) - 1):
        assert np.allclose(steps[i], np.array([all_x_i[i], all_y_i[i]]), atol=1e-6)


# check the projected GD gives the same results as the non-projected one for a scenario when no projection is needed
def test_projected_2D():
    max_steps = 100
    step_size = 1e-3

    test_functions = [create_convex_function_1(), create_convex_function_2()]
    dimensions = [2, 3,4]

    f_test,g_test,h_test = generate_quadratic(4, convex=True)
    test_functions.append((f_test,g_test))
    print(f"test functions {test_functions}")

    for i in range(len(test_functions)):

        f,g = test_functions[i]
        dimension = dimensions[i]

        x0 = np.random.uniform(-1, 1, dimension)
        const = Constant(step_size)
        epsilon = 1e-6

        #  my GD
        res = grad_descent(f, g, x0, epsilon=epsilon, record_trace=False, record_gradients=False,
                           record_objective_values=True, learning_rate=const,
                           termination_criterion="grad", max_steps=max_steps, nesterov=False, verbose=False)
        y_values = res[6]
        gradients = res[5]
        steps = res[0]

        # my projected GD
        res_proj = grad_descent(f, g, x0, epsilon=epsilon, record_trace=False, record_gradients=False,
                              record_objective_values=True, learning_rate=const, constraints=(-1,1),
                                termination_criterion="grad", max_steps=max_steps, nesterov=False, verbose=False)

        y_values_proj = res_proj[6]
        gradients_proj = res_proj[5]
        steps_proj = res_proj[0]

        # check that the objective values are the same; -1 bc of an extra step in my implementation
        for i in range(len(y_values)):
            assert np.isclose(y_values[i], y_values_proj[i], atol=1e-6)

        # check that the gradients are the same
        for i in range(len(gradients)):
            assert np.isclose(gradients[i], gradients_proj[i], atol=1e-6)

        # check that the steps are the same
        for i in range(len(steps)):
            assert np.allclose(steps[i], steps_proj[i], atol=1e-6)


# check gradient projection for GD for some points

def test_gradient_projection():

    # assume the constraints are (-1,1) in all dimensions
    # for a point 1,1 with grad -1,-1 the projected grad should be 0,0

    x = np.array([1,1])
    grad = np.array([-1,-1])
    proj_grad = project_gradient(grad, (-1,1), x)
    print(f"proj_grad {proj_grad}")
    assert np.allclose(proj_grad, np.array([0,0]))

    # for a point 1,0.5 with grad -0.2,-0.2 the projected grad should be 0,-0.2
    x = np.array([1,0.5])
    grad = np.array([-0.2,-0.2])
    proj_grad = project_gradient(grad, (-1,1), x)
    print(f"proj_grad {proj_grad}")
    assert np.allclose(proj_grad, np.array([0,-0.2]))

    # for a point -1,0 with grad 0.2,-0.2 the projected grad should be 0,-0.2
    x = np.array([-1,0])
    grad = np.array([0.2,-0.2])
    proj_grad = project_gradient(grad, (-1,1), x)
    print(f"proj_grad {proj_grad}")
    assert np.allclose(proj_grad, np.array([0,-0.2]))

    # for a point -1,-1 with grad 1,1 the projected grad should be 0,0
    x = np.array([-1,-1])
    grad = np.array([1,1])
    proj_grad = project_gradient(grad, (-1,1), x)
    print(f"proj_grad {proj_grad}")
    assert np.allclose(proj_grad, np.array([0,0]))

    # for a point 1,1, with grad 1,1 the projected grad should be 1,1
    x = np.array([1,1])
    grad = np.array([1,1])
    proj_grad = project_gradient(grad, (-1,1), x)
    print(f"proj_grad {proj_grad}")
    assert np.allclose(proj_grad, np.array([1,1]))

    # for a point -1,-0.2 with grad -0.2,-0.2 the projected grad should be -0.2,-0.2
    x = np.array([-1,-0.2])
    grad = np.array([-0.2,-0.2])
    proj_grad = project_gradient(grad, (-1,1), x)
    print(f"proj_grad {proj_grad}")
    assert np.allclose(proj_grad, np.array([-0.2,-0.2]))


# test projected GD against external implementation (ChatGPT)
def test_projected_gd_2():
    p_generator = PolynomialGenerator()
    test_functions = [p_generator.generate(2,2), p_generator.generate(2,3), p_generator.generate(3,3)]

    dimensions = [2,2,3]
    max_steps = 100

    for i in range(len(test_functions)):
        f, g = test_functions[i][:2]
        dimension = dimensions[i]

        x0 = np.random.uniform(-1, 1, dimension)
        const = Constant(1e-3)
        epsilon = 1e-6

        #  my GD
        res = grad_descent(f, g, x0, epsilon=epsilon, record_trace=False, record_gradients=False,
                           record_objective_values=True, learning_rate=const,  constraints=(-1,1),
                           termination_criterion="grad", max_steps=max_steps, nesterov=False, verbose=False)

        y_values = res[6]
        gradients = res[5]
        steps = res[0]

        # chatGPT GD
        result, history  = gradient_descent_chatGPT(f, g, x0, learning_rate=1e-3, max_iterations=max_steps, tolerance=epsilon, projected=True)

        chatGPT_steps = [x[1] for x in history]
        chatGPT_y_values = [f(x) for x in chatGPT_steps]

        # check that the objective values are the same; -1 bc of an extra step in my implementation
        for i in range(len(y_values)-1):
            assert np.isclose(y_values[i], chatGPT_y_values[i], atol=1e-6)

        # check if the steps are the same
        for i in range(len(steps)-1):
            assert np.allclose(steps[i], chatGPT_steps[i], atol=1e-6)

