import numpy as np
import pytest
from grad_descent_test import grad_descent_test
from function_generation.PolynomialGenerator import PolynomialGenerator


def test_julia_grad_descent_same_as_manual():
    # read in the file with the path C:\Users\Owner\Desktop\MastersProject\tests\test_data\test1.txt
    test_data_path = r"C:\Users\Owner\Desktop\MastersProject\tests\test_data\test1.txt"
    number_of_dims = 4

    def generate_polynomial(dimension, coeffs_quad_p, coeffs_linear_p):

        def f(x):
            return np.sum([coeffs_quad_p[i] * x[i] ** 2 for i in range(dimension)])

        def gf(x):
            return np.array([2 * coeffs_quad_p[i] * x[i] + coeffs_linear_p[i] for i in range(dimension)])

        return f, gf

    with open(test_data_path, 'r') as file:
        line = file.readline()

        while line:
            coeffs_quad = []
            coeffs_linear = []
            start = []
            steps = []
            if line.startswith(" - - -"):
                for i in range(number_of_dims):
                    coeff = float(file.readline())
                    coeffs_quad.append(coeff)
                _ = file.readline()
                for i in range(number_of_dims):
                    coeff = float(file.readline())
                    coeffs_linear.append(coeff)
                _ = file.readline()
                for i in range(number_of_dims):
                    start_i = float(file.readline())
                    start.append(start_i)
                for i in range(3):
                    _ = file.readline()
                step = file.readline()
                while step != "\n" and not step.startswith(" "):
                    steps.append([float(s_i.strip()) for s_i in step.split(",")])
                    step = file.readline()
                line = file.readline()

                f, gf = generate_polynomial(number_of_dims, coeffs_quad, coeffs_linear)
                end, steps_py, tries = grad_descent(f, gf, np.array(start), epsilon=0.001, termination_criteria='x',
                                                    max_steps=1000, LR=0.01)
                assert np.allclose(steps_py, np.array(steps))

            line = file.readline()


def test_julia_grad_descent_with_julia_polynomial_auto_generation_same_as_python():
    test_data_path_1 = r"C:\Users\Owner\Desktop\MastersProject\tests\test_data\test_polynomial_1.txt"
    test_data_path_2 = r"C:\Users\Owner\Desktop\MastersProject\tests\test_data\test_polynomial_2.txt"

    params1 = (2, 3)
    params2 = (3, 5)

    for i in range(1, 3):
        test_data_path = test_data_path_1 if i == 1 else test_data_path_2
        params = params1 if i == 1 else params2

        with open(test_data_path, 'r') as file:
            line = file.readline()

            while line:
                coeffs = []
                start = []
                steps = []
                if line.startswith(" - - -"):
                    while not line.startswith("\n"):
                        coeffs.append(np.array(list(map(float, line.split(',')))))

                    for i in range(params[1]):
                        start.append(float(file.readline()))
                    for i in range(3):
                        _ = file.readline()
                    step = file.readline()
                    while step != "\n" and not step.startswith(" "):
                        steps.append([float(s_i.strip()) for s_i in step.split(",")])
                        step = file.readline()
                    line = file.readline()

                    f, gf = PolynomialGenerator().generate(params[1], params[0], coefficients=coeffs)
                    end, steps_py, tries = grad_descent(f, gf, np.array(start), epsilon=0.001, termination_criteria='x',
                                                        max_steps=1000, LR=0.01)
                    assert np.allclose(steps_py, np.array(steps))

                line = file.readline()

def test_julia_custom_grad_descent_same_as_manual():
    test_data_path = r"C:\Users\Owner\Desktop\MastersProject\tests\test_data\test1.txt"
    epsilon = 0.01
    learing_rate = 0.01

    with open(test_data_path, 'r') as file:
        line = file.readline()

        while line:
            coeffs = []
            start = []
            steps = []
            if line.startswith(" - - -"):
                while not line.startswith("\n"):
                    coeffs.append(np.array(list(map(float, line.split(',')))))
                for i in range(4):
                    start.append(float(file.readline()))
                for i in range(3):
                    _ = file.readline()
                step = file.readline()
                while step != "\n" and not step.startswith(" "):
                    steps.append([float(s_i.strip()) for s_i in step.split(",")])
                    step = file.readline()
                line = file.readline()

                f, gf = PolynomialGenerator().generate(4, 2, coefficients=coeffs)
                end, steps_py, tries = grad_descent(f, gf, np.array(start), epsilon=epsilon, termination_criteria='grad',
                                                    max_steps=1000, LR=epsilon)
                assert np.allclose(steps_py, np.array(steps))

            line = file.readline()