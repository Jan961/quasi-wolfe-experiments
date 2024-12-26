import numpy as np
import matplotlib.pyplot as plt
from grad_descent_test import grad_descent
from plotting.plot_2d import plot_grad_descent_2d

def calculate_upper_bound_on_steps_x(coefficients, start_x, epsilon):
    # note that all the quadratic terms are positive

    largest_coefficient = np.max(coefficients[:, 2])
    smallest_coefficient = np.min(coefficients[:, 2])
    index_of_lipschitz = np.argmax(coefficients[:, 2])
    ratios = np.delete(coefficients[:, 2], index_of_lipschitz) / largest_coefficient
    terms = np.delete(start_x, index_of_lipschitz) * ratios + (
            np.delete(coefficients[:, 1], index_of_lipschitz) / (2 * largest_coefficient))
    sum = np.linalg.norm(terms)
    numerator = np.log2(epsilon / sum)

    denominator = np.log2(1 - smallest_coefficient / largest_coefficient)
    expected = np.ceil(numerator / denominator) + 2
    # print(f"expected {expected}")
    return expected if expected > 2 else 2


def calculate_upper_bound_on_steps_grad(coefficients, start_x, epsilon):
    largest_coefficient = np.max(coefficients[:, 2])
    smallest_coefficient = np.min(coefficients[:, 2])
    index_of_lipschitz = np.argmax(coefficients[:, 2])
    eigenvalues = 2 * np.delete(coefficients[:, 2], index_of_lipschitz)
    terms = np.delete(start_x, index_of_lipschitz) * eigenvalues + np.delete(coefficients[:, 1], index_of_lipschitz)
    sum = np.linalg.norm(terms)
    numerator = np.log2(epsilon / sum)

    denominator = np.log2(1 - smallest_coefficient / largest_coefficient)
    expected = np.ceil(numerator / denominator) + 3
    return expected if expected > 3 else 3


def experiment_one_function(no_boundary=True, stopping_condition="x", dimensions=2, restarts=100, number_of_epsilons=20,
                            plot=False, plot_when_diff_greater_than=0):
    coefficients = np.concatenate((np.random.uniform(-1, 1, (dimensions, 2)), np.random.uniform(0, 1, (dimensions, 1))),
                                  axis=1)

    lipschitz = 2 * np.abs(np.max(coefficients[:, 2]))

    if no_boundary:
        centres = -coefficients[:, 1] / (2 * coefficients[:, 2])
        mask = np.logical_or(centres > 1, centres < -1)
        centres_outside = np.where(mask, centres, 0)
        coefficients[:, 1] = coefficients[:, 1] + 2 * coefficients[:, 2] * centres_outside
        coefficients[:, 0] = coefficients[:, 0] + coefficients[:, 2] * centres_outside ** 2 + coefficients[:,
                                                                                              1] * centres_outside

    f, f_prime = gen_quadratic(coefficients, np.zeros((dimensions * (dimensions - 1) // 2,)))

    epsilons = np.logspace(-4, -9, number_of_epsilons)
    constraints = np.tile(np.array([[-1, 1]]), (dimensions, 1))

    results = np.zeros((restarts, len(epsilons), 2))
    for r in range(restarts):
        start = np.random.uniform(-1, 1, dimensions)
        for id, e in enumerate(epsilons):
            x, steps, _ = grad_descent(f, f_prime, start, epsilon=e, max_steps=1000,
                                       termination_criteria=stopping_condition, learning_rate="constant",
                                       LR=1 / lipschitz, constraints=constraints)
            if stopping_condition == "x":
                max_number_of_steps = calculate_upper_bound_on_steps_x(coefficients, start, e)
            else:
                max_number_of_steps = calculate_upper_bound_on_steps_grad(coefficients, start, e)

            results[r, id, 0] = steps.shape[0]
            results[r, id, 1] = max_number_of_steps
            if plot and max_number_of_steps - len(steps) < -plot_when_diff_greater_than:
                print(f"coefficients {coefficients}")
                print(f" steps {steps}")
                print(f" number of steps {len(steps)}")
                print(f" max number of steps {max_number_of_steps}")
                print(f"learning rate: {1 / lipschitz} ")
                plot_grad_descent_2d(steps, f, bounds=constraints)
                # return coefficients, steps, max_number_of_steps, (1 / lipschitz)

    return results
