# create some functions convex on the interval [-1, 1]
import numpy as np


def create_convex_function_1():


    def f(x: np.ndarray):
        return x[0] ** 2 - np.cos(x[1]) + 2 * x[0] * x[1]

    def g(x: np.ndarray):
        return np.array([2 * x[0] + 2 * x[1], np.sin(x[1]) + 2 * x[0]])

    return f, g

def create_convex_function_2():

    def f(x: np.ndarray):
        return x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + 2 * x[0] * x[1] + 2 * x[0] * x[2] + 2 * x[1] * x[2]

    def g(x: np.ndarray):
        return np.array([2 * x[0] + 2 * x[1] + 2 * x[2], 2 * x[1] + 2 * x[0] + 2 * x[2], 2 * x[2] + 2 * x[0] + 2 * x[1]])

    return f, g