import numpy as np
import matplotlib.pyplot as plt
from plotting.plot_2d import plot_grad_descent_2d
from math import factorial


def comb_r(n: int, r: int) -> int:
    return factorial(n + r - 1) // (factorial(r) * factorial(n - 1))


def generate_quadratic(dimension: int,
                       squared: bool = True,
                       linear: bool = True,
                       mixed: bool = True,
                       convex=False,
                        return_coeffs = False  ) -> tuple:
    assert squared or mixed, "One of mixed or squared must be non-zero"
    if convex:
        mixed = False
        linear = False


    if mixed:
        A = np.random.uniform(-0.5, 0.5, (dimension, dimension))
        A[np.diag_indices(dimension)] *= 2
    else:
        A = np.zeros((dimension, dimension))
        if not convex:
            A[np.diag_indices(dimension)] = np.random.uniform(-1, 1, dimension)
        else:
            A[np.diag_indices(dimension)] = np.random.uniform(0, 1, dimension)


    if not squared:
        A[np.diag_indices(dimension)] = 0

    if linear:
        b = np.random.uniform(-1, 1, dimension)
    else:
        b = np.zeros(dimension)

    # Generate random coefficients for linear and homogeneous parts
    def f(x: np.ndarray):
        return x @ (A @ x) + b @ x

    def g(x: np.ndarray):
        return (A + A.T) @ x + b

    def h():
        return A + A.T

    if return_coeffs:
        return f, g, h, A, b
    else:
        return f, g, h
