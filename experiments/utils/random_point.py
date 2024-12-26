from function_generation.PolynomialGenerator import PolynomialGenerator
from grad_descent_test import grad_descent
import numpy as np


def random_point_in_hypersphere(circle_radius=1, dimensions=2):
    point = np.zeros(dimensions)
    r = np.random.uniform(0, circle_radius)
    theta = np.random.uniform(0, 2 * np.pi)
    point[0] = circle_radius * np.cos(theta)
    point[1] = circle_radius * np.sin(theta)

    for i in range(2, dimensions):
        phi = np.random.uniform(0, 2 * np.pi)
        point[i] = r * np.cos(phi)

    return point


def random_point_in_hypercube(cube_radius=1, dimensions=2):
    point = np.zeros(dimensions)
    for i in range(dimensions):
        point[i] = np.random.uniform(-cube_radius, cube_radius)

    return point
