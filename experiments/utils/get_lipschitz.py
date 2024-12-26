import numpy as np
from sympy import diff
import sympy as sp


def get_lipschitz(f: sp.Expr, boundary_one: float, boundary_two: float):

    sym = f.free_symbols.pop()
    # print(f"symbol {sym}")
    second_degree_derivative = diff(diff(f))
    # print(f"\n second degree derivative {second_degree_derivative} \n")
    third_degree_derivative = diff(second_degree_derivative)
    zeros = sp.solve(third_degree_derivative, sym)
    # print(f"\n zeros {zeros} \n")
    if len(zeros) > 0:
        ys = np.abs(np.array([second_degree_derivative.evalf(subs={sym: val}) for val in zeros if -1 <= val <= 1]))
    else:
        ys = []
    max_y = np.max(ys) if len(ys) > 0 else 0
    # print(f"max_y {max_y}")
    value_boundary_one = np.abs(second_degree_derivative.evalf(subs={sym: boundary_one}))
    # print(f"boundary_one {boundary_one}")
    # print(f"\n value_boundary_one {value_boundary_one}")
    value_boundary_two = np.abs(second_degree_derivative.evalf(subs={sym: boundary_two}))
    # print(f"boundary_two {boundary_two}")
    # print(f"\n value_boundary_two {value_boundary_two}")
    # print(f"\n final values = {np.array([max_y, value_boundary_one, value_boundary_two])}")

    return np.max(np.array([max_y, value_boundary_one, value_boundary_two]).astype(np.float64))
