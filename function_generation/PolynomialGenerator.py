import numpy as np
from numpy.polynomial import polynomial
import matplotlib.pyplot as plt
import itertools
import sympy as sp
from function_generation.F import F
from function_generation.GradF import GradF
from function_generation.HessF import HessF


class PolynomialGenerator:
    a = sp.Symbol('a', real=True)
    b = sp.Symbol('b', real=True)
    c = sp.Symbol('c', real=True)
    d = sp.Symbol('d', real=True)
    e = sp.Symbol('e', real=True)
    f = sp.Symbol('f', real=True)
    g = sp.Symbol('g', real=True)
    h = sp.Symbol('h', real=True)
    i = sp.Symbol('i', real=True)
    j = sp.Symbol('j', real=True)
    k = sp.Symbol('k', real=True)
    l = sp.Symbol('l', real=True)
    m = sp.Symbol('m', real=True)
    n = sp.Symbol('n', real=True)
    o = sp.Symbol('o', real=True)
    p = sp.Symbol('p', real=True)
    q = sp.Symbol('q', real=True)
    r = sp.Symbol('r', real=True)
    s = sp.Symbol('s', real=True)
    t = sp.Symbol('t', real=True)
    u = sp.Symbol('u', real=True)
    v = sp.Symbol('v', real=True)
    w = sp.Symbol('w', real=True)
    x = sp.Symbol('x', real=True)
    y = sp.Symbol('y', real=True)
    z = sp.Symbol('z', real=True)

    all_possible_vars = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p,
                         q, r, s, t, u, v, w, x, y, z]

    def __init__(self):
        pass

    def generate(self, n_vars: int, degree: int,
                 all_positive: bool = False,
                 use_float_32: bool = False,
                 uniform_distrib_multiplier: float = 1.0,
                 coefficient_multipliers: np.ndarray | None = None,
                 coefficients: list | None = None,
                 get_hessian: bool = False,
                 constant_factor: float = 0.0,
                 include_degrees: list | None = None
                 ):
        assert 0 < n_vars < 25, "n_vars must be between 1 and 24"

        if include_degrees is None:
            include_degrees = list(range(1, degree + 1))

        variabs = self.all_possible_vars[:n_vars]

        var_ids = [i for i in range(n_vars)]
        # print(f" var_ids: {var_ids}")
        # print(f" variabs: {variabs}")

        combs = [np.array(list(itertools.combinations_with_replacement(var_ids, d))) for d
                 in include_degrees]
        # print(f" combs: {combs}")

        if not coefficients:
            if all_positive:
                coeffs = [uniform_distrib_multiplier * np.random.rand(len(c)) for c in combs]
            else:
                coeffs = [uniform_distrib_multiplier * 2 * np.random.rand(len(c)) - 1 for c in combs]
        else:
            coeffs = coefficients

        if use_float_32:
            coeffs = [np.float32(c) for c in coeffs]

        if coefficient_multipliers is not None:
            all_functions = []
            for c in coefficient_multipliers:
                f = F(variabs, coeffs, combs, uniform_distrib_multiplier, coefficient_multiplier=c, constant_factor=constant_factor)
                grad_f = GradF(f, variabs)
                hess_f = HessF(grad_f, get_hessian=get_hessian)
                all_functions.append((f, grad_f, hess_f))
            return all_functions

        else:
            f = F(variabs, coeffs, combs, uniform_distrib_multiplier)
            grad_f = GradF(f, variabs)
            hess_f = HessF(grad_f, get_hessian=get_hessian)
            return f, grad_f, hess_f
