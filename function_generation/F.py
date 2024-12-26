import itertools

import numpy as np
import sympy as sp
from collections import Counter
from sympy import lambdify


class F:

    def __init__(self, variables: list,
                 coeffs: list, combs: list,
                 multiplier: float,
                 coefficient_multiplier: float = 1,
                 constant_factor: float = None):

        self.multiplier = multiplier
        self.coefficient_multiplier = coefficient_multiplier
        self.variables = variables  # list of sympy symbols
        self.var_ids = [i for i in range(len(variables))]
        self.coeffs = coeffs
        # print(f" coeffs: {coeffs}")
        self.combs = combs

        self.constant_factor = constant_factor if constant_factor else multiplier * np.random.rand() * (
                variables[0] ** 0)

        expr = self.constant_factor

        for i in range(len(self.combs)):
            comb = self.combs[i]
            for j in range(len(comb)):
                c = Counter(comb[j])
                monomial_lst = [self.variables[k] ** v for k, v in c.items()]
                expr += np.prod(monomial_lst) * coeffs[i][j] * self.coefficient_multiplier

        self.expression = expr
        self.lambda_expr = lambdify(self.variables, self.expression, "numpy")

    # get precise evaluation as outlined in the documentation, parameter epsilon controls the precision
    # chop=True removes rounding errors smaller than the set precision
    def __call__(self, x):
        # assert len(x) == len(self.variables), "number of variables must match number of arguments"
        # out = self.expr.evalf(15,subs={self.vars[i]: x[i] for i in range(len(self.vars))}, chop=True)
        # print(f" y with eval is {out}")
        # print(f"  x : {x}")
        # print(f" y with lambda is {out_2}")
        return np.float64(self.lambda_expr(*x))

    def latex(self):
        return sp.print_latex(self.expression)

    def lambda_for_plotting_2d(self):
        return lambdify([self.variables[0], self.variables[1]], self.expression, "numpy")
