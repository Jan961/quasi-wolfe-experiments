import sympy as sp
import numpy as np
from sympy import diff

from function_generation.F import F


class GradF:
    def __init__(self, function: F, variables: list = None):
        self.function = function
        self.variables = variables if variables is not None else function.variables
        self.expressions = sp.zeros(len(self.variables), 1)

        for i in range(len(self.variables)):
            self.expressions[i] = diff(self.function.expression, self.variables[i])

        self.lambdas = sp.lambdify(self.variables, self.expressions, "numpy")

    def __call__(self,x):
        # print(f" len variables {len(self.variables)}")
        assert len(x) == len(self.variables), "number of variables must match number of arguments"
        # print(f" args : {x},")
        # print(f" args[0] : {x[0]},")
        # print(f"  x : {x}")

        # print(f"grad[0] before conversion is {values_list[0]},  {type(values_list[0])}")
        return np.array(self.lambdas(*x)).squeeze().astype(np.float64)

    def latex(self):
        return sp.print_latex(self.expressions)
