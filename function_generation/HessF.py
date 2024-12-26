import sympy as sp
import numpy as np

from function_generation.GradF import GradF


class HessF:

    def __init__(self, gradf: GradF | list, variables: list = None, get_hessian: bool = True, is_test: bool = False):
        self.gradf = gradf
        self.variables = variables if variables is not None else gradf.variables
        self.is_test = is_test
        if get_hessian:
            self.expressions = sp.zeros(len(self.variables), len(self.variables))

            for col in range(len(self.variables)):
                for row in range(col, len(self.variables)):
                    if not is_test:
                        derivative = self.gradf.expressions[col].diff(self.variables[row])
                    else:
                        derivative = self.gradf[col].diff(self.variables[row])
                    self.expressions[row, col] = derivative
                    if row != col:
                        self.expressions[col, row] = derivative

            self.lambdas = sp.lambdify(self.variables, self.expressions, "numpy")

        else:
            self.expressions = None

    def __call__(self, x: np.ndarray):
        if self.expressions is not None:
            return self.lambdas(*x)
        else:
            return None

    def latex(self):
        return sp.print_latex(self.expressions)
