import numpy as np

from experiments.utils.generate_quadratics import generate_quadratic
from experiments.utils.get_lipschitz import get_lipschitz
from function_generation.PolynomialGenerator import PolynomialGenerator
from function_generation.HessF import HessF
from grad_descent import grad_descent
from grad_descent_test import grad_descent_test
from line_searches.QuasiWolfe import QuasiWolfe
from line_searches.StrongWolfe import StrongWolfe
import sympy as sp
from line_searches.Constant import Constant
from line_searches.QuasiWolfeTest import QuasiWolfeTest
from line_searches.utils import eliminate_kinks, get_kink_step_alphas, find_kink_step_indices, check_is_k_step, \
    create_phi, create_omega, check_step_is_quasi_wolfe
from experiments.utils.miscellaneous import trim_a_results_array




def test_trim_results():

    repeats = 5
    conditions = 3
    max_steps= 4

    a1 = np.random.uniform(-1,1, (repeats, conditions))
    a2 = np.random.uniform(-1,1, (repeats, conditions))
    a3 = np.zeros((repeats,conditions))
    a3[1,1] = 0.5
    a4 = np.zeros((repeats,conditions))

    test_results_1 = np.dstack((a1,a2,a3,a4))
    trimmed1 = trim_a_results_array(test_results_1)
    assert trimmed1.shape == (repeats, conditions,3)

    test_results_2 = np.dstack((a1,a4,a4,a4))
    trimmed2 = trim_a_results_array(test_results_2)
    assert trimmed2.shape == (repeats,conditions,1)

    test_results_3 = np.dstack((a3,a3,a3,a3))
    trimmed3 = trim_a_results_array(test_results_3)
    assert trimmed3.shape == (repeats,conditions,4)


    test_results_4 = np.dstack((a4,a4,a4,a4))
    trimmed4 = trim_a_results_array(test_results_4)
    assert trimmed4 is None



