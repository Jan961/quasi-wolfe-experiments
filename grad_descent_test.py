from typing import Callable

import numpy as np
from line_searches.Constant import Constant
from line_searches.QuasiWolfeTest2 import QuasiWolfeTest2
from line_searches.StrongWolfe import StrongWolfe
from line_searches.utils import check_step_is_quasi_wolfe, create_phi


# from line_searches.QuasiMoreThuente

# learning rates =
#        "Armijo" - check is satisfies Armijo condition if not halve
#        "constant" - constant learning rate
#        "progress bound" - check is satisfies progress bound if not halve

# termination criteria =
#        "grad" - terminate when gradient is small
#        "x star" - terminate when x is close to x_star
#        "y" - terminate when function value change is small
#        "x" - terminate when change in x is small
def grad_descent_test(fn,
                      fn_grad,
                      start: np.ndarray,
                      epsilon: float = 0.01,
                      learning_rate=Constant(0.01),
                      termination_criteria: str = "grad",
                      norm=2,
                      x_star: np.ndarray = None,
                      max_steps=1000,
                      constraints: np.ndarray | list | tuple = None,  # shape ( 2)
                      nesterov=False,
                      verbose=False,
                      is_gradient_projected=True,
                      record_trace=False,
                      test_wolfe=True):
    stage_one_tries, stage_two_currents, stage_two_brackets, line_search_params, success_record \
        = [], [], [], [], []

    error = False

    if nesterov:
        assert isinstance(learning_rate, Constant), "Nesterov only supports constant learning rate"
    if constraints is not None:
        assert len(constraints) == 2, "constraints must be of shape (2)"

    assert isinstance(start, np.ndarray), "start must be a numpy array"

    if record_trace:
        steps = []

    step_count = 0
    f_calls_count = 0
    g_calls_count = 0
    x = start

    while True:

        # print(f" this is x outside {x[0]}, {type(x[0])}")

        grad = fn_grad(x)
        y = fn(x)

        # print(f" grad: {grad}")
        # print(f"x: {x}")

        # if len(x) == 1:
        #     grad = np.array([grad])
        # x = np.array([x])
        if constraints is not None and is_gradient_projected:
            grad = project_gradient(grad, constraints, x)

        next_x, sot, stc, stb, lr = get_next_x(learning_rate, fn, fn_grad, x, grad, constraints)

        line_search_params_one_step = [x, next_x, grad, lr, step_count]

        # if not np.allclose(next_x, np.clip(x - lr * grad, constraints[0], constraints[1]), rtol=0,
        #                                atol=1e-12):
        #     error = True
        #     print("!!!!!!??????????!???????????!")
        #     print(f" next x calculation wrong")
        #     print(f"next_x {next_x}")
        #     print(f"next_x {np.clip(x - lr * grad, constraints[0], constraints[1])}")


        for i in range(len(stc)):
            stage_two_currents.append(stc[i])
        for i in range(len(stb)):
            stage_two_brackets.append(stb[i])
        for i in range(len(sot)):
            # print(f" sot {sot[i]}")
            stage_one_tries.append(sot[i])
        # f_calls_count += temp_f_count
        # g_calls_count += temp_g_count

        if verbose:
            print(f"step {step_count} \nx: {x}\n y: {y} \ngrad: {grad} \nnext_x: {next_x}")

        # if test_wolfe:
        #     if isinstance(learning_rate, QuasiWolfeTest):
        #         phi, dphi_minus, dphi_plus = create_phi(fn, fn_grad, x, grad, constraints)
        #
        #         if success and not check_step_is_quasi_wolfe(1e-4, 0.9, x, lr, phi(0), dphi_plus(0), phi(lr),
        #                                                      dphi_minus(lr),
        #                                                      dphi_plus(lr), grad, constraints):
        #             error = True  # print(f"step {step_count} \nx: {x}\n y: {y} \ngrad: {grad} \nnext_x: {next_x}")
        #             print(f"Quasi Wolfe condition not satisfied")
        #             print("NOt Wolfe - wtf!!!!??????!!!!!!!???????")

        if step_count >= max_steps or \
                (termination_criteria == "grad" and np.linalg.norm(grad, ord=norm) < epsilon) or \
                (termination_criteria == "x star" and np.linalg.norm(next_x - x_star, ord=norm) < epsilon) or \
                (termination_criteria == "y" and abs(fn(next_x) - y) < epsilon) or \
                (termination_criteria == "x" and np.linalg.norm(next_x - x, ord=norm) < epsilon):

            if not np.allclose(x, start, rtol=0, atol=1e-12):
                if record_trace:
                    steps.append(x)
                    line_search_params.append(line_search_params_one_step)

            if record_trace:
                # print(f" steps {steps}")
                return np.array(steps),\
                       step_count, \
                       f_calls_count, \
                       g_calls_count, \
                       x, \
                       np.array(stage_one_tries),\
                       np.array(stage_two_currents),\
                       np.array(stage_two_brackets), \
                       line_search_params
            else:
                return step_count, f_calls_count, g_calls_count, x

        else:
            step_count += 1
            if record_trace:
                steps.append(x)
                line_search_params.append(line_search_params_one_step)
            x = next_x


def get_next_x(learning_rate,
               fn: Callable,
               fn_prime: Callable,
               x: np.ndarray,
               gradient_at_x: np.ndarray,
               constraints):
    if isinstance(learning_rate, Constant):
        lr = learning_rate.search()
        next_x = x - lr * gradient_at_x
        if constraints is not None:
            next_x = np.clip(next_x, constraints[0], constraints[1])
        return next_x, 1, 1, 0, 0

    if isinstance(learning_rate, QuasiWolfeTest2):
        lr, _, stage_one_tries, stage_two_currents, stage_two_brackets, \
        _, _, _ = learning_rate.search(fn, fn_prime, x, gradient_at_x)

        next_x = x - lr * gradient_at_x
        print(f" next_x {next_x} insed GD")
        if constraints is not None:
            next_x = np.clip(next_x, constraints[0], constraints[1])
            print(f" next_x clipped {next_x} insed GD")
        return  next_x, stage_one_tries, stage_two_currents, stage_two_brackets, lr

    if isinstance(learning_rate, StrongWolfe):
        lr, _, stage_one_tries, stage_two_currents, stage_two_brackets = learning_rate.search(fn, fn_prime, x,
                                                                                              gradient_at_x)
        next_x = x - lr * gradient_at_x
        if constraints is not None:
            next_x = np.clip(next_x, constraints[0], constraints[1])
        return next_x, stage_one_tries, stage_two_currents, stage_two_brackets, 0


    else:
        print(f" learning rate {learning_rate} not implemented")

    # elif learning_rate == "Armijo":
    #     alpha = Armijo_alpha
    #     delta = 10e-3
    #     while True:
    #         next_x = x - alpha * grad
    #         if constraints is not None:
    #             next_x = np.clip(next_x, constraints[:, 0], constraints[:, 1])
    #         other_side = y - alpha * delta * np.linalg.norm(grad) ** 2
    #
    #         next_y = fn(next_x)
    #
    #         if next_y <= other_side:
    #             return next_x
    #         else:
    #
    #             alpha /= 2


def get_next_x_nesterov(x, LR, fn_grad, constraints, step_count):
    # if step_count == 0:
    #     projected_x = x
    # else:
    #     projected_x = tries[-1]
    #
    # next_x = projected_x - LR * fn_grad(projected_x)
    #
    # if constraints is not None:
    #     next_x = np.clip(next_x, constraints[:, 0], constraints[:, 1])
    #
    # beta_k = (step_count - 1) / (step_count + 2) if step_count > 0 else 0
    #
    # next_projected_x = next_x + beta_k * (next_x - x)
    #
    # if constraints is not None:
    #     next_projected_x = np.clip(next_projected_x, constraints[:, 0], constraints[:, 1])
    #
    # tries.append(next_projected_x)

    # return next_x
    pass


def project_gradient(gradient, constraints, x):
    gradient = np.where(np.logical_and(gradient < 0, x == constraints[1]), 0, gradient)
    gradient = np.where(np.logical_and(gradient > 0, x == constraints[0]), 0, gradient)

    return gradient


def get_max_step_size(x, gradient, constraints):
    alphas_1 = (np.array([1]) - x) / gradient
    alphas_2 = (np.array([-1]) - x) / gradient

    filtered_alphas_1 = np.where(alphas_1 > 0, alphas_1, 0)
    filtered_alphas_2 = np.where(alphas_2 > 0, alphas_2, 0)

    return min(np.min(filtered_alphas_1), np.min(filtered_alphas_2))
