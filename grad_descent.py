from typing import Callable

import numpy as np
from line_searches.Constant import Constant
from line_searches.QuasiWolfe import QuasiWolfe
from line_searches.StrongWolfe import StrongWolfe
from line_searches.MoreThuente import MoreThuente


# learning rates =
#        "Armijo" - check is satisfies Armijo condition if not halve
#        "constant" - constant learning rate
#        "progress bound" - check is satisfies progress bound if not halve

# termination criteria =
#        "grad" - terminate when gradient is small
#        "x star" - terminate when x is close to x_star
#        "y" - terminate when function value change is small
#        "x" - terminate when change in x is small


# returned steps will not include the initial point if it is set to all zeros
def grad_descent(fn,
                 fn_grad,
                 start: np.ndarray,
                 epsilon: float = 0.01,
                 learning_rate=Constant(0.01),
                 termination_criterion: str = "grad",
                 norm=2,
                 x_star: np.ndarray = None,
                 max_steps=1000,
                 constraints: np.ndarray | tuple | list = None,  # shape (n_vars, 2)
                 nesterov=False,
                 verbose=False,
                 is_gradient_projected=True,
                 record_trace=False,
                 record_gradients=False,
                 record_objective_values=False, ):
    if nesterov:
        assert isinstance(learning_rate, Constant), "Nesterov only supports constant learning rate"
        tries = []

    if constraints is not None:
        assert len(constraints) == 2

    assert isinstance(start, np.ndarray), "start must be a numpy array"

    if record_trace:
        steps = np.zeros((max_steps + 1, len(start)))
    else:
        steps = []

    if record_gradients:
        gradients = np.zeros(max_steps + 1)
    else:
        gradients = []

    if record_objective_values:
        objective_values = np.zeros(max_steps + 1)
    else:
        objective_values = []

    step_count = 0
    f_calls_count = 0
    g_calls_count = 0
    x = start

    while True:

        # print(f" this is x outside {x[0]}, {type(x[0])}")

        grad = fn_grad(x)
        y = fn(x)

        f_calls_count += 1
        g_calls_count += 1


        # print(f" grad: {grad}")
        # print(f"x: {x}")

        # if len(x) == 1:
        #     grad = np.array([grad])
        # x = np.array([x])
        if constraints is not None and is_gradient_projected:
            grad = project_gradient(grad, constraints, x)

        if nesterov:
            next_x, temp_f_count, temp_g_count = get_next_x_nesterov(x, learning_rate.alpha,
                                                                     fn_grad, constraints, step_count, tries)
        else:
            next_x, temp_f_count, temp_g_count = get_next_x(learning_rate, fn, fn_grad, x, grad, constraints)

        f_calls_count += temp_f_count
        g_calls_count += temp_g_count

        if verbose:
            print(f"step {step_count} \nx: {x}\n y: {y} \ngrad: {grad} \nnext_x: {next_x}")

        if (termination_criterion == "grad" and np.linalg.norm(grad, ord=norm) <= epsilon) or \
                (termination_criterion == "x star" and np.linalg.norm(next_x - x_star, ord=norm) <= epsilon) or \
                (termination_criterion == "y" and abs(fn(next_x) - y) <= epsilon) or \
                (termination_criterion == "x" and np.linalg.norm(next_x - x, ord=norm) <= epsilon):

            if not np.allclose(x, start, rtol=0, atol=1e-12):
                if record_trace:
                    steps[step_count] = x
                if record_gradients:
                    gradients[step_count] = np.linalg.norm(grad)
                if record_objective_values:
                    objective_values[step_count] = y

            # print(f" step count 2 {step_count}")
            if record_trace:
                steps = steps[:step_count + 1]

            success = True

            return steps, \
                   step_count, \
                   f_calls_count, \
                   g_calls_count, \
                   x, \
                   gradients[:step_count + 1], \
                   objective_values[:step_count + 1], \
                   success

        elif step_count >= max_steps:
            success = False

            # print(f" steps {steps}")
            return steps, \
                   step_count, \
                   f_calls_count, \
                   g_calls_count, \
                   x, \
                   gradients[:step_count], \
                   objective_values[:step_count], \
                   success


        else:
            if record_trace:
                steps[step_count] = x
            if record_gradients:
                gradients[step_count] = np.linalg.norm(grad)
            if record_objective_values:
                objective_values[step_count] = y
            x = next_x
            step_count += 1


def get_next_x(learning_rate, fn: Callable, fn_prime: Callable, x: np.ndarray,
               gradient_at_x: np.ndarray, constraints):
    if isinstance(learning_rate, Constant):
        lr = learning_rate.search()
        next_x = x - lr * gradient_at_x
        if constraints is not None:
            next_x = np.clip(next_x, constraints[0], constraints[1])
        return next_x, 1, 1

    if isinstance(learning_rate, QuasiWolfe):
        lr, _, f_calls, g_calls = learning_rate.search(fn, fn_prime, x,
                                     gradient_at_x)
        next_x = x - lr * gradient_at_x
        if constraints is not None:
            next_x = np.clip(next_x, constraints[0], constraints[1])
        return next_x, f_calls, g_calls

    if isinstance(learning_rate, StrongWolfe):
        lr, _, _, _, _ = learning_rate.search(fn, fn_prime, x,
                                              gradient_at_x)
        next_x = x - lr * gradient_at_x
        if constraints is not None:
            next_x = np.clip(next_x, constraints[0], constraints[1])
        return next_x, 0, 0

    if isinstance(learning_rate, MoreThuente):
        lr, _ = learning_rate.search(fn, fn_prime, x, gradient_at_x)
        next_x = x - lr * gradient_at_x
        if constraints is not None:
            next_x = np.clip(next_x, constraints[0], constraints[1])
        return next_x, 0, 0

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


def get_next_x_nesterov(x, LR, fn_grad, constraints, step_count, tries):
    if step_count == 0:
        projected_x = x
    else:
        projected_x = tries[-1]

    next_x = projected_x - LR * fn_grad(projected_x)

    if constraints is not None:
        next_x = np.clip(next_x, constraints[0], constraints[1])

    beta_k = (step_count - 1) / (step_count + 2) if step_count > 0 else 0

    next_projected_x = next_x + beta_k * (next_x - x)

    if constraints is not None:
        next_projected_x = np.clip(next_projected_x, constraints[0], constraints[1])

    tries.append(next_projected_x)

    return next_x, 0, 0
    pass


def project_gradient(gradient, constraints, x):
    gradient_positive = np.where(np.logical_and(gradient < 0, np.isclose(x, constraints[1], rtol=0, atol=1e-12)), 0,
                                 gradient)
    gradient_all_zeroed = np.where(np.logical_and(gradient > 0, np.isclose(x, constraints[0], rtol=0, atol=1e-12)), 0,
                                   gradient_positive)

    return gradient_all_zeroed


def get_max_step_size(x, gradient, constraints):
    alphas_1 = (np.array([1]) - x) / gradient
    alphas_2 = (np.array([-1]) - x) / gradient

    filtered_alphas_1 = np.where(alphas_1 > 0, alphas_1, 0)
    filtered_alphas_2 = np.where(alphas_2 > 0, alphas_2, 0)

    return min(np.min(filtered_alphas_1), np.min(filtered_alphas_2))
