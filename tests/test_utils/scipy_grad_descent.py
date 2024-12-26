import numpy as np
from scipy.optimize import optimize

from grad_descent import project_gradient


# grad descent with wolfe line serach copied from:
# https://scipy-lectures.org/advanced/mathematical_optimization/auto_examples/plot_gradient_descent.html

def gradient_descent(x0, f, f_prime, wolfe=True, step_size=None, max_iter=100):

    if not wolfe:
        assert step_size is not None, "step_size must be provided if wolfe is False"

    x_i, y_i = x0
    all_x_i = list()
    all_y_i = list()
    all_f_i = list()

    for i in range(1, max_iter):
        all_x_i.append(x_i)
        all_y_i.append(y_i)
        all_f_i.append(f([x_i, y_i]))
        dx_i, dy_i = f_prime(np.asarray([x_i, y_i]))
        if wolfe:
            # Compute a step size using a line_search to satisfy the Wolf
            # conditions
            step = optimize.line_search(f, f_prime,
                                np.r_[x_i, y_i], -np.r_[dx_i, dy_i],
                                np.r_[dx_i, dy_i], c2=.05)
            step = step[0]
            if step is None:
                step = 0
        else:
            step = step_size
        x_i += - step*dx_i
        y_i += - step*dy_i
        if np.abs(all_f_i[-1]) < 1e-16:
            break

    return all_x_i, all_y_i, all_f_i


# GD from ChatGPT
# for projection the constraints (-1,1) are assumed and the simple portion logic from grad_descent is used
def gradient_descent_chatGPT(func,
                             grad_func,
                             initial_point,
                             learning_rate=0.01,
                             max_iterations=1000,
                             tolerance=1e-6,
                             projected = False):
    """
    Perform gradient descent on a given function with an arbitrary number of dimensions.

    Parameters:
    - func: Callable. The objective function to minimize.
    - grad_func: Callable. The gradient of the objective function.
    - initial_point: ndarray. Initial point for the descent (array of shape (n,)).
    - learning_rate: float. Step size multiplier for updates.
    - max_iterations: int. Maximum number of iterations.
    - tolerance: float. Convergence criterion based on gradient norm.

    Returns:
    - current_point: ndarray. The optimized point found by gradient descent.
    - history: list of tuples. Each tuple contains (iteration, point, function value).
    """
    current_point = np.array(initial_point, dtype=float)
    history = []

    for iteration in range(max_iterations):
        # Evaluate the gradient and function value
        gradient = np.array(grad_func(current_point))
        func_value = func(current_point)
        history.append((iteration, current_point.copy(), func_value))

        # Check convergence
        grad_norm = np.linalg.norm(gradient)
        if grad_norm < tolerance:
            print(f"Converged at iteration {iteration} with gradient norm {grad_norm}")
            break

        # Update the current point
        if projected:
            gradient = project_gradient(gradient, (-1,1), current_point)

        current_point -= learning_rate * gradient

        if projected:
            current_point = np.clip(current_point, -1, 1)

    else:
        print("Maximum iterations reached without convergence.")

    return current_point, history

