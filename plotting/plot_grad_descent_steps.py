from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


def plot_steps_2d(steps: np.ndarray, axes: plt.Axes):
    axes.plot(steps[:, 0], steps[:, 1], 'ro')
    axes.plot(steps[:, 0], steps[:, 1], 'r--')
    axes.set_xlabel('x')
    axes.set_ylabel('y')


def plot_grad_descent_2d(steps, fn, samples_per_side=200, levels=50,
                         bounds_x: tuple =(-1, 1), bounds_y: tuple = (-1, 1), margin=0.1):
    x = np.linspace(bounds_x[0] - margin, bounds_x[1] + margin, samples_per_side)
    y = np.linspace(bounds_y[0] - margin, bounds_y[1] + margin, samples_per_side)

    # create a matplotlip contour plot of the function around its minimum
    xv, yv = np.meshgrid(x, y)
    stacked = np.dstack((xv,yv))

    fig, axes = plt.subplots()

    plt.hlines(np.array([bounds_y[0], bounds_y[1]]), xmin=(bounds_x[0]), xmax=bounds_x[1], colors='grey', linestyles='dashed')
    plt.vlines(np.array([bounds_x[0], bounds_x[1]]), ymin=(bounds_y[0]), ymax=bounds_y[1], colors='grey', linestyles='dashed')
    plot_steps_2d(steps, axes)

    # lbd = fn.lambda_for_plotting_2d()
    contours = axes.contourf(x, y, np.apply_along_axis(fn,2,stacked), levels)
    fig.colorbar(contours, ax=axes, orientation='vertical',
                 location='right', shrink=1)
    plt.show()


def plot_steps_1d(steps: np.ndarray, axes: plt.Axes, fn: Callable):
    axes.plot(steps, [fn(np.array([step])) for step in steps], 'ro')
    axes.plot(steps, [fn(np.array([step])) for step in steps], 'r--')
    axes.legend(['Steps of Gradient Descent'])


# plot the steps of gradient descent on a 1D function
def plot_grad_descent_1d(steps, fn, title: str = 'Gradient Descent Steps'):
    x = np.linspace(-1, 1, 1000)

    # create a matplotlip contour plot of the function around its minimum
    fig, axes = plt.subplots()
    plot_steps_1d(steps, axes, fn)
    y = [fn(np.array([x_i])) for x_i in x]
    axes.plot(x, y)
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_title(title)
    plt.show()
