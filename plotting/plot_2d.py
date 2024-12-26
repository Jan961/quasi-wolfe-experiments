import numpy as np
import matplotlib.pyplot as plt


def plot_grad_descent_2d(steps, f, margin=0.5, bounds=(-1,1), samples_per_side=500, levels=50,
                         border_color='grey', f_is_from_polynomial_generator=True):


    x = np.linspace(bounds[0] - margin, bounds[1] + margin, samples_per_side)
    y = np.linspace(bounds[0] - margin, bounds[1] + margin, samples_per_side)

    # create a matplotlip contour plot of the function around its minimum
    xv, yv = np.meshgrid(x, y)
    if not f_is_from_polynomial_generator:
        stacked = np.stack((xv, yv), axis=2)

        ys = np.apply_along_axis(f, 2, stacked)


    fig, axes = plt.subplots()

    plt.hlines(np.array([bounds[0], bounds[1]]), xmin=(bounds[0]), xmax=bounds[1], colors=border_color,
               linestyles='dashed')
    plt.vlines(np.array([bounds[0], bounds[1]]), ymin=(bounds[0]), ymax=bounds[1], colors=border_color,
               linestyles='dashed')
    if not f_is_from_polynomial_generator:
        contours = axes.contourf(x, y, ys, levels)
    else:
        contours = axes.contourf(x, y, f([xv,yv]), levels)
    fig.colorbar(contours, ax=axes, orientation='vertical',
                 location='right', shrink=1)
    axes.plot(steps[:, 0], steps[:, 1], 'ro')
    axes.plot(steps[:, 0], steps[:, 1], 'r--')
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    # plt.show()
    return fig, axes