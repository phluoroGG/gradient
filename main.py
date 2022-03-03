import numpy as np
import matplotlib.pyplot as plt


def function(x, y):
    return np.cos(x) ** 2 * np.log(y ** 2 + 1)


def analytical_derivative_x(x, y):
    return - 2 * np.cos(x) * np.sin(x) * np.log(y ** 2 + 1)


def analytical_derivative_y(x, y):
    return np.cos(x) ** 2 * 2 * y / (y ** 2 + 1)


def numerical_derivative_x(x, y, step):
    return (function(x + step, y) - function(x - step, y)) / (2 * step)


def numerical_derivative_y(x, y, step):
    return (function(x, y + step) - function(x, y - step)) / (2 * step)


def draw_function():
    x = np.arange(-5, 5, 0.1)
    y = np.arange(-5, 5, 0.1)
    x_grid, y_grid = np.meshgrid(x, y)
    z = function(x_grid, y_grid)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set(xlabel='x', ylabel='y', zlabel='z',
           title='np.log(y) ** 2 / (np.tan(x) ** 2 + 2) - np.sin(np.log(x)) / (y ** 2 + 1)')
    ax.plot_surface(x_grid, y_grid, z, cmap='Wistia')

    plt.show()


if __name__ == '__main__':
    x_ = 1.5
    y_ = 1.5
    step_ = 1e-3

    draw_function()

    print('function value at x = {0} and y = {1} is {2}'.format(x_, y_, function(x_, y_)))
    print('analytical gradient value at x = {0} and y = {1} is ({2}, {3})'
          .format(x_, y_, analytical_derivative_x(x_, y_), analytical_derivative_y(x_, y_)))
    print('numerical gradient value at x = {0} and y = {1} with step = {2} is ({3}, {4})'
          .format(x_, y_, step_, numerical_derivative_x(x_, y_, step_), numerical_derivative_y(x_, y_, step_)))
