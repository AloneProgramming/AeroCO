import numpy as np


def numerical_gradient(f, x, h=1e-8):
    x = np.array(x, dtype=float)
    grad = np.zeros_like(x)

    for i in range(len(x)):
        x_forward = x.copy()
        x_backward = x.copy()
        x_forward[i] += h
        x_backward[i] -= h

        grad[i] = (f(x_forward) - f(x_backward)) / (2 * h)

    return grad


def quadratic_function(x):
    return x[0] ** 2 + x[1] ** 2  # f(x, y) = x² + y²


def quadratic_function_analytical(point):
    return [2 * point[0], 2 * point[1]]


def linear_combination(x):
    return x[0] * x[1] + x[1] * x[2] + x[2] * x[0]  # f(x, y, z) = x*y + y*z + z*x


def linear_combination_analytical(point):
    return [point[1] + point[2], point[0] + point[2], point[1] + point[0]]


point = [1.0, 2.0, 3.0]
analytical_grad = linear_combination_analytical(point)
numerical_grad = numerical_gradient(linear_combination, point)

print(f"Point: {point}")
print(f"Analytical gradient: {analytical_grad}")
print(f"Numerical gradient: {numerical_grad}")
