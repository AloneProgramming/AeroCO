import numpy as np
import matplotlib.pyplot as plt

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

def trigonometric_combination(x):
    return np.sin(x[0]) + np.cos(x[1])  #  f(x, y) = sin(x) + cos(y)

def trigonometric_combination_analytical(point):
    return [float(np.cos(point[0])), float(- np.sin(point[1]))]

def two_var_polynomial(x):
    return x[1] * x[0] ** 2 + x[1] ** 3  # f(x, y) = x² * y + y³

def two_var_polynomial_analytical(point):
    return [2 * point[0] * point[1], point[0] ** 2 + 3 * point[1] ** 2]

def plot_gradient_demo(f, x_range, y_range, point):
    X, Y = np.meshgrid(np.linspace(x_range[0], x_range[1], 20),
                       np.linspace(y_range[0], y_range[1], 20))

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f([X[i, j], Y[i, j]])

    grad = numerical_gradient(f, point)

    plt.figure(figsize=(10, 8))

    contour = plt.contour(X, Y, Z, levels=20)
    plt.clabel(contour, inline=True, fontsize=8)

    plt.plot(point[0], point[1], 'ro', markersize=10, label=f'Point {point}')

    plt.arrow(point[0], point[1], grad[0], grad[1],
              head_width=0.1, head_length=0.1, fc='red', ec='red',
              linewidth=2, label='Grad')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Grad visualisation')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

point = [5, 3]
analytical_grad = two_var_polynomial_analytical(point)
numerical_grad = numerical_gradient(two_var_polynomial, point)
plot_gradient_demo(two_var_polynomial, [-50, 50], [-50, 50], point)

print(f"Point: {point}")
print(f"Analytical gradient: {analytical_grad}")
print(f"Numerical gradient: {numerical_grad}")