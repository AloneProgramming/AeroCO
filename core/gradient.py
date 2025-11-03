import numpy as np
import matplotlib.pyplot as plt


def calculate_gradient(f, x, h=1e-8):
    x = np.array(x, dtype=float)
    grad = np.zeros_like(x)

    for i in range(len(x)):
        x_forward = x.copy()
        x_backward = x.copy()
        x_forward[i] += h
        x_backward[i] -= h
        grad[i] = (f(x_forward) - f(x_backward)) / (2 * h)

    return grad


def quadratic_func(x):
    """f(x,y) = x² + y²; grad: (2x, 2y)"""
    value = x[0]**2 + x[1]**2
    gradient = [2*x[0], 2*x[1]]
    return value, gradient

def trig_func(x):
    """f(x,y) = sin(x) + cos(y); grad: (cos(x), -sin(y))"""
    value = np.sin(x[0]) + np.cos(x[1])
    gradient = [np.cos(x[0]), -np.sin(x[1])]
    return value, gradient

def polynomial_func(x):
    """f(x,y) = x²y + y³; grad: (2xy, x² + 3y²)"""
    value = x[0]**2 * x[1] + x[1]**3
    gradient = [2*x[0]*x[1], x[0]**2 + 3*x[1]**2]
    return value, gradient

def quadratic(x):
    return quadratic_func(x)[0]

def trig(x):
    return trig_func(x)[0]

def polynomial(x):
    return polynomial_func(x)[0]


def plot_gradient_simple(func, func_name, point, x_range=(-3, 3), y_range=(-3, 3)):
    X, Y = np.meshgrid(np.linspace(x_range[0], x_range[1], 20),
                       np.linspace(y_range[0], y_range[1], 20))

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]])

    grad = calculate_gradient(func, point)

    plt.figure(figsize=(10, 8))

    contour = plt.contour(X, Y, Z, levels=15)
    plt.clabel(contour, inline=True, fontsize=8)

    plt.plot(point[0], point[1], 'ro', markersize=10, label=f'Point {point}')
    plt.arrow(point[0], point[1], grad[0], grad[1],
              head_width=0.1, head_length=0.1, fc='red', ec='red',
              linewidth=2, label='Grad')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Grad {func_name}')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":

    #point1 = [1.0, 2.0]
    #plot_gradient_simple(quadratic, "x² + y²", point1)

    point2 = [np.pi / 4, np.pi / 3]
    plot_gradient_simple(trig, "sin(x) + cos(y)", point2, (-2 * np.pi, 2 * np.pi), (-2 * np.pi, 2 * np.pi))