import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple


class ScalarFunction(ABC):
    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def analytical_gradient(self, x: np.ndarray) -> np.ndarray:
        pass

    def dimension(self) -> int:
        return self._dimension

    def __call__(self, x: np.ndarray) -> float:
        return self.evaluate(x)


class QuadraticFunction(ScalarFunction):
    """f(x, y) = x² + y²"""
    def __init__(self):
        self._dimension = 2

    def evaluate(self, x: np.ndarray) -> float:
        return x[0] ** 2 + x[1] ** 2

    def analytical_gradient(self, x: np.ndarray) -> np.ndarray:
        return np.array([2 * x[0], 2 * x[1]])


class LinearCombinationFunction(ScalarFunction):
    """f(x, y, z) = x*y + y*z + z*x"""
    def __init__(self):
        self._dimension = 3

    def evaluate(self, x: np.ndarray) -> float:
        return x[0] * x[1] + x[1] * x[2] + x[2] * x[0]

    def analytical_gradient(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[1] + x[2], x[0] + x[2], x[0] + x[1]])


class TrigonometricFunction(ScalarFunction):
    """f(x, y) = sin(x) + cos(y)"""
    def __init__(self):
        self._dimension = 2

    def evaluate(self, x: np.ndarray) -> float:
        return np.sin(x[0]) + np.cos(x[1])

    def analytical_gradient(self, x: np.ndarray) -> np.ndarray:
        return np.array([np.cos(x[0]), -np.sin(x[1])])


class PolynomialFunction(ScalarFunction):
    """f(x, y) = x² * y + y³"""
    def __init__(self):
        self._dimension = 2

    def evaluate(self, x: np.ndarray) -> float:
        return x[0] ** 2 * x[1] + x[1] ** 3

    def analytical_gradient(self, x: np.ndarray) -> np.ndarray:
        return np.array([2 * x[0] * x[1], x[0] ** 2 + 3 * x[1] ** 2])


class GradientCalculator:
    @staticmethod
    def central_difference(f: Callable, x: np.ndarray, h: float = 1e-8) -> np.ndarray:
        x = np.array(x, dtype=float)
        grad = np.zeros_like(x)

        for i in range(len(x)):
            x_forward = x.copy()
            x_backward = x.copy()
            x_forward[i] += h
            x_backward[i] -= h

            grad[i] = (f(x_forward) - f(x_backward)) / (2 * h)

        return grad

    @staticmethod
    def forward_difference(f: Callable, x: np.ndarray, h: float = 1e-8) -> np.ndarray:
        x = np.array(x, dtype=float)
        grad = np.zeros_like(x)
        f_base = f(x)

        for i in range(len(x)):
            x_forward = x.copy()
            x_forward[i] += h
            grad[i] = (f(x_forward) - f_base) / h

        return grad

    @staticmethod
    def backward_difference(f: Callable, x: np.ndarray, h: float = 1e-8) -> np.ndarray:
        x = np.array(x, dtype=float)
        grad = np.zeros_like(x)
        f_base = f(x)

        for i in range(len(x)):
            x_backward = x.copy()
            x_backward[i] -= h
            grad[i] = (f_base - f(x_backward)) / h

        return grad


class GradientVisualizer:
    @staticmethod
    def plot_function_2d(func: ScalarFunction, x_range: Tuple[float, float],
                         y_range: Tuple[float, float], resolution: int = 50):
        if func.dimension() != 2:
            raise ValueError("2D functions only")

        X, Y = np.meshgrid(np.linspace(x_range[0], x_range[1], resolution),
                           np.linspace(y_range[0], y_range[1], resolution))

        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = func([X[i, j], Y[i, j]])

        plt.figure(figsize=(12, 10))
        contour = plt.contour(X, Y, Z, levels=20)
        plt.clabel(contour, inline=True, fontsize=8)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Function')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

        return X, Y, Z

    @staticmethod
    def plot_gradient_field(func: ScalarFunction, x_range: Tuple[float, float],
                            y_range: Tuple[float, float], points: np.ndarray = None,
                            resolution: int = 15):
        if func.dimension() != 2:
            raise ValueError("2D functions only")

        X, Y = np.meshgrid(np.linspace(x_range[0], x_range[1], resolution),
                           np.linspace(y_range[0], y_range[1], resolution))

        U, V = np.zeros_like(X), np.zeros_like(Y)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                grad = GradientCalculator.central_difference(func, [X[i, j], Y[i, j]])
                U[i, j] = grad[0]
                V[i, j] = grad[1]

        plt.figure(figsize=(12, 10))

        plt.quiver(X, Y, U, V, scale=50, color='red', alpha=0.7)

        if points is not None:
            plt.plot(points[:, 0], points[:, 1], 'bo', markersize=8, label='Analysis point')

            for point in points:
                grad = GradientCalculator.central_difference(func, point)
                plt.arrow(point[0], point[1], grad[0] * 0.1, grad[1] * 0.1,
                          head_width=0.1, head_length=0.1, fc='blue', ec='blue',
                          linewidth=2)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Gradient field')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()


class GradientAnalyzer:
    @staticmethod
    def compare_methods(func: ScalarFunction, point: np.ndarray, h: float = 1e-8):
        analytical_grad = func.analytical_gradient(point)
        central_grad = GradientCalculator.central_difference(func, point, h)
        forward_grad = GradientCalculator.forward_difference(func, point, h)
        backward_grad = GradientCalculator.backward_difference(func, point, h)

        print(f"Methods' comparison in {point}:")
        print(f"Analytical:      {analytical_grad}")
        print(f"Central difference:  {central_grad} (Err: {np.linalg.norm(analytical_grad - central_grad):.2e})")
        print(f"Forward difference:       {forward_grad} (Err: {np.linalg.norm(analytical_grad - forward_grad):.2e})")
        print(f"Backward difference:     {backward_grad} (Err: {np.linalg.norm(analytical_grad - backward_grad):.2e})")
        print()

    @staticmethod
    def analyze_step_size(func: ScalarFunction, point: np.ndarray):
        analytical_grad = func.analytical_gradient(point)
        steps = [10 ** (-i) for i in range(1, 14)]

        central_errors = []
        forward_errors = []

        for h in steps:
            central_grad = GradientCalculator.central_difference(func, point, h)
            forward_grad = GradientCalculator.forward_difference(func, point, h)

            central_errors.append(np.linalg.norm(analytical_grad - central_grad))
            forward_errors.append(np.linalg.norm(analytical_grad - forward_grad))

        plt.figure(figsize=(12, 8))
        plt.loglog(steps, central_errors, 'b-', linewidth=2, label='Central difference')
        plt.loglog(steps, forward_errors, 'r--', linewidth=2, label='Forward difference')
        plt.xlabel('Step h')
        plt.ylabel('Err')
        plt.title('Err(h)')
        plt.legend()
        plt.grid(True)
        plt.show()


def main():
    quadratic = QuadraticFunction()
    trigonometric = TrigonometricFunction()
    polynomial = PolynomialFunction()

    points = np.array([[1, 2], [-1, 1], [2, -1], [-2, -2]])

    #GradientAnalyzer.compare_methods(quadratic, [1, 2])
    #GradientVisualizer.plot_function_2d(quadratic, [-3, 3], [-3, 3])
    #GradientVisualizer.plot_gradient_field(quadratic, [-3, 3], [-3, 3], points)

    #GradientAnalyzer.compare_methods(trigonometric, [np.pi / 4, np.pi / 3])
    #GradientVisualizer.plot_function_2d(trigonometric, [-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi])
    GradientVisualizer.plot_gradient_field(trigonometric, [-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi], points)

    #GradientAnalyzer.analyze_step_size(quadratic, [1, 2])


if __name__ == "__main__":
    main()