import numpy as np
import matplotlib.pyplot as plt

class ThinAirfoil:
    def __init__(self, alpha_degrees=5.0):
        self.alpha = np.radians(alpha_degrees)
        self.c = 1.0

    def calculate_coefficients_cambered(self, dz_dx_func):
        theta = np.linspace(0, np.pi, 1000)
        x = 0.5 * (1 - np.cos(theta))

        dz_dx = dz_dx_func(x)

        A0 = self.alpha - (1 / np.pi) * np.trapz(dz_dx, theta)
        A1 = (2 / np.pi) * np.trapz(dz_dx * np.cos(theta), theta)
        A2 = (2 / np.pi) * np.trapz(dz_dx * np.cos(2 * theta), theta)

        return A0, A1, A2

    def calculate_coefficients(self, dz_dx_func=None):
        if dz_dx_func is None:
            A0 = self.alpha
            A1 = 0.0
            A2 = 0.0
        else:
            A0, A1, A2 = self.calculate_coefficients_cambered(dz_dx_func)

        Cl = 2 * np.pi * (A0 + A1 / 2)
        Cm_LE = -0.5 * np.pi * (A0 + A1 - A2 / 2)
        Cm_AC = -0.25 * np.pi * (A1 - A2)

        return Cl, Cm_LE, Cm_AC, A0, A1, A2

    def plot_results(self, Cl, Cm_LE, Cm_AC, A0, A1, A2):
        print("Symmetric airfoil results.")
        print(f"AOA: {np.degrees(self.alpha):.1f}°")
        print(f"Cl: {Cl:.4f}")
        print(f"Cm_LE: {Cm_LE:.4f}")
        print(f"Cm_AC: {Cm_AC:.4f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        x = np.linspace(0, 1, 100)
        y = np.zeros_like(x)

        ax1.plot(x, y, 'b-', linewidth=2, label='Symmetric airfoil')
        ax1.fill_between(x, -0.02, 0.02, alpha=0.2, color='blue')
        ax1.set_xlabel('x/c')
        ax1.set_ylabel('y/c')
        ax1.set_title('Symmetric airfoil')
        ax1.grid(True)
        ax1.axis('equal')
        ax1.legend()

        alpha_test = np.linspace(-10, 10, 50)
        Cl_test = 2 * np.pi * np.radians(alpha_test)

        ax2.plot(alpha_test, Cl_test, 'r-', linewidth=2, label='Thin Airfoil Theory')
        ax2.plot(np.degrees(self.alpha), Cl, 'bo', markersize=8, label='Point')
        ax2.set_xlabel('α, deg')
        ax2.set_ylabel('Cl')
        ax2.set_title('Cl(α)')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def plot_airfoil(self, dz_dx_func=None, n_points=100):
        x = np.linspace(0, 1, n_points)

        if dz_dx_func is None:
            z = np.zeros_like(x)
        else:
            z = np.zeros_like(x)
            for i in range(1, len(x)):
                z[i] = z[i - 1] + dz_dx_func(x[i - 1]) * (x[i] - x[i - 1])
            z = z - z[0]
            z = z - x * (z[-1] - 0)

        plt.figure(figsize=(10, 4))
        plt.plot(x, z, 'b-', linewidth=2, label='Camber line')
        plt.fill_between(x, z - 0.01, z + 0.01, alpha=0.2, color='blue', label='Airfoil thickness (schematical)')
        plt.xlabel('x/c')
        plt.ylabel('z/c')
        plt.title('Airfoil Profile')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.show()

        return x, z