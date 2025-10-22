import numpy as np
import matplotlib.pyplot as plt

class ThinAirfoil:
    def __init__(self, alpha_degrees=5.0):
        self.alpha = np.radians(alpha_degrees)
        self.c = 1.0

    def calculate_coefficients_symmetric(self):
        A0 = self.alpha
        A1 = 0.0
        A2 = 0.0

        Cl = 2 * np.pi * (A0 + A1/2)
        Cm_LE = -0.5 * np.pi * (A0 + A1 - A2/2)
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