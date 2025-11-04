import numpy as np
import matplotlib.pyplot as plt
from core.camber_functions import parabolic_camber, naca_4_digit_camber
from methods.thin_airfoil import ThinAirfoilTheory
from methods.discrete_vortex import DiscreteVortexMethod
from demo.demo_advanced import *

def grid_convergence_study(airfoil_type="naca", code="0012", alpha=5.0):
    print("Running grid convergence study...")

    if airfoil_type == "naca":
        camber_func = naca_4_digit_camber(code=code)
    else:
        camber_func = None

    panel_counts = [10, 20, 40, 80, 160]
    cl_values = []

    for panels in panel_counts:
        dvm = DiscreteVortexMethod(
            alpha_degrees=alpha,
            n_panels=panels,
            camber_func=camber_func,
            debug=False
        )
        cl, _ = dvm.calculate_aerodynamics()
        cl_values.append(cl)

    theoretical_cl = 2 * np.pi * (np.radians(alpha) + np.atan(camber_func.m / (1 - camber_func.p)))

    plt.figure(figsize=(10, 6))
    plt.plot(panel_counts, cl_values, 'bo-', label='Numerical')
    plt.axhline(y=theoretical_cl, color='r', linestyle='--',
                label=f'Theoretical: {theoretical_cl:.4f}')
    plt.xlabel('Number of Panels')
    plt.ylabel('Lift Coefficient Cl')
    plt.title('Grid Convergence Study')
    plt.grid(True)
    plt.legend()
    plt.show()

    return panel_counts, cl_values


if __name__ == "__main__":
    # grid_convergence_study(airfoil_type="naca", code="4415", alpha=5.0)

    airfoil_test = DiscreteVortexMethod(
        alpha_degrees=10,
        U_inf=10,
        n_panels=200,
        camber_func=naca_4_digit_camber(code="8315"),
        debug=False)

    airfoil_data(airfoil=airfoil_test)
    airfoil_flow(airfoil=airfoil_test)
