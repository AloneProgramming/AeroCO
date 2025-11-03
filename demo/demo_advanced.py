import numpy as np
import matplotlib.pyplot as plt
from methods.discrete_vortex import DiscreteVortexMethod
from methods.thin_airfoil import ThinAirfoilTheory
from core.camber_functions import naca_4_digit_camber


def demo_grid_convergence():
    print("Grid Convergence Study")

    panel_counts = [10, 20, 40, 80, 160]
    cl_values_symmetric = []
    cl_values_cambered = []

    for panels in panel_counts:
        dvm = DiscreteVortexMethod(alpha_degrees=5.0, n_panels=panels, debug=False)
        cl, _ = dvm.calculate_aerodynamics()
        cl_values_symmetric.append(cl)

    camber_func = naca_4_digit_camber(code="4412")
    for panels in panel_counts:
        dvm = DiscreteVortexMethod(alpha_degrees=5.0, n_panels=panels,
                                   camber_func=camber_func, debug=False)
        cl, _ = dvm.calculate_aerodynamics()
        cl_values_cambered.append(cl)

    theoretical_cl_symmetric = 2 * np.pi * np.radians(5.0)
    tat = ThinAirfoilTheory(camber_func=camber_func)
    theoretical_cl_cambered = tat.calculate_lift(5.0)

    plt.figure(figsize=(10, 6))
    plt.plot(panel_counts, cl_values_symmetric, 'bo-', label='Symmetric (DVM)')
    plt.axhline(y=theoretical_cl_symmetric, color='b', linestyle='--',
                label=f'Symmetric (Theoretical)')
    plt.plot(panel_counts, cl_values_cambered, 'ro-', label='Cambered (DVM)')
    plt.axhline(y=theoretical_cl_cambered, color='r', linestyle='--',
                label=f'Cambered (Theoretical)')
    plt.xlabel('Number of Panels')
    plt.ylabel('Lift Coefficient Cl')
    plt.title('Grid Convergence Study')
    plt.grid(True)
    plt.legend()
    plt.show()


def demo_naca_series():
    print("\nNACA Series Demo")

    naca_codes = ["0012", "2412", "4412", "6412"]
    alpha = 5.0
    results = {}

    for code in naca_codes:
        camber_func = naca_4_digit_camber(code=code)
        dvm = DiscreteVortexMethod(alpha_degrees=alpha, n_panels=100,
                                   camber_func=camber_func, debug=False)
        cl, _ = dvm.calculate_aerodynamics()
        results[code] = cl

        tat = ThinAirfoilTheory(camber_func=camber_func)
        cl_theory = tat.calculate_lift(alpha)

        print(f"NACA {code}: Cl = {cl:.4f} (DVM), Cl = {cl_theory:.4f} (Theory)")

    codes = list(results.keys())
    cl_values = list(results.values())

    plt.figure(figsize=(10, 6))
    plt.bar(codes, cl_values, color='skyblue')
    plt.xlabel('NACA Profile')
    plt.ylabel('Lift Coefficient Cl')
    plt.title(f'Lift Coefficient for NACA Profiles at {alpha}° AOA')
    plt.grid(True, alpha=0.3)
    plt.show()


def demo_alpha_sweep():
    print("\nAngle of Attack Sweep")

    alpha_range = np.arange(-5, 15, 2)
    cl_symmetric = []
    cl_cambered = []

    for alpha in alpha_range:
        dvm = DiscreteVortexMethod(alpha_degrees=alpha, n_panels=100, debug=False)
        cl, _ = dvm.calculate_aerodynamics()
        cl_symmetric.append(cl)

    camber_func = naca_4_digit_camber(code="2412")
    for alpha in alpha_range:
        dvm = DiscreteVortexMethod(alpha_degrees=alpha, n_panels=100,
                                   camber_func=camber_func, debug=False)
        cl, _ = dvm.calculate_aerodynamics()
        cl_cambered.append(cl)

    plt.figure(figsize=(10, 6))
    plt.plot(alpha_range, cl_symmetric, 'bo-', label='Symmetric (NACA 0012)')
    plt.plot(alpha_range, cl_cambered, 'ro-', label='Cambered (NACA 2412)')
    plt.xlabel('Angle of Attack (degrees)')
    plt.ylabel('Lift Coefficient Cl')
    plt.title('Lift Curve')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    demo_grid_convergence()
    #demo_naca_series()
    #demo_alpha_sweep()