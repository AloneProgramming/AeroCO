import numpy as np
import matplotlib.pyplot as plt
import aerosandbox as asb
from core.camber_functions import parabolic_camber, naca_4_digit_camber
from methods.thin_airfoil import ThinAirfoilTheory
from methods.discrete_vortex import DiscreteVortexMethod


def calculate_airfoil_performance(airfoil_type="naca", code="0012",
                                  alpha_range=(-5, 15, 5), panels=100):
    print(f"Calculating {airfoil_type} {code} airfoil performance...")

    if airfoil_type == "naca":
        camber_func = naca_4_digit_camber(code=code)
        nf_airfoil = asb.Airfoil("naca" + code)
    elif airfoil_type == "parabolic":
        camber_func = parabolic_camber(h=0.05)
    else:
        camber_func = None

    alpha_values = np.arange(alpha_range[0], alpha_range[1] + 1, alpha_range[2])
    cl_dvm = []

    for alpha in alpha_values:
        dvm = DiscreteVortexMethod(
            alpha_degrees=alpha,
            n_panels=panels,
            camber_func=camber_func,
            debug=False
        )
        cl, _ = dvm.calculate_aerodynamics()
        cl_dvm.append(cl)

    cl_nf = []

    if airfoil_type == "naca":
        for alpha in alpha_values:
            cl = nf_airfoil.get_aero_from_neuralfoil(alpha=alpha, Re=3e5, model_size="xxxlarge")
            cl_nf.append(cl['CL'])

    if camber_func:
        tat = ThinAirfoilTheory(camber_func=camber_func)
        cl_tat = [tat.calculate_lift(alpha) for alpha in alpha_values]
    else:
        tat = ThinAirfoilTheory()
        cl_tat = [tat.calculate_lift(alpha) for alpha in alpha_values]

    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, cl_dvm, 'bo-', label='Discrete Vortex Method')
    plt.plot(alpha_values, cl_tat, 'r--', label='Thin Airfoil Theory')
    plt.plot(alpha_values, cl_nf, 'black', label='NeuralFoil')
    plt.xlabel('Angle of Attack (degrees)')
    plt.ylabel('Lift Coefficient Cl')
    plt.title(f'{airfoil_type.upper()} {code} Airfoil Performance')
    plt.grid(True)
    plt.legend()
    plt.show()

    return alpha_values, cl_dvm, cl_tat


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
    calculate_airfoil_performance(airfoil_type="naca", code="4415",
                                   alpha_range=(-5, 15, 2), panels=100)

    #  grid_convergence_study(airfoil_type="naca", code="4415", alpha=5.0)
