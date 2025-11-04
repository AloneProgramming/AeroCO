import numpy as np
import matplotlib.pyplot as plt
from methods.discrete_vortex import DiscreteVortexMethod
from methods.thin_airfoil import ThinAirfoilTheory
from core.potential_flows import *
from core.camber_functions import *
from visualization.plot_flow import *
from visualization.plot_data import *


def airfoil_coordinates(airfoil, n_points=200):
    x_points = np.linspace(0, 1, n_points)
    if airfoil.camber_func is not None:
        y_points = airfoil.camber_func(x_points)
    else:
        y_points = np.zeros_like(x_points)
    return np.column_stack([x_points, y_points])


def airfoil_flow(airfoil):
    flow_model = FlowModel()
    flow_model.add_component(UniformFlow(strength=airfoil.U_inf, alpha=airfoil.alpha))

    for i in range(airfoil.n_panels):
        x_v = airfoil.vortex_positions[i]
        if airfoil.camber_func is not None:
            y_v = airfoil.camber_func(x_v)
        else:
            y_v = 0.0
        flow_model.add_component(Vortex(strength=airfoil.circulations[i], dx=x_v, dy=y_v))

    airfoil_coords = airfoil_coordinates(airfoil)

    plot_combined_flow(flow_model, xlim=(-0.5, 1.5), ylim=(-0.8, 0.8),
                       resolution=100, U_inf=airfoil.U_inf, airfoil_coords=airfoil_coords)

def airfoil_data(airfoil, type="CL", alpha_range=(-5, 15, 5)):
    alpha_values = np.arange(alpha_range[0], alpha_range[1] + 1, alpha_range[2])

    cl_dvm = []
    for alpha in alpha_values:
        dvm = DiscreteVortexMethod(
            alpha_degrees=alpha,
            n_panels=airfoil.n_panels,
            camber_func=airfoil.camber_func,
            debug=False
        )
        cl, _ = dvm.calculate_aerodynamics()
        cl_dvm.append(cl)

    if airfoil.camber_func:
        tat = ThinAirfoilTheory(camber_func=airfoil.camber_func)
        cl_tat = [tat.calculate_lift(alpha) for alpha in alpha_values]
    else:
        tat = ThinAirfoilTheory()
        cl_tat = [tat.calculate_lift(alpha) for alpha in alpha_values]

    plot_data(
        data_list=[(alpha_values, cl_dvm), (alpha_values, cl_tat)],
        colors=['red', 'black'],
        labels=['dvm', 'tat']
    )

if __name__ == "__main__":
    airfoil_test = DiscreteVortexMethod(
        alpha_degrees=10,
        U_inf=10,
        n_panels=200,
        camber_func=naca_4_digit_camber(code="8315"),
        debug=False)

    airfoil_data(airfoil=airfoil_test)
    airfoil_flow(airfoil=airfoil_test)