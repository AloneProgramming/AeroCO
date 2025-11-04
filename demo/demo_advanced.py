import numpy as np
import matplotlib.pyplot as plt
from methods.discrete_vortex import DiscreteVortexMethod
from methods.thin_airfoil import ThinAirfoilTheory
from core.potential_flows import *
from core.camber_functions import *
from visualization.plot_flow import *


def get_airfoil_coordinates(airfoil, n_points=200):
    x_points = np.linspace(0, 1, n_points)
    if airfoil.camber_func is not None:
        y_points = airfoil.camber_func(x_points)
    else:
        y_points = np.zeros_like(x_points)
    return np.column_stack([x_points, y_points])


def airfoil_flow_analysis(airfoil):
    flow_model = FlowModel()
    flow_model.add_component(UniformFlow(strength=airfoil.U_inf, alpha=airfoil.alpha))

    for i in range(airfoil.n_panels):
        x_v = airfoil.vortex_positions[i]
        if airfoil.camber_func is not None:
            y_v = airfoil.camber_func(x_v)
        else:
            y_v = 0.0
        flow_model.add_component(Vortex(strength=airfoil.circulations[i], dx=x_v, dy=y_v))

    airfoil_coords = get_airfoil_coordinates(airfoil)

    plot_combined_flow(flow_model, xlim=(-0.5, 1.5), ylim=(-0.8, 0.8),
                       resolution=100, U_inf=airfoil.U_inf, airfoil_coords=airfoil_coords)


if __name__ == "__main__":
    airfoil = DiscreteVortexMethod(
        alpha_degrees=10,
        U_inf=10,
        n_panels=200,
        camber_func=naca_4_digit_camber(code="8315"),
        debug=False)

    airfoil_flow_analysis(airfoil=airfoil)