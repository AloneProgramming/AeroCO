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

    if type == "CL":
        cl_dvm = []
        for alpha in alpha_values:
            dvm = DiscreteVortexMethod(
                alpha_degrees=alpha,
                n_panels=airfoil.n_panels,
                camber_func=airfoil.camber_func,
                debug=airfoil.debug
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
            labels=['dvm', 'tat'],
            x_label="alpha",
            y_label="Cl",
            title="Cl(alpha)"
        )

    if type == "CM_LE":
        cm_le_dvm = []
        for alpha in alpha_values:
            dvm = DiscreteVortexMethod(
                alpha_degrees=alpha,
                n_panels=airfoil.n_panels,
                camber_func=airfoil.camber_func,
                debug=airfoil.debug
            )
            _, cm_le = dvm.calculate_aerodynamics()
            cm_le_dvm.append(cm_le)

        if airfoil.camber_func:
            tat = ThinAirfoilTheory(camber_func=airfoil.camber_func)
            cm_le_tat = [tat.calculate_leading_edge_moment(alpha) for alpha in alpha_values]
        else:
            tat = ThinAirfoilTheory()
            cm_le_tat = [tat.calculate_leading_edge_moment(alpha) for alpha in alpha_values]

        plot_data(
            data_list=[(alpha_values, cm_le_dvm), (alpha_values, cm_le_tat)],
            colors=['red', 'black'],
            labels=['dvm', 'tat'],
            x_label="alpha",
            y_label="Cm_LE",
            title="Cm_LE(alpha)"
        )

def airfoil_grid_convergence(airfoil):
    panel_counts = [10, 20, 40, 80, 160, 320]
    cl_dvm = []

    for panels in panel_counts:
        dvm = DiscreteVortexMethod(
            alpha_degrees=airfoil.alpha,
            n_panels=panels,
            camber_func=airfoil.camber_func,
            debug=airfoil.debug
        )
        cl, _ = dvm.calculate_aerodynamics()
        cl_dvm.append(cl)

    tat = ThinAirfoilTheory(camber_func=airfoil.camber_func)
    cl_tat = np.full(np.size(panel_counts), tat.calculate_lift(airfoil.alpha))

    plot_data(
        data_list=[(panel_counts, cl_dvm), (panel_counts, cl_tat)],
        colors=['red', 'black'],
        labels=['dvm', 'tat'],
        x_label="number of panels",
        y_label="Cl",
        title="Grid covergence"
    )


if __name__ == "__main__":
    airfoil_test = DiscreteVortexMethod(
        alpha_degrees=0,
        U_inf=10,
        n_panels=200,
        camber_func=naca_4_digit_camber(code="8315"),
        debug=True)

    #airfoil_data(airfoil=airfoil_test, type="CM_LE", alpha_range=(0, 10, 5))
    #airfoil_flow(airfoil=airfoil_test)
    airfoil_grid_convergence(airfoil=airfoil_test)