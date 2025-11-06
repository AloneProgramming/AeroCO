import numpy as np
import matplotlib.pyplot as plt
from methods.discrete_vortex import DiscreteVortexMethod
from methods.thin_airfoil import ThinAirfoilTheory
from core.potential_flows import *
from core.camber_functions import *
from visualization.plot_flow import *
from visualization.plot_data import *


def airfoil_coordinates(airfoil):
    x_points = np.linspace(0, 1, airfoil.n_panels + 1)
    if airfoil.camber_func is not None:
        y_points = airfoil.camber_func(x_points)
    else:
        y_points = np.zeros_like(x_points)
    return np.column_stack([x_points, y_points])


def airfoil_flow(airfoil):
    airfoil.calculate_aerodynamics()
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

    plot_velocity_field(flow_model, xlim=(-0.5, 1.5), ylim=(-0.8, 0.8),
                       resolution=100, airfoil_coords=airfoil_coords)


def airfoil_data(airfoil, types=("CL", "CM_LE", "CM_AC"), alpha_range=(-5, 15, 5)):
    alpha_values = np.arange(alpha_range[0], alpha_range[1] + 1, alpha_range[2])

    results = {
        'CL': {'dvm': [], 'tat': []},
        'CM_LE': {'dvm': [], 'tat': []},
        'CM_AC': {'dvm': [], 'tat': []}
    }

    tat = ThinAirfoilTheory(camber_func=airfoil.camber_func) if airfoil.camber_func else ThinAirfoilTheory()

    for alpha in alpha_values:
        dvm = DiscreteVortexMethod(
            alpha_degrees=alpha,
            n_panels=airfoil.n_panels,
            camber_func=airfoil.camber_func,
            debug=airfoil.debug
        )
        cl, cm_le, cm_ac = dvm.calculate_aerodynamics()

        results['CL']['dvm'].append(cl)
        results['CM_LE']['dvm'].append(cm_le)
        results['CM_AC']['dvm'].append(cm_ac)
        results['CL']['tat'].append(tat.calculate_lift(alpha))
        results['CM_LE']['tat'].append(tat.calculate_leading_edge_moment(alpha))
        results['CM_AC']['tat'].append(tat.calculate_quarter_chord_moment(alpha))

    for data_type in types:
        if data_type == "CL":
            plot_data(
                data_list=[
                    (alpha_values, results['CL']['dvm']),
                    (alpha_values, results['CL']['tat'])
                ],
                colors=['red', 'black'],
                labels=['dvm', 'tat'],
                x_label="alpha",
                y_label="Cl",
                title="Cl(alpha)"
            )
        elif data_type == "CM_LE":
            plot_data(
                data_list=[
                    (alpha_values, results['CM_LE']['dvm']),
                    (alpha_values, results['CM_LE']['tat'])
                ],
                colors=['red', 'black'],
                labels=['dvm', 'tat'],
                x_label="alpha",
                y_label="Cm_LE",
                title="Cm_LE(alpha)"
            )
        elif data_type == "CM_AC":
            plot_data(
                data_list=[
                    (alpha_values, results['CM_AC']['dvm']),
                    (alpha_values, results['CM_AC']['tat'])
                ],
                colors=['red', 'black'],
                labels=['dvm', 'tat'],
                x_label="alpha",
                y_label="Cm_AC",
                title="Cm_AC(alpha)"
            )

    return results, alpha_values

def airfoil_grid_convergence(airfoil):
    panel_counts = [10, 20, 40, 80, 160, 320]
    cl_dvm = []

    for panels in panel_counts:
        dvm = DiscreteVortexMethod(
            alpha_degrees=airfoil.alpha_deg,
            n_panels=panels,
            camber_func=airfoil.camber_func,
            debug=airfoil.debug
        )
        cl, _, _ = dvm.calculate_aerodynamics()
        cl_dvm.append(cl)

    tat = ThinAirfoilTheory(camber_func=airfoil.camber_func)
    cl_tat = np.full(np.size(panel_counts), tat.calculate_lift(airfoil.alpha_deg))

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
        alpha_degrees=5,
        n_panels=200,
        camber_func=naca_4_digit_camber(code="9412"),
        debug=True)

    airfoil_data(airfoil=airfoil_test, types=("CL", "CM_LE", "CM_AC",), alpha_range=(0, 15, 3))
    airfoil_flow(airfoil=airfoil_test)
    airfoil_grid_convergence(airfoil=airfoil_test)