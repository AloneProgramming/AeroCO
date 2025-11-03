import numpy as np
import matplotlib.pyplot as plt
from core.potential_flows import FlowModel, UniformFlow, Vortex, Doublet
from core.camber_functions import parabolic_camber, naca_4_digit_camber
from methods.thin_airfoil import ThinAirfoilTheory
from methods.discrete_vortex import DiscreteVortexMethod
from visualization.plot_flow import plot_velocity_field, plot_pressure_field

def demo_thin_airfoil():
    print("Thin Airfoil Theory Demo")

    for alpha_deg in [0, 5, 10]:
        airfoil = ThinAirfoilTheory()
        results = airfoil.calculate_coefficients(alpha_degrees=alpha_deg)
        print(f"AOA: {alpha_deg}° -> Cl = {results['Cl']:.4f}, Cm_AC = {results['Cm_AC']:.4f}")

    h = 0.05
    def dz_dx_parabolic(x):
        return 4 * h * (1 - 2 * x)

    airfoil = ThinAirfoilTheory()
    results = airfoil.calculate_coefficients(alpha_degrees=10.0, dz_dx_func=dz_dx_parabolic)
    print(f"Parabolic camber (h=0.05, AOA=10°): Cl = {results['Cl']:.4f}")

    Cl_analytical = 2 * np.pi * (np.radians(10.0) + 2 * h)
    print(f"Analytical Cl (same): {Cl_analytical:.4f}")


def demo_discrete_vortex():
    print("\nDiscrete Vortex Method Demo")

    dvm = DiscreteVortexMethod(alpha_degrees=5.0, n_panels=20, debug=False)
    cl, cm = dvm.calculate_aerodynamics()
    print(f"Symmetric airfoil: Cl = {cl:.4f}")

    camber_func = parabolic_camber(h=0.04)
    dvm_camber = DiscreteVortexMethod(alpha_degrees=5.0, n_panels=20,
                                      camber_func=camber_func, debug=False)
    cl_camber, cm_camber = dvm_camber.calculate_aerodynamics()
    print(f"Cambered airfoil: Cl = {cl_camber:.4f}")

    dvm_camber.plot_detailed_results()


def demo_potential_flows():
    print("\nPotential Flows Demo")

    flow = FlowModel()
    flow.add_component(Vortex(strength=1.0, dx=0, dy=0))
    plot_velocity_field(flow, xlim=(-3, 3), ylim=(-3, 3), resolution=30)

    flow_pair = FlowModel()
    flow_pair.add_component(Vortex(strength=10.0, dx=-1, dy=0))
    flow_pair.add_component(Vortex(strength=-10.0, dx=1, dy=0))
    plot_velocity_field(flow_pair, xlim=(-3, 3), ylim=(-3, 3), resolution=30)

    flow_cylinder = FlowModel()
    flow_cylinder.add_component(UniformFlow(strength=5.0, alpha=0))
    R = 2.0
    strength_doublet = 2 * np.pi * 5.0 * R ** 2
    flow_cylinder.add_component(Doublet(strength=strength_doublet, dx=0, dy=0))
    plot_velocity_field(flow_cylinder, xlim=(-5, 5), ylim=(-5, 5), resolution=40)


if __name__ == "__main__":
    #demo_thin_airfoil()
    #demo_discrete_vortex()
    demo_potential_flows()