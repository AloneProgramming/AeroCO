import numpy as np
import matplotlib.pyplot as plt
from core.potential_flows import *
from core.camber_functions import parabolic_camber, naca_4_digit_camber
from methods.thin_airfoil import ThinAirfoilTheory
from methods.diffusion_vortex import DiffusionVortex1D
from visualization.plot_flow import *
from visualization.plot_data import *


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


def demo_potential_flows():
    print("Potential Flows Demo")

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


def gaussian_vortex_test():
    print("Gaussian Vortex Demo")

    flow = FlowModel()
    flow.add_component(GaussianVortex(strength=2.0, sigma=0.5))
    plot_velocity_field(flow, xlim=(-5, 5), ylim=(-5, 5), resolution=40)
    plot_vorticity_field(flow, xlim=(-5, 5), ylim=(-5, 5), resolution=40)


def demo_diffusion():
    test = DiffusionVortex1D(viscosity=1.0, sigma=0.5)
    test.add_vortex(-0.5, 1.0)
    test.add_vortex(0.5, 1.0)

    x_plot = np.linspace(-2, 2, 100)

    print("Initial vortex positions:", [v[0] for v in test.vortices])

    vortex_positions = np.array([v[0] for v in test.vortices])
    omega, d_omega_dx = test.calculate_vorticity_and_gradient(vortex_positions)

    print("Vorticity at vortex positions:", omega)
    print("Vorticity gradient at vortex positions:", d_omega_dx)
    print("Diffusion velocity:", test.calculate_diffusion_velocity())

    for step in range(10):
        test.step(dt=0.1)

        omega, grad = test.calculate_vorticity_and_gradient(x_plot)
        vortex_x = [v[0] for v in test.vortices]

        print(f"Step {step}: positions = {vortex_x}")
        print(f"Step {step}: diffusion velocity = {test.calculate_diffusion_velocity()}")

        plt.figure(figsize=(10, 4))
        plt.plot(x_plot, omega, 'b-', label=f'ω (step {step})', linewidth=2)
        plt.plot(x_plot, grad, 'r--', label='dω/dx', alpha=0.7)

        vortex_omega = test.calculate_vorticity_and_gradient(vortex_x)[0]
        plt.scatter(vortex_x, vortex_omega, color='black', s=80, zorder=5, label='Vortices')

        plt.legend()
        plt.title(f'Vortex diffusion - step {step}')
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('Vorticity ω')
        plt.show()


if __name__ == "__main__":
    # demo_thin_airfoil()
    # demo_potential_flows()
    # gaussian_vortex_test()
    demo_diffusion()
