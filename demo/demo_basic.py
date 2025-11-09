import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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


def demo_diffusion_animated():
    test = DiffusionVortex1D(viscosity=1.0, sigma=0.5)
    test.add_vortex(-0.25, 1.0)
    test.add_vortex(-0.125, 1.0)
    test.add_vortex(0, 1.0)
    test.add_vortex(0.125, 1.0)
    test.add_vortex(0.25, 1.0)

    x_plot = np.linspace(-2, 2, 100)
    steps = 100
    dt = 0.1

    fig, ax = plt.subplots(figsize=(10, 4))
    line_omega, = ax.plot([], [], 'b-', label='ω', linewidth=2)
    line_grad, = ax.plot([], [], 'r--', label='dω/dx', alpha=0.7)
    scatter_vortices = ax.scatter([], [], color='black', s=80, zorder=5, label='Vortices')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-10, 10)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('Vorticity ω')

    def init():
        line_omega.set_data([], [])
        line_grad.set_data([], [])
        scatter_vortices.set_offsets(np.empty((0, 2)))
        return line_omega, line_grad, scatter_vortices

    def update(frame):
        test.step(dt=dt)
        omega, grad = test.calculate_vorticity_and_gradient(x_plot)
        vortex_x = np.array([v[0] for v in test.vortices])
        vortex_omega = test.calculate_vorticity_and_gradient(vortex_x)[0]

        line_omega.set_data(x_plot, omega)
        line_grad.set_data(x_plot, grad)
        scatter_vortices.set_offsets(np.c_[vortex_x, vortex_omega])

        ax.set_title(f'Vortex diffusion - step {frame + 1}')
        return line_omega, line_grad, scatter_vortices

    ani = FuncAnimation(fig, update, frames=steps,
                        init_func=init, blit=True, interval=200, repeat=True)

    plt.show()
    return ani


if __name__ == "__main__":
    # demo_thin_airfoil()
    # demo_potential_flows()
    # gaussian_vortex_test()
    demo_diffusion_animated()
