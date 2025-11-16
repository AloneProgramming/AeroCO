import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from core.potential_flows import *
from core.camber_functions import parabolic_camber, naca_4_digit_camber
from methods.thin_airfoil import ThinAirfoilTheory
from methods.diffusion_velocity import *
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


def demo_1D_diffusion_animated():
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


def validation_1D_diffusion():
    print("1D rectangular diffusion validation")

    h = 1.0
    nu = 1.0
    sigma = 0.4 * h
    n_vortices = 50

    model = DiffusionVortex1D(viscosity=nu, sigma=sigma)
    model.create_rectangular_impulse(h, n_vortices)

    dimensionless_times = [0, 0.2, 1.0, 4.0, 20.0]  # νt/h²
    dt = 0.04 * h ** 2 / nu

    x_plot = np.linspace(-3 * h, 3 * h, 200)

    for i, dim_time in enumerate(dimensionless_times):
        t = dim_time * h ** 2 / nu

        if dim_time > 0:
            n_steps = int(t / dt)
            print(f"Evolution to t = {t:.3f}; (νt/h² = {dim_time}); steps: {n_steps}")

            for step in range(n_steps):
                model.step(dt)

        omega_numerical = model.get_numerical_solution(x_plot)
        omega_analytical = analytical_solution_1d(x_plot, t, h, nu)
        plt.figure(figsize=(10, 6))

        plt.plot(x_plot, omega_numerical, 'b-', linewidth=2,
                 label=f'Numerical (N={n_vortices})')
        plt.plot(x_plot, omega_analytical, 'r--', linewidth=2,
                 label='Analytical')

        vortex_x = [v[0] for v in model.vortices]
        vortex_omega = model.get_numerical_solution(vortex_x)
        plt.scatter(vortex_x, vortex_omega, color='blue', s=30, alpha=0.6,
                    label='Vortices')

        plt.xlabel('x/h')
        plt.ylabel('Vorticity ω')
        plt.title(f'Rectangular impulse diffusion: νt/h² = {dim_time}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(-3, 3)
        plt.ylim(-0.1, 1.1)

        error = np.sqrt(np.mean((omega_numerical - omega_analytical) ** 2))
        print(f"RMS err at νt/h²={dim_time}: {error:.6f}")

        plt.show()


def verification_1D_diffusion():
    print("1D rectangular diffusion verification")

    h = 1.0
    nu = 1.0
    sigma = 0.4 * h
    t_final = 1.0
    dt = 0.01

    n_vortices_list = [10, 20, 40, 80, 160, 320]
    err = []

    x_plot = np.linspace(-3*h, 3*h, 300)

    for n_vortices in n_vortices_list:
        model = DiffusionVortex1D(viscosity=nu, sigma=sigma)
        model.create_rectangular_impulse(h, n_vortices)

        n_steps = int(t_final / dt)
        for step in range(n_steps):
            model.step(dt)

        omega_num = model.get_numerical_solution(x_plot)
        omega_anal = analytical_solution_1d(x_plot, t_final, h, nu)

        error = np.sqrt(np.mean((omega_num - omega_anal) ** 2))
        err.append(error)

        print(f"N={n_vortices}: err = {error:.6f}")

    plt.figure(figsize=(10, 6))
    plt.loglog(n_vortices_list, err, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of vortices')
    plt.ylabel('RMS err')
    plt.title('Verification (grid convergence)')
    plt.grid(True, which="both", alpha=0.3)
    plt.show()


def demo_2D_diffusion():
    print("2D diffusion demo")

    model = DiffusionVortex2D(viscosity=0.5, sigma=0.3)

    n_vortices = 20

    for i in range(n_vortices):
        r = 0.5 * np.sqrt(np.random.rand())
        theta = 2 * np.pi * np.random.rand()
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        model.add_vortex(x, y, 1.0/n_vortices)

    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)

    for step in range(10):
        model.step(dt=0.1)

        omega = model.vorticity_field(X, Y)

        plt.figure(figsize=(10, 8))
        plt.contourf(X, Y, omega, levels=20, cmap='viridis')
        plt.colorbar(label='Vorticity ω')

        vortex_x = [v[0] for v in model.vortices]
        vortex_y = [v[1] for v in model.vortices]
        plt.scatter(vortex_x, vortex_y, color='white', s=30, alpha=0.7, label='Vortices')

        plt.title(f'2D Diffusion - Step {step}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.legend()
        plt.show()

        u_d, v_d = model.calculate_diffusion_velocity()
        print(f"Step {step}: Max diffusion velocity = {np.max(np.sqrt(u_d ** 2 + v_d ** 2)):.6f}")

if __name__ == "__main__":
    # demo_thin_airfoil()
    # demo_potential_flows()
    # gaussian_vortex_test()
    # demo_1D_diffusion_animated()
    # validation_1d_diffusion()
    # verification_1d_diffusion()
    demo_2D_diffusion()
