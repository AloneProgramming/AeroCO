from core.potential_flows import *
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

    flow_gaussian = FlowModel()
    flow_gaussian.add_component(GaussianVortex(strength=2.0, sigma=0.5))
    plot_velocity_field(flow_gaussian, xlim=(-5, 5), ylim=(-5, 5), resolution=40)
    plot_vorticity_field(flow_gaussian, xlim=(-5, 5), ylim=(-5, 5), resolution=40)


def demo_diffusion():
    print("Diffusion + convection demo")

    solver = DiffusionVelocityMethod(
        viscosity=0.1,
        sigma=0.2,
        U_inf=1.0,
        alpha=5.0
    )

    n_vortices = 20
    for i in range(n_vortices):
        r = 0.3 * np.sqrt(np.random.rand())
        theta = 2 * np.pi * np.random.rand()
        x = -1.0 + r * np.cos(theta)
        y = r * np.sin(theta)
        solver.add_vortex(x, y, 1.0 / n_vortices)

    x = np.linspace(-3, 3, 80)
    y = np.linspace(-2, 2, 60)
    X, Y = np.meshgrid(x, y)

    for step in range(3):
        solver.step(dt=0.1)

        omega = solver.vorticity_field(X, Y)

        plt.figure(figsize=(12, 8))
        plt.contourf(X, Y, omega, levels=25, cmap='viridis', alpha=0.8)
        plt.colorbar(label='Vorticity ω')

        vortex_x = solver.positions[:, 0]
        vortex_y = solver.positions[:, 1]
        plt.scatter(vortex_x, vortex_y, color='red', s=50, alpha=0.7, label='Vortices')

        plt.title(f'Step {step}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # demo_thin_airfoil()
    # demo_potential_flows()
    demo_diffusion()