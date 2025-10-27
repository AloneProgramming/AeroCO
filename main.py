from thin_airfoil_theory import *
from potential_flows import *


def demo_symmetric_airfoil():
    print("Thin Airfoil Theory")
    print("=" * 40)

    for alpha_deg in [0, 5, 10]:
        airfoil = ThinAirfoil(alpha_degrees=alpha_deg)
        Cl, Cm_LE, Cm_AC, A0, A1, A2 = airfoil.calculate_coefficients()

        print(f"\nAOA: {alpha_deg}°")
        print(f"Cl = {Cl:.4f}")
        print(f"Cm_AC = {Cm_AC:.4f}")

    airfoil = ThinAirfoil(alpha_degrees=5.0)
    Cl, Cm_LE, Cm_AC, A0, A1, A2 = airfoil.calculate_coefficients()
    airfoil.plot_results(Cl, Cm_LE, Cm_AC, A0, A1, A2)


def demo_parabolic_airfoil():
    h = 0.05  # max camber height

    def dz_dx_parabolic(x):
        return 4 * h * (1 - 2 * x)

    airfoil = ThinAirfoil(alpha_degrees=5.0)
    Cl, Cm_LE, Cm_AC, A0, A1, A2 = airfoil.calculate_coefficients(dz_dx_parabolic)

    print("Parabolic airfoil results:")
    print(f"AOA: {np.degrees(airfoil.alpha):.1f}°")
    print(f"h: {h}")
    print(f"A0: {A0:.4f}, A1: {A1:.4f}, A2: {A2:.4f}")
    print(f"Cl: {Cl:.4f}")
    print(f"Cm_LE: {Cm_LE:.4f}")
    print(f"Cm_AC: {Cm_AC:.4f}")

    Cl_analytical = 2 * np.pi * (airfoil.alpha + 2 * h)
    print(f"Cl (analytical): {Cl_analytical:.4f}")

    airfoil.plot_airfoil(dz_dx_parabolic)

def demo_potential_flows():
    print("Potential flows demo")
    print("=" * 40)

    flow = FlowModel()

    U_inf = 5.0

    flow.add_component(UniformFlow(strength=U_inf, alpha=np.radians(0)))
    # flow.add_component(SourceSink(strength=15.0, dx=-3.0, dy=0.0))
    # flow.add_component(SourceSink(strength=-15.0, dx=3.0, dy=0.0))
    # flow.add_component(Doublet(strength=15.0, dx=0.0, dy=0.0))
    flow.add_component(Vortex(strength=1.0, dx=0.0, dy=0.0))

    # flow.plot_velocity_field(xlim=(-5, 5), ylim=(-5, 5), resolution=200)
    flow.plot_pressure_field(xlim=(-5, 5), ylim=(-5, 5), resolution=200, U_inf=U_inf)


if __name__ == "__main__":
    #demo_symmetric_airfoil()
    demo_parabolic_airfoil()
    # demo_potential_flows()
