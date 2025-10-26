from thin_airfoil_theory import *
from potential_flows import *


def test_thin_airfoil():
    print("Thin Airfoil Theory")
    print("=" * 40)

    for alpha_deg in [0, 5, 10]:
        airfoil = ThinAirfoil(alpha_degrees=alpha_deg)
        Cl, Cm_LE, Cm_AC, A0, A1, A2 = airfoil.calculate_coefficients_symmetric()

        print(f"\nAOA: {alpha_deg}°")
        print(f"Cl = {Cl:.4f}")
        print(f"Cm_AC = {Cm_AC:.4f}")

    airfoil = ThinAirfoil(alpha_degrees=5.0)
    Cl, Cm_LE, Cm_AC, A0, A1, A2 = airfoil.calculate_coefficients_symmetric()
    airfoil.plot_results(Cl, Cm_LE, Cm_AC, A0, A1, A2)


def demo_potential_flows():
    print("Potential flows demo")
    print("=" * 40)

    flow = FlowModel()

    U_inf = 5.0

    flow.add_component(UniformFlow(strength=U_inf, alpha=np.radians(0)))
    #flow.add_component(SourceSink(strength=15.0, dx=-3.0, dy=0.0))
    #flow.add_component(SourceSink(strength=-15.0, dx=3.0, dy=0.0))
    #flow.add_component(Doublet(strength=15.0, dx=0.0, dy=0.0))
    flow.add_component(Vortex(strength=1.0, dx=0.0, dy=0.0))

    #flow.plot_velocity_field(xlim=(-5, 5), ylim=(-5, 5), resolution=200)
    flow.plot_pressure_field(xlim=(-5, 5), ylim=(-5, 5), resolution=200, U_inf=U_inf)

if __name__ == "__main__":
    #test_thin_airfoil()
    demo_potential_flows()