import numpy as np
import matplotlib.pyplot as plt
from thin_airfoil_theory import ThinAirfoil
from potential_flows import UniformFlow, Source, Doublet


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

    uniform = UniformFlow(u_inf=1.0, alpha=0)
    source = Source(strength=2.0, x=0, y=0)
    #doublet = Doublet(strength=3, x=0, y=0)

    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    psi_total = uniform.stream_function(X, Y) + source.stream_function(X, Y)

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, psi_total, levels=50, colors='blue')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Potential flows demo')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    #test_thin_airfoil()
    demo_potential_flows()