from thin_airfoil_theory import ThinAirfoil


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

if __name__ == "__main__":
    test_thin_airfoil()