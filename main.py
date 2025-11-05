from demo.demo_advanced import *

if __name__ == "__main__":
    airfoil_test = DiscreteVortexMethod(
        alpha_degrees=10,
        U_inf=10,
        n_panels=200,
        camber_func=naca_4_digit_camber(code="8315"),
        debug=False)

    airfoil_data(airfoil=airfoil_test, type="CL", alpha_range=(-5, 15, 5))
    airfoil_data(airfoil=airfoil_test, type="CM_LE", alpha_range=(-5, 15, 5))
    airfoil_flow(airfoil=airfoil_test)
    airfoil_grid_convergence(airfoil=airfoil_test)
