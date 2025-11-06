import numpy as np
import matplotlib.pyplot as plt
from core.airfoil import AirfoilAnalysis


class DiscreteVortexMethod(AirfoilAnalysis):
    def __init__(self, chord=1.0, alpha_degrees=0.0, n_panels=20,
                 U_inf=1.0, camber_func=None, debug=True):
        super().__init__("Discrete Vortex Method")
        self.chord = chord
        self.alpha = np.radians(alpha_degrees)
        self.alpha_deg = alpha_degrees
        self.n_panels = n_panels
        self.U_inf = U_inf
        self.camber_func = camber_func
        self.debug = debug

        self.validate_parameters(alpha_degrees, n_panels)


    def compute_gradient(self, y, x):
        n = len(y)
        dydx = np.zeros(n)

        if n == 1:
            return dydx

        dydx[0] = (y[1] - y[0]) / (x[1] - x[0])

        for i in range(1, n - 1):
            dydx[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])

        dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])

        return dydx

    def setup_geometry(self):
        self.x_nodes = np.linspace(0, self.chord, self.n_panels + 1)
        self.control_points = self.x_nodes[:-1] + 0.75 * (self.x_nodes[1:] - self.x_nodes[:-1])

        if self.camber_func is not None:
            self.z_control = self.camber_func(self.control_points)
            self.dz_dx = self.compute_gradient(self.z_control, self.control_points)
        else:
            self.z_control = np.zeros_like(self.control_points)
            self.dz_dx = np.zeros_like(self.control_points)

        if self.debug:
            print("(...) Calculations started for alpha: ", np.degrees(self.alpha))
            print("✓ Geometry setup completed")

    def setup_vortices(self):
        self.vortex_positions = self.x_nodes[:-1] + 0.25 * (self.x_nodes[1:] - self.x_nodes[:-1])

        if self.debug:
            print("✓ Vortex positions defined")

    def calculate_influence_matrix(self):
        self.A = np.zeros((self.n_panels, self.n_panels))

        for i in range(self.n_panels):
            if self.camber_func is not None:
                tangent_slope = self.dz_dx[i]
                nx = -tangent_slope / np.sqrt(1 + tangent_slope ** 2)
                nz = 1.0 / np.sqrt(1 + tangent_slope ** 2)
            else:
                nx, nz = 0.0, 1.0

            for j in range(self.n_panels):
                x_v = self.vortex_positions[j]
                if self.camber_func is not None:
                    z_v = self.camber_func(x_v)
                else:
                    z_v = 0.0

                dx = self.control_points[i] - x_v
                dz = self.z_control[i] - z_v

                r_sq = dx ** 2 + dz ** 2

                u_ij = (-1.0 / (2 * np.pi)) * dz / r_sq
                w_ij = (1.0 / (2 * np.pi)) * dx / r_sq

                normal_velocity = u_ij * nx + w_ij * nz

                self.A[i, j] = normal_velocity

        if self.debug:
            print("✓ Influence matrix is calculated.")

    def calculate_rhs(self):
        if self.camber_func is not None:
            b_values = np.zeros(self.n_panels)
            for i in range(self.n_panels):
                tangent_slope = self.dz_dx[i]
                surface_angle = np.arctan(tangent_slope)
                effective_alpha = self.alpha - surface_angle
                b_values[i] = -self.U_inf * np.sin(effective_alpha)
        else:
            b_values = -self.U_inf * (self.alpha - self.dz_dx)

        self.b = b_values

        if self.debug:
            print("✓ RHS is calculated.")

    def solve_circulation(self):
        self.circulations = np.linalg.solve(self.A, self.b)

        if self.debug:
            print("✓ Circulation is calculated.")

    def calculate_aerodynamics(self):
        self.setup_geometry()
        self.setup_vortices()
        self.calculate_influence_matrix()
        self.calculate_rhs()
        self.solve_circulation()

        total_circulation = np.sum(self.circulations)
        Cl = round(-2.0 * total_circulation / self.U_inf, 6)
        Cm_LE = round(2.0 * (np.sum(self.vortex_positions * self.circulations)) / self.U_inf, 6)
        Cm_AC = round(Cm_LE + Cl * 0.25, 6)

        self.results = {
            'Cl': Cl,
            'CM_LE': Cm_LE,
            #'CM_AC': Cm_AC,
            'circulation': total_circulation,
            'vortex_distribution': self.circulations
        }

        return Cl, Cm_LE, Cm_AC
