import numpy as np
import matplotlib.pyplot as plt
from camber_functions import *


class DiscreteVortexAirfoil:
    def __init__(self, chord=1.0, alpha_degrees=5.0, n_panels=20, U_inf=1.0,
                 camber_func=None, debug=True):
        self.chord = chord
        self.alpha = np.radians(alpha_degrees)
        self.n_panels = n_panels
        self.U_inf = U_inf
        self.camber_func = camber_func
        self.debug = debug

        self.setup_geometry()
        self.setup_vortices()
        self.calculate_influence_matrix()
        self.calculate_rhs()
        self.solve_circulation()
        #self.calculate_aerodynamics()

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
            print("Geometry setup is done.")
            print(f"Nodes: {self.x_nodes}")
            print(f"Control points: {self.control_points}")
            print(f"dz/dx: {self.dz_dx}")

    def setup_vortices(self):
        self.vortex_positions = self.x_nodes[:-1] + 0.25 * (self.x_nodes[1:] - self.x_nodes[:-1])

        if self.debug:
            print("Vortices setup is done.")
            print(f"Vortices positions: {self.vortex_positions}")

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
            print("Influence matrix is calculated.")
            print(f"Matrix size: {self.A.shape}")

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
            print("RHS is calculated.")
            print(f"b = {self.b}")

    def solve_circulation(self):
        self.circulations = np.linalg.solve(self.A, self.b)

        if self.debug:
            print("Circulation is calculated.")
            print(f"Г = {self.circulations}")
            print(f"Г summ = {np.sum(self.circulations):.6f}")

    def plot_basic_results(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].plot(self.vortex_positions, self.circulations, 'bo-', linewidth=2)
        axes[0, 0].set_xlabel('x/c')
        axes[0, 0].set_ylabel('Circulatuion Γ')
        axes[0, 0].set_title('Г distribution along chord')
        axes[0, 0].grid(True)

        theta_analytic = np.linspace(0, np.pi, 100)
        x_analytic = 0.5 * (1 - np.cos(theta_analytic))
        gamma_analytic = 2 * self.U_inf * (1 + np.cos(theta_analytic)) / np.sin(theta_analytic) * self.alpha

        axes[0, 1].plot(x_analytic, gamma_analytic, 'r-', label='Analytical')
        axes[0, 1].plot(self.vortex_positions, self.circulations, 'bo-', label='Numerical')
        axes[0, 1].set_xlabel('x/c')
        axes[0, 1].set_ylabel('γ(θ)')
        axes[0, 1].set_title('Compassion (analytical vs numerical)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[1, 0].plot(self.control_points, np.zeros_like(self.control_points), 'ro', label='Control points')
        axes[1, 0].plot(self.vortex_positions, np.zeros_like(self.vortex_positions), 'b^', label='Vortices')
        axes[1, 0].set_xlabel('x/c')
        axes[1, 0].set_ylabel('y/c')
        axes[1, 0].set_title('Geometry')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].axis('equal')

        im = axes[1, 1].imshow(self.A, cmap='coolwarm', aspect='auto')
        axes[1, 1].set_title('Influence matrix A')
        axes[1, 1].set_xlabel('Vortex index')
        axes[1, 1].set_ylabel('Control point vortex')
        plt.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()
        plt.show()

    def calculate_aerodynamics(self):
        total_circulation = np.sum(self.circulations)
        self.Cl = 2.0 * total_circulation / self.U_inf
        self.Cm_LE = -2.0 * np.sum(self.circulations * self.vortex_positions) / self.U_inf
        x_ac = 0.25 * self.chord
        self.Cm_AC = self.Cm_LE + self.Cl * x_ac

        #print(f"AOA = {np.degrees(self.alpha):.0f}°; Cl(numerical) = {-self.Cl:.6f}; Cl(analytical) = {2 * np.pi * (self.alpha + 2 * 0.05):.6f}")
        #print(f"Cm_LE = {self.Cm_LE:.6f}")
        #print(f"Cm_AC = {self.Cm_AC:.6f}")
        return self.Cl

alpha = 5
h = 0.05

panels = [5, 10, 20, 40, 80, 160, 320]
Cl_numerical = np.array([])
Cl_analytical = np.full(7, 2 * np.pi * (np.radians(alpha) + 2 * h))

for i in panels:
    airfoil = DiscreteVortexAirfoil(alpha_degrees=alpha, n_panels=i, debug=False, camber_func=parabolic_camber(h=h))
    Cl_numerical = np.append(Cl_numerical, -airfoil.calculate_aerodynamics())

plt.plot(panels, Cl_numerical, marker='o', linestyle='-', color='red', label='D.V.M.')
plt.plot(panels, Cl_analytical, marker='', linestyle='--', color='black', label='Analytical')

plt.title('Grid sensitivity')
plt.xlabel('Number of panels')
plt.ylabel('Cl')
plt.grid(True)
plt.legend()

plt.show()
