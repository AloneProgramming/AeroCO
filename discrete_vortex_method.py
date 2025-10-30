import numpy as np
import matplotlib.pyplot as plt

class DiscreteVortexAirfoil:
    def __init__(self, chord=1.0, alpha_degrees=5.0, n_panels=20):
        self.chord = chord
        self.alpha = np.radians(alpha_degrees)
        self.n_panels = n_panels
        self.U_inf = 1.0

        self.setup_geometry()
        self.setup_vortices()
        self.calculate_influence_matrix()
        self.calculate_rhs()
        self.solve_circulation()
        self.calculate_aerodynamics()

    def setup_geometry(self):
        self.x_nodes = np.linspace(0, self.chord, self.n_panels + 1)
        self.control_points = 0.5 * (self.x_nodes[1:] + self.x_nodes[:-1])
        self.dz_dx = np.zeros_like(self.control_points)

        print("Geometry setup is done.")
        print(f"Nodes: {self.x_nodes}")
        print(f"Control points: {self.control_points}")

    def setup_vortices(self):
        self.vortex_positions = self.x_nodes[:-1] + 0.25 * (self.x_nodes[1:] - self.x_nodes[:-1])

        print("Vortices setup is done.")
        print(f"Vortices positions: {self.vortex_positions}")

    def calculate_influence_matrix(self):
        self.A = np.zeros((self.n_panels, self.n_panels))

        for i in range(self.n_panels):
            for j in range(self.n_panels):
                dx = self.control_points[i] - self.vortex_positions[j]
                dy = 0.0

                r_sq = dx**2 + dy**2

                w_ij = (1.0 / (2 * np.pi)) * dx / r_sq

                self.A[i, j] = w_ij

        print("Influence matrix is calculated.")
        print(f"Matrix size: {self.A.shape}")

    def calculate_rhs(self):
        self.b = self.U_inf * (self.alpha - self.dz_dx)

        print("RHS is calculated.")
        print(f"b = {self.b}")

    def solve_circulation(self):
        self.circulations = np.linalg.solve(self.A, self.b)

        print("Circulation is calculated.")
        print(f"Г = {self.circulations}")
        print(f"Г summ = {np.sum(self.circulations):.6f}")

    def calculate_aerodynamics(self):

        total_circulation = np.sum(self.circulations)
        self.Cl = 2.0 * total_circulation / self.U_inf

        print(f"Cl = {self.Cl:.6f}")

airfoil = DiscreteVortexAirfoil()
