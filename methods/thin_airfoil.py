import numpy as np
import matplotlib.pyplot as plt
from core.airfoil import AirfoilAnalysis


class ThinAirfoilTheory(AirfoilAnalysis):
    def __init__(self, camber_func=None, name="Thin Airfoil"):
        super().__init__(name)
        self.camber_func = camber_func

    def calculate_coefficients_cambered(self, dz_dx_func, alpha_rad):
        theta = np.linspace(0, np.pi, 1000)
        x = 0.5 * (1 - np.cos(theta))

        dz_dx = dz_dx_func(x)

        A0 = alpha_rad - (1 / np.pi) * np.trapz(dz_dx, theta)
        A1 = (2 / np.pi) * np.trapz(dz_dx * np.cos(theta), theta)
        A2 = (2 / np.pi) * np.trapz(dz_dx * np.cos(2 * theta), theta)

        return A0, A1, A2

    def calculate_coefficients(self, alpha_degrees=5.0, dz_dx_func=None):
        alpha_rad = np.radians(alpha_degrees)

        if dz_dx_func is None:
            A0 = alpha_rad
            A1 = 0.0
            A2 = 0.0
        else:
            A0, A1, A2 = self.calculate_coefficients_cambered(dz_dx_func, alpha_rad)

        Cl = 2 * np.pi * (A0 + A1 / 2)
        Cm_LE = -0.5 * np.pi * (A0 + A1 - A2 / 2)
        Cm_AC = -0.25 * np.pi * (A1 - A2)

        results = {
            'Cl': Cl, 'Cm_LE': Cm_LE, 'Cm_AC': Cm_AC,
            'A0': A0, 'A1': A1, 'A2': A2
        }

        self.results = results
        return results

    def calculate_lift(self, alpha_degrees):
        if self.camber_func:
            results = self.calculate_coefficients(alpha_degrees, self._dz_dx_from_camber)
            return results['Cl']
        else:
            return 2 * np.pi * np.radians(alpha_degrees)

    def calculate_leading_edge_moment(self, alpha_degrees):
        if self.camber_func:
            results = self.calculate_coefficients(alpha_degrees, self._dz_dx_from_camber)
            return results['Cm_LE']
        else:
            return -0.5 * np.pi * np.radians(alpha_degrees)

    def calculate_quarter_chord_moment(self, alpha_degrees):
        if self.camber_func:
            results = self.calculate_coefficients(alpha_degrees, self._dz_dx_from_camber)
            return results['Cm_AC']
        else:
            return 0


    def _dz_dx_from_camber(self, x):
        if not self.camber_func:
            return np.zeros_like(x)

        dx = 1e-6
        y_plus = self.camber_func(x + dx)
        y_minus = self.camber_func(x - dx)
        return (y_plus - y_minus) / (2 * dx)