import numpy as np
import matplotlib.pyplot as plt
from visualization.plot_data import *


class DiffusionVortex1D:
    def __init__(self, viscosity=1.0, sigma=1.0):
        self.nu = viscosity  # kinematic viscosity
        self.sigma = sigma
        self.vortices = []
        self.history = []

    def add_vortex(self, x, gamma):
        self.vortices.append([x, gamma])

    def vorticity_field(self, x_points):
        x_points = np.array(x_points)
        omega = np.zeros_like(x_points)

        for x_v, gamma_v in self.vortices:
            r_sq = (x_points - x_v) ** 2
            omega += (gamma_v / (np.sqrt(np.pi) * self.sigma)) * np.exp(-r_sq / self.sigma ** 2)

        return omega


def plot_vorticity_field(omega, x):
    plot_data(
        data_list=[
            (x, omega)
        ],
        colors=['red'],
        labels=['vorticity'],
        x_label='x',
        y_label='omega',
        title='Vorticity field demo'
    )


def demo_test():
    test = DiffusionVortex1D()
    test.add_vortex(-2, 2)
    test.add_vortex(2, -2)

    x = np.linspace(-5, 5, 50)

    omega = test.vorticity_field(x_points=x)

    plot_vorticity_field(omega=omega, x=x)

demo_test()