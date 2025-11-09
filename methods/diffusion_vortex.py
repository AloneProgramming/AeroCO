import numpy as np


class DiffusionVortex1D:
    def __init__(self, viscosity=1.0, sigma=1.0):
        self.nu = viscosity  # kinematic viscosity
        self.sigma = sigma
        self.vortices = []
        self.history = []

    def add_vortex(self, x, gamma):
        self.vortices.append([x, gamma])

    def calculate_vorticity_and_gradient(self, x_points):
        if x_points is None:
            x_points = np.array([v[0] for v in self.vortices])

        x_points = np.array(x_points)
        omega = np.zeros_like(x_points)
        d_omega_dx = np.zeros_like(x_points)

        for x_v, gamma_v in self.vortices:
            r = x_points - x_v
            r_sq = r ** 2

            kernel = gamma_v / (np.sqrt(np.pi) * self.sigma) * np.exp(-r_sq / self.sigma ** 2)
            omega += kernel

            kernel_derivative = kernel * (-2 * r / self.sigma ** 2)
            d_omega_dx += kernel_derivative

        return omega, d_omega_dx

    def calculate_diffusion_velocity(self):
        if not self.vortices:
            return np.array([])

        vortex_positions = np.array([v[0] for v in self.vortices])

        omega, d_omega_dx = self.calculate_vorticity_and_gradient(vortex_positions)

        omega_safe = np.where(np.abs(omega) > 1e-12, omega, 1e-12 * np.sign(omega + 1e-20))

        u_d = -self.nu / omega_safe * d_omega_dx

        return u_d

    def step(self, dt):
        if not self.vortices:
            return

        u_d = self.calculate_diffusion_velocity()

        for i, (x, gamma) in enumerate(self.vortices):
            new_x = x + u_d[i] * dt
            self.vortices[i] = [new_x, gamma]

        self.history.append(np.array(self.vortices))

