import numpy as np
import matplotlib.pyplot as plt


class DiffusionVelocityMethod:
    def __init__(self, viscosity=1.0, sigma=1.0, U_inf=1.0, alpha=0.0):
        self.nu = viscosity
        self.sigma = sigma
        self.U_inf = U_inf
        self.alpha = np.radians(alpha)

        self.positions = np.array([]).reshape(0, 2)
        self.circulations = np.array([])
        self.history = []

    def add_vortex(self, x, y, gamma):
        self.positions = np.vstack([self.positions, [x, y]])
        self.circulations = np.append(self.circulations, gamma)

    def vorticity_field(self, X, Y):
        omega = np.zeros_like(X)

        for (x_v, y_v), gamma_v in zip(self.positions, self.circulations):
            r_sq = (X - x_v) ** 2 + (Y - y_v) ** 2
            kernel = (gamma_v / (np.pi * self.sigma ** 2)) * np.exp(-r_sq / self.sigma ** 2)
            omega += kernel

        return omega

    def _freestream_velocity(self, x, y):
        u = np.ones_like(x) * self.U_inf * np.cos(self.alpha)
        v = np.ones_like(y) * self.U_inf * np.sin(self.alpha)
        return u, v

    def _vortex_induced_velocity(self, x, y, x_v, y_v, gamma_v):
        dx = x - x_v
        dy = y - y_v
        r_sq = dx ** 2 + dy ** 2 + 1e-12

        u_ind = (-gamma_v / (2 * np.pi)) * (dy / r_sq)
        v_ind = (gamma_v / (2 * np.pi)) * (dx / r_sq)

        return u_ind, v_ind

    def calculate_convection_velocity(self):
        if len(self.positions) == 0:
            return np.array([]), np.array([])

        x, y = self.positions[:, 0], self.positions[:, 1]
        u_conv, v_conv = self._freestream_velocity(x, y)

        for i, ((x_v, y_v), gamma_v) in enumerate(zip(self.positions, self.circulations)):
            u_ind, v_ind = self._vortex_induced_velocity(x, y, x_v, y_v, gamma_v)
            u_conv += u_ind
            v_conv += v_ind

        return u_conv, v_conv

    def calculate_diffusion_velocity(self):
        if len(self.positions) == 0:
            return np.array([]), np.array([])

        x, y = self.positions[:, 0], self.positions[:, 1]

        omega = np.zeros_like(x)
        d_omega_dx = np.zeros_like(x)
        d_omega_dy = np.zeros_like(x)

        for (x_v, y_v), gamma_v in zip(self.positions, self.circulations):
            dx = x - x_v
            dy = y - y_v
            r_sq = dx ** 2 + dy ** 2

            kernel = (gamma_v / (np.pi * self.sigma ** 2)) * np.exp(-r_sq / self.sigma ** 2)
            omega += kernel

            d_kernel_dx = kernel * (-2 * dx / self.sigma ** 2)
            d_kernel_dy = kernel * (-2 * dy / self.sigma ** 2)

            d_omega_dx += d_kernel_dx
            d_omega_dy += d_kernel_dy

        omega_safe = np.where(np.abs(omega) > 1e-12, omega, 1e-12 * np.sign(omega + 1e-20))

        u_d = -self.nu / omega_safe * d_omega_dx
        v_d = -self.nu / omega_safe * d_omega_dy

        return u_d, v_d

    def step(self, dt):
        if len(self.positions) == 0:
            return

        self.history.append((self.positions.copy(), self.circulations.copy()))

        u_conv, v_conv = self.calculate_convection_velocity()
        u_diff, v_diff = self.calculate_diffusion_velocity()

        self.positions[:, 0] += (u_conv + u_diff) * dt
        self.positions[:, 1] += (v_conv + v_diff) * dt

    def total_velocity_field(self, X, Y):
        u, v = self._freestream_velocity(X, Y)

        for (x_v, y_v), gamma_v in zip(self.positions, self.circulations):
            u_ind, v_ind = self._vortex_induced_velocity(X, Y, x_v, y_v, gamma_v)
            u += u_ind
            v += v_ind

        return u, v