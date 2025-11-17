import numpy as np
from scipy import special


def analytical_solution_1d(x, t, h, nu):
    if t == 0:
        return np.where(np.abs(x) <= h, 1.0, 0.0)
    else:
        term1 = (h - x) / (2 * np.sqrt(nu * t))
        term2 = (h + x) / (2 * np.sqrt(nu * t))
        return 0.5 * (special.erf(term1) + special.erf(term2))


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

    def create_rectangular_impulse(self, h, n_vortices):
        self.vortices = []
        x_positions = np.linspace(-h, h, n_vortices)
        total_circulation = 2 * h * 1.0
        gamma_per_vortex = total_circulation / n_vortices

        for x in x_positions:
            self.add_vortex(x, gamma_per_vortex)

    def get_numerical_solution(self, x_points):
        omega, _ = self.calculate_vorticity_and_gradient(x_points)
        return omega


class DiffusionVortex2D:
    def __init__(self, viscosity=1.0, sigma=1.0):
        self.nu = viscosity
        self.sigma = sigma
        self.vortices = []
        self.history = []

    def add_vortex(self, x, y, gamma):
        self.vortices.append([x, y, gamma])

    def vorticity_field(self, X, Y):
        omega = np.zeros_like(X)

        for x_v, y_v, gamma_v in self.vortices:
            r_sq = (X - x_v) ** 2 + (Y - y_v) ** 2

            kernel = (gamma_v / (np.pi * self.sigma ** 2)) * np.exp(-r_sq / self.sigma ** 2)
            omega += kernel

        return omega

    def calculate_vorticity_and_gradient(self, x_points, y_points=None):
        if y_points is None:
            x_points = np.array([v[0] for v in self.vortices])
            y_points = np.array([v[1] for v in self.vortices])

        x_points = np.array(x_points)
        y_points = np.array(y_points)

        omega = np.zeros_like(x_points)
        d_omega_dx = np.zeros_like(x_points)
        d_omega_dy = np.zeros_like(x_points)

        for x_v, y_v, gamma_v in self.vortices:
            dx = x_points - x_v
            dy = y_points - y_v
            r_sq = dx ** 2 + dy ** 2

            kernel = (gamma_v / (np.pi * self.sigma ** 2)) * np.exp(-r_sq / self.sigma ** 2)
            omega += kernel

            kernel_derivative_x = kernel * (-2 * dx / self.sigma ** 2)
            kernel_derivative_y = kernel * (-2 * dy / self.sigma ** 2)

            d_omega_dx += kernel_derivative_x
            d_omega_dy += kernel_derivative_y

        return omega, d_omega_dx, d_omega_dy

    def calculate_diffusion_velocity(self):
        if not self.vortices:
            return np.array([]), np.array([])

        omega, d_omega_dx, d_omega_dy = self.calculate_vorticity_and_gradient(None, None)

        omega_safe = np.where(np.abs(omega) > 1e-12, omega, 1e-12 * np.sign(omega + 1e-20))

        u_d = -self.nu / omega_safe * d_omega_dx
        v_d = -self.nu / omega_safe * d_omega_dy

        return u_d, v_d

    def step(self, dt, u_conv=None, v_conv=None):
        if not self.vortices:
            return

        u_d, v_d = self.calculate_diffusion_velocity()

        if u_conv is None:
            u_conv = np.zeros_like(u_d)
        if v_conv is None:
            v_conv = np.zeros_like(v_d)

        for i, (x, y, gamma) in enumerate(self.vortices):
            new_x = x + (u_conv[i] + u_d[i]) * dt
            new_y = y + (v_conv[i] + v_d[i]) * dt
            self.vortices[i] = [new_x, new_y, gamma]

        self.history.append(np.array(self.vortices))


class DiffusionVelocityMethod(DiffusionVortex2D):
    def __init__(self, viscosity=1.0, sigma=1.0, U_inf=1.0, alpha=0.0):
        super().__init__(viscosity, sigma)
        self.U_inf = U_inf
        self.alpha = alpha

    def set_freestream(self, U_inf, alpha_degrees=0.0):
        self.U_inf = U_inf
        self.alpha = np.radians(alpha_degrees)

    def _freestream_velocity(self, x, y):
        u = np.ones_like(x) * self.U_inf * np.cos(self.alpha)
        v = np.ones_like(y) * self.U_inf * np.sin(self.alpha)
        return u, v

    def _vortex_induced_velocity(self, x, y, x_v, y_v, gamma_v):
        dx = x - x_v
        dy = y - y_v
        r_sq = dx ** 2 + dy ** 2

        mask = r_sq > 1e-12
        u_ind = np.zeros_like(x)
        v_ind = np.zeros_like(y)

        u_ind[mask] = (-gamma_v / (2 * np.pi)) * (dy[mask] / r_sq[mask])
        v_ind[mask] = (gamma_v / (2 * np.pi)) * (dx[mask] / r_sq[mask])

        return u_ind, v_ind

    def calculate_convection_velocity(self, x, y):
        u_conv, v_conv = self._freestream_velocity(x, y)

        for x_v, y_v, gamma_v in self.vortices:
            u_ind, v_ind = self._vortex_induced_velocity(x, y, x_v, y_v, gamma_v)
            u_conv += u_ind
            v_conv += v_ind

        return u_conv, v_conv

    def step(self, dt):
        if not self.vortices:
            return

        vortex_positions = np.array([[v[0], v[1]] for v in self.vortices])
        x_v = vortex_positions[:, 0]
        y_v = vortex_positions[:, 1]

        u_conv, v_conv = self.calculate_convection_velocity(x_v, y_v)

        super().step(dt, u_conv, v_conv)

    def total_velocity_field(self, X, Y):
        u_conv, v_conv = self._freestream_velocity(X, Y)

        for x_v, y_v, gamma_v in self.vortices:
            u_ind, v_ind = self._vortex_induced_velocity(X, Y, x_v, y_v, gamma_v)
            u_conv += u_ind
            v_conv += v_ind

        return u_conv, v_conv