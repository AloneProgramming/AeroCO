import numpy as np
import matplotlib.pyplot as plt


class UniformFlow:
    def __init__(self, u_inf=1.0, alpha=0.0):
        self.u_inf = u_inf
        self.alpha = np.radians(alpha)

    def velocity(self, x, y):
        u = self.u_inf * np.cos(self.alpha)
        v = self.u_inf * np.sin(self.alpha)
        return u, v

    def stream_function(self, x, y):
        return self.u_inf * (y * np.cos(self.alpha) - x * np.sin(self.alpha))


class Source:
    def __init__(self, strength, x, y):
        self.strength = strength
        self.x, self.y = x, y

    def velocity(self, x, y):
        dx, dy = x - self.x, y - self.y
        r = np.sqrt(dx ** 2 + dy ** 2)

        u = np.where(r == 0, 0, (self.strength / (2 * np.pi)) * (dx / r ** 2))
        v = np.where(r == 0, 0, (self.strength / (2 * np.pi)) * (dy / r ** 2))
        return u, v

    def stream_function(self, x, y):
        dx, dy = x - self.x, y - self.y
        return (self.strength / (2 * np.pi)) * np.arctan2(dy, dx)


class Doublet:
    def __init__(self, strength, x, y):
        self.strength = strength
        self.x, self.y = x, y

    def velocity(self, x, y):
        dx, dy = x - self.x, y - self.y
        r_sq = dx ** 2 + dy ** 2

        u = np.where(r_sq == 0, 0, (-self.strength / (2 * np.pi)) * (dx ** 2 - dy ** 2) / r_sq ** 2)
        v = np.where(r_sq == 0, 0, (-self.strength / (2 * np.pi)) * (2 * dx * dy) / r_sq ** 2)
        return u, v

    def stream_function(self, x, y):
        dx, dy = x - self.x, y - self.y
        r_sq = dx ** 2 + dy ** 2

        psi = np.where(r_sq == 0, 0, (-self.strength / (2 * np.pi)) * (dy / r_sq))
        return psi


U_inf = 1.0
R = 1.0
kappa = 2 * np.pi * U_inf * R ** 2

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# flows
uniform = UniformFlow(u_inf=U_inf, alpha=0)
doublet = Doublet(strength=kappa, x=0, y=0)

# total stream function
psi_total = uniform.stream_function(X, Y) + doublet.stream_function(X, Y)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.contour(X, Y, psi_total, levels=50, colors='blue', linewidths=0.8)
circle = plt.Circle((0, 0), R, color='red', fill=False, linewidth=2)
plt.gca().add_patch(circle)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Flow over cylinder')
plt.grid(True)
plt.axis('equal')
plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.subplot(1, 2, 2)
theta = np.linspace(0, 2 * np.pi, 100)
V_surface = 2 * U_inf * np.sin(theta)
Cp = 1 - (V_surface / U_inf) ** 2

plt.plot(np.degrees(theta), Cp)
plt.xlabel('Angle θ (degrees)')
plt.ylabel('Cp')
plt.title('Pressure distribution')
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()