import numpy as np
import matplotlib.pyplot as plt


class UniformFlow:
    def __init__(self, u_inf=1.0, alpha=0.0):
        self.u_inf = u_inf  # U.F. velocity
        self.alpha = np.radians(alpha)  # U.F. angle of attack

    def velocity(self, x, y):
        """Returns U.F. velocity components (u, v) in (x, y)"""
        u = self.u_inf * np.cos(self.alpha)
        v = self.u_inf * np.sin(self.alpha)
        return u, v

    def stream_function(self, x, y):
        """Returns U.F. stream function"""
        return self.u_inf * (y * np.cos(self.alpha) - x * np.sin(self.alpha))


class Source:
    def __init__(self, strength, x, y):
        self.strength = strength  # S. power
        self.x, self.y = x, y  # S. position

    def velocity(self, x, y):
        """Returns S. velocity in (x, y)"""
        dx, dy = x - self.x, y - self.y
        r = np.sqrt(dx ** 2 + dy ** 2)

        if r == 0:
            return 0, 0

        u = (self.strength / (2 * np.pi)) * (dx / r ** 2)
        v = (self.strength / (2 * np.pi)) * (dy / r ** 2)
        return u, v

    def stream_function(self, x, y):
        """Returns S. stream function"""
        dx, dy = x - self.x, y - self.y
        return (self.strength / (2 * np.pi)) * np.arctan2(dy, dx)


# mesh
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# flows
uniform = UniformFlow(u_inf=1.0, alpha=0)
source = Source(strength=2.0, x=0, y=0)

# total stream function
psi_total = uniform.stream_function(X, Y) + source.stream_function(X, Y)

# visualization
plt.figure(figsize=(10, 8))
plt.contour(X, Y, psi_total, levels=50, colors='blue')  # streamlines
plt.xlabel('X')
plt.ylabel('Y')
plt.title('U.F. + S.')
plt.grid(True)
plt.axis('equal')
plt.show()