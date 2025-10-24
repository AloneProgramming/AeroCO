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

if __name__ == "__main__":
    print("Visualisation moved to main.py.")