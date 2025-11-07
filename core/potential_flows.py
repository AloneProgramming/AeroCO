import numpy as np
import matplotlib.pyplot as plt


class FlowComponent:
    def __init__(self, strength, dx=0, dy=0):
        self.strength = strength
        self.dx = dx
        self.dy = dy

    def velocity_field(self, X, Y):
        return np.zeros_like(X), np.zeros_like(Y)

    def stream_function(self, X, Y):
        return np.zeros_like(X)

    def _shifted_coordinates(self, X, Y):
        return X - self.dx, Y - self.dy


class UniformFlow(FlowComponent):
    def __init__(self, strength, alpha=0, dx=0, dy=0):
        super().__init__(strength, dx, dy)
        self.strength = strength
        self.alpha = alpha

    def velocity_field(self, X, Y):
        u = self.strength * np.cos(self.alpha)
        v = self.strength * np.sin(self.alpha)
        return u * np.ones_like(X), v * np.ones_like(Y)

    def stream_function(self, X, Y):
        return self.strength * (Y * np.cos(self.alpha) - X * np.sin(self.alpha))


class SourceSink(FlowComponent):
    def velocity_field(self, X, Y):
        X_shifted, Y_shifted = self._shifted_coordinates(X, Y)
        r_sq = np.maximum(X_shifted ** 2 + Y_shifted ** 2, 1e-10)
        u = (self.strength / (2 * np.pi)) * X_shifted / r_sq
        v = (self.strength / (2 * np.pi)) * Y_shifted / r_sq
        return u, v

    def stream_function(self, X, Y):
        X_shifted, Y_shifted = self._shifted_coordinates(X, Y)
        return (self.strength / (2 * np.pi)) * np.arctan2(Y_shifted, X_shifted)


class Doublet(FlowComponent):
    def velocity_field(self, X, Y):
        X_shifted, Y_shifted = self._shifted_coordinates(X, Y)
        r_sq = np.maximum(X_shifted ** 2 + Y_shifted ** 2, 1e-10)
        factor = self.strength / r_sq ** 2
        u = factor * (X_shifted ** 2 - Y_shifted ** 2)
        v = factor * (2 * X_shifted * Y_shifted)
        return u, v

    def stream_function(self, X, Y):
        X_shifted, Y_shifted = self._shifted_coordinates(X, Y)
        r_sq = np.maximum(X_shifted ** 2 + Y_shifted ** 2, 1e-10)
        return (-self.strength * Y_shifted) / (2 * np.pi * r_sq)


class Vortex(FlowComponent):
    def velocity_field(self, X, Y):
        X_shifted, Y_shifted = self._shifted_coordinates(X, Y)
        r_sq = np.maximum(X_shifted ** 2 + Y_shifted ** 2, 1e-10)
        u = (-self.strength / (2 * np.pi)) * Y_shifted / r_sq
        v = (self.strength / (2 * np.pi)) * X_shifted / r_sq
        return u, v

    def stream_function(self, X, Y):
        X_shifted, Y_shifted = self._shifted_coordinates(X, Y)
        r_sq = np.maximum(X_shifted ** 2 + Y_shifted ** 2, 1e-10)
        return (-self.strength / (2 * np.pi)) * np.log(np.sqrt(r_sq))


class GaussianVortex(Vortex):
    def __init__(self, strength, sigma=1.0, dx=0, dy=0):
        super().__init__(strength, dx, dy)
        self.sigma = sigma

    def velocity_field(self, X, Y):
        X_shifted, Y_shifted = self._shifted_coordinates(X, Y)
        r_sq = np.maximum(X_shifted ** 2 + Y_shifted ** 2, 1e-10)
        r = np.sqrt(r_sq)
        velocity_magnitude = (self.strength / (2 * np.pi * r)) * (1 - np.exp(-r_sq / self.sigma ** 2))
        u = -velocity_magnitude * (Y_shifted / r)
        v = velocity_magnitude * (X_shifted / r)
        return u, v

    def vorticity_field(self, X, Y):
        X_shifted, Y_shifted = self._shifted_coordinates(X, Y)
        r_sq = X_shifted ** 2 + Y_shifted ** 2
        vorticity = (self.strength / (np.pi * self.sigma ** 2)) * np.exp(-r_sq / self.sigma ** 2)
        return vorticity


class FlowModel:
    def __init__(self, components=None):
        self.components = components if components is not None else []

    def add_component(self, component):
        self.components.append(component)

    def velocity_field(self, X, Y):
        u_total = np.zeros_like(X)
        v_total = np.zeros_like(Y)

        for comp in self.components:
            u, v = comp.velocity_field(X, Y)
            u_total += u
            v_total += v

        return u_total, v_total

    def stream_function(self, X, Y):
        psi_total = np.zeros_like(X)

        for comp in self.components:
            psi_total += comp.stream_function(X, Y)

        return psi_total

    def pressure_coefficient(self, X, Y, U_inf=1.0):
        u, v = self.velocity_field(X, Y)
        V_sq = u ** 2 + v ** 2
        Cp = 1.0 - V_sq / (U_inf ** 2)
        return Cp