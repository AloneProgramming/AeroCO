import numpy as np
import scipy
import matplotlib.pyplot as plt

class Vortex1D:
    def __init__(self, x, strength):
        self.x = x
        self.strength = strength

def vorticity_at_point(x, vortices, sigma):
    total = 0.0
    for vortex in vortices:
        r_sq = (x - vortex.x) ** 2
        total += vortex.strength * np.exp(-r_sq / sigma ** 2)
    return total / (np.sqrt(np.pi) * sigma)

def diffusion_velocity(vortex_index, vortices, nu, sigma):
    vortex_i = vortices[vortex_index]
    x_i = vortex_i.x

    omega_i = vorticity_at_point(x_i, vortices, sigma)

    if abs(omega_i) < 1e-12:
        return 0.0

    total = 0.0
    for j, vortex_j in enumerate(vortices):
        if j == vortex_index:
            continue
        dx = x_i - vortex_j.x
        r_sq = dx ** 2
        total += dx * vortex_j.strength * np.exp(-r_sq / sigma ** 2)

    factor = 2 * nu / (np.sqrt(np.pi) * sigma ** 2 * omega_i)
    return factor * total
