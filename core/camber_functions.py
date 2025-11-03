import numpy as np


def parabolic_camber(h=0.05):
    def camber_func(x):
        return 4 * h * x * (1 - x)

    camber_func.h = h
    camber_func.type = "parabolic"
    return camber_func


def naca_4_digit_camber(code="0012", chord=1.0):
    m = float(code[0]) / 100.0
    p = float(code[1]) / 10.0

    def camber_func(x):
        x_norm = x / chord
        yc = np.zeros_like(x_norm)

        mask_front = x_norm <= p
        if p > 0:
            yc[mask_front] = (m / p ** 2) * (2 * p * x_norm[mask_front] - x_norm[mask_front] ** 2)

        mask_back = x_norm > p
        if p < 1:
            yc[mask_back] = (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x_norm[mask_back] - x_norm[mask_back] ** 2)

        return yc * chord

    camber_func.m = m
    camber_func.p = p
    camber_func.code = code
    camber_func.type = "naca_4_digit"

    return camber_func


def symmetric_airfoil():
    def camber_func(x):
        return np.zeros_like(x)

    camber_func.type = "symmetric"
    return camber_func