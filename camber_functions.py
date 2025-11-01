import numpy as np

def parabolic_camber(h=0.05):
    def camber_func(x):
        return 4 * h * x * (1-x)
    return camber_func