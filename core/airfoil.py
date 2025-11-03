import numpy as np
import matplotlib.pyplot as plt


class AirfoilAnalysis:
    def __init__(self, name="Airfoil"):
        self.name = name
        self.results = {}

    def validate_parameters(self, alpha, panels, reynolds=None):
        if abs(alpha) > 90:
            raise ValueError("Angle of attack must be between -90 and 90 degrees")
        if panels < 3:
            raise ValueError("Number of panels must be at least 3")
        if reynolds and reynolds <= 0:
            raise ValueError("Reynolds number must be positive")
        return True