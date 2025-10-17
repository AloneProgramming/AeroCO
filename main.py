import numpy as np
import matplotlib.pyplot as plt

# parameters
N = 64  # number of panels
R = 1.0  # test circle radius
V_inf = 1.0  # flow velocity
alpha_deg = 0.0  # AOA (deg)
alpha_rad = np.radians(alpha_deg)  # AOA (rad)

print("Initialization is complete.")
print(f"Number of panels: {N}, Test circle radius: {R}, AOA: {alpha_deg} deg.")

# test circle creating
theta = np.linspace(0, 2*np.pi, N+1)  # test circles angles dist
x = R * np.cos(theta)
y = R * np.sin(theta)

print(f"{len(x)} circle points were prepared.")
print(f"First point: ({x[0]:.3f}, {y[0]:.3f})")
print(f"Last point: ({x[-1]:.3f}, {y[-1]:.3f})")
