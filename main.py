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

# panels
x_control = np.zeros(N)
y_control = np.zeros(N)
theta_panel = np.zeros(N)
length = np.zeros(N)
nx = np.zeros(N)
ny = np.zeros(N)

for i in range(N):
    x_start, y_start = x[i], y[i]
    x_end, y_end = x[i+1], y[i+1]

    length[i] = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)

    theta_panel[i] = np.arctan2(y_end - y_start, x_end - x_start)

    nx[i] = -np.sin(theta_panel[i])
    ny[i] = np.cos(theta_panel[i])

    x_control[i] = (x_start + x_end) / 2
    y_control[i] = (y_start + y_end) / 2

print("Panels have been calculated")
print(f"Mean panel length is: {np.mean(length):.4f}")

# visualization
plt.figure(figsize=(10, 8))
plt.plot(x, y, 'b-', linewidth=2, label='Airfoil')
plt.axis('equal')
plt.grid(True)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Test circle visualization')
plt.legend()
plt.show()