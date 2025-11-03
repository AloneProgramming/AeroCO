import numpy as np
import matplotlib.pyplot as plt

def plot_velocity_field(flow_model, xlim=(-5, 5), ylim=(-5, 5), resolution=50):
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(11, 10))

    u, v = flow_model.velocity_field(X, Y)

    psi = flow_model.stream_function(X, Y)
    plt.contour(X, Y, psi, levels=20, colors='black', linewidths=0.8)

    speed = np.sqrt(u ** 2 + v ** 2)
    plt.contourf(X, Y, speed, levels=20, alpha=0.5, cmap='viridis')
    plt.colorbar(label='Speed')

    plt.title('Velocity Field Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal')
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_pressure_field(flow_model, xlim=(-5, 5), ylim=(-5, 5), resolution=200, U_inf=1.0):
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)

    Cp = flow_model.pressure_coefficient(X, Y, U_inf)

    plt.figure(figsize=(11, 10))

    contour = plt.contourf(X, Y, Cp, levels=50, cmap='coolwarm', vmin=-10.0)
    plt.colorbar(contour, label='Pressure Coefficient')

    plt.contour(X, Y, Cp, levels=20, colors='black', linewidths=0.5, alpha=0.5)

    psi = flow_model.stream_function(X, Y)
    plt.contour(X, Y, psi, levels=20, colors='white', linewidths=0.8, alpha=0.3)

    plt.title('Pressure Coefficient Field')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal')
    plt.show()


def plot_streamlines(flow_model, xlim=(-5, 5), ylim=(-5, 5), resolution=100):
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)

    psi = flow_model.stream_function(X, Y)

    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, psi, levels=30, colors='blue', linewidths=1)
    plt.title('Streamlines')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal')
    plt.grid(True, alpha=0.3)
    plt.show()