import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_velocity_field(flow_model, xlim=(-5, 5), ylim=(-5, 5), resolution=50, airfoil_coords=None):
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)

    velocity_cmap = plt.cm.plasma

    plt.figure(figsize=(12, 10))

    u, v = flow_model.velocity_field(X, Y)
    speed = np.sqrt(u ** 2 + v ** 2)

    contourf = plt.contourf(X, Y, speed, levels=30, alpha=0.8, cmap=velocity_cmap)
    plt.colorbar(contourf, label='Speed [m/s]', shrink=0.8)

    psi = flow_model.stream_function(X, Y)
    plt.contour(X, Y, psi, levels=25, colors='white', linewidths=0.8, alpha=0.6)

    if airfoil_coords is not None:
        plt.plot(airfoil_coords[:, 0], airfoil_coords[:, 1], 'k-', linewidth=2, label='Airfoil')
        plt.fill(airfoil_coords[:, 0], airfoil_coords[:, 1], 'gray', alpha=0.3)

    plt.title('Velocity Field with Streamlines', fontsize=14, fontweight='bold')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.gca().set_aspect('equal')
    plt.grid(True, alpha=0.2)
    if airfoil_coords is not None:
        plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pressure_field(flow_model, xlim=(-5, 5), ylim=(-5, 5), resolution=200, U_inf=1.0, airfoil_coords=None):
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)

    pressure_cmap = plt.cm.seismic

    Cp = flow_model.pressure_coefficient(X, Y, U_inf)

    Cp_clipped = np.clip(Cp, -2.0, 1.5)

    plt.figure(figsize=(12, 10))

    contourf = plt.contourf(X, Y, Cp_clipped, levels=50, cmap=pressure_cmap, alpha=0.9)
    cbar = plt.colorbar(contourf, label='Pressure Coefficient (Cp)', shrink=0.8)

    contour = plt.contour(X, Y, Cp, levels=15, colors='black', linewidths=0.5, alpha=0.5)
    plt.clabel(contour, inline=True, fontsize=8, fmt='%.1f')

    if airfoil_coords is not None:
        plt.plot(airfoil_coords[:, 0], airfoil_coords[:, 1], 'k-', linewidth=2)
        plt.fill(airfoil_coords[:, 0], airfoil_coords[:, 1], 'gray', alpha=0.5)

    plt.title('Pressure Coefficient Field', fontsize=14, fontweight='bold')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()


def plot_combined_flow(flow_model, xlim=(-5, 5), ylim=(-5, 5), resolution=100, U_inf=1.0, airfoil_coords=None):
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    u, v = flow_model.velocity_field(X, Y)
    speed = np.sqrt(u ** 2 + v ** 2)
    im1 = ax1.contourf(X, Y, speed, levels=25, alpha=0.9, cmap='plasma')
    plt.colorbar(im1, ax=ax1, label='Speed [m/s]', shrink=0.8)

    psi = flow_model.stream_function(X, Y)
    ax1.contour(X, Y, psi, levels=20, colors='white', linewidths=0.7, alpha=0.6)

    Cp = flow_model.pressure_coefficient(X, Y, U_inf)
    Cp_clipped = np.clip(Cp, -2.0, 1.5)
    im2 = ax2.contourf(X, Y, Cp_clipped, levels=40, alpha=0.9, cmap='seismic')
    plt.colorbar(im2, ax=ax2, label='Pressure Coefficient (Cp)', shrink=0.8)

    for ax in (ax1, ax2):
        if airfoil_coords is not None:
            ax.plot(airfoil_coords[:, 0], airfoil_coords[:, 1], 'k-', linewidth=2)
            ax.fill(airfoil_coords[:, 0], airfoil_coords[:, 1], 'gray', alpha=0.5)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    ax1.set_title('Velocity Field', fontweight='bold')
    ax2.set_title('Pressure Field', fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_streamlines_advanced(flow_model, xlim=(-5, 5), ylim=(-5, 5), resolution=150, airfoil_coords=None):
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)

    u, v = flow_model.velocity_field(X, Y)
    speed = np.sqrt(u ** 2 + v ** 2)

    plt.figure(figsize=(12, 10))

    plt.contourf(X, Y, speed, levels=30, alpha=0.4, cmap='viridis')

    psi = flow_model.stream_function(X, Y)
    stream = plt.streamplot(X, Y, u, v, color=speed, cmap='plasma',
                            linewidth=1.5, density=2, arrowsize=1.2)
    plt.colorbar(stream.lines, label='Speed [m/s]', shrink=0.8)

    if airfoil_coords is not None:
        plt.plot(airfoil_coords[:, 0], airfoil_coords[:, 1], 'k-', linewidth=3)
        plt.fill(airfoil_coords[:, 0], airfoil_coords[:, 1], 'darkgray', alpha=0.8)

    plt.title('Streamlines Colored by Velocity', fontsize=14, fontweight='bold')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.gca().set_aspect('equal')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()