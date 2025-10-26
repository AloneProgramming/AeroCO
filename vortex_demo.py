import numpy as np
import matplotlib.pyplot as plt
from potential_flows import *


def demo_single_vortex():
    print("Single Vortex Demo")

    flow = FlowModel()
    flow.add_component(Vortex(strength=1.0, dx=0, dy=0))

    flow.plot_velocity_field(xlim=(-3, 3), ylim=(-3, 3), resolution=30)
    flow.plot_pressure_field(xlim=(-3, 3), ylim=(-3, 3), U_inf=1.0)


def demo_vortex_pair():
    print("Vortex Pair Demo")

    flow = FlowModel()
    flow.add_component(Vortex(strength=10.0, dx=-1, dy=0))
    flow.add_component(Vortex(strength=-10.0, dx=1, dy=0))

    flow.plot_velocity_field(xlim=(-3, 3), ylim=(-3, 3), resolution=30)
    flow.plot_pressure_field(xlim=(-3, 3), ylim=(-3, 3), U_inf=1.0)


def demo_cylinder_flow():
    print("Cylinder Flow Demo")

    flow = FlowModel()
    flow.add_component(UniformFlow(strength=5.0, alpha=0))

    R = 2.0
    strength_doublet = 2 * np.pi * 5.0 * R ** 2
    flow.add_component(Doublet(strength=strength_doublet, dx=0, dy=0))

    flow.plot_velocity_field(xlim=(-5, 5), ylim=(-5, 5), resolution=40)
    flow.plot_pressure_field(xlim=(-5, 5), ylim=(-5, 5), U_inf=5.0)


def demo_cylinder_with_circulation():
    print("Cylinder with Circulation Demo")

    flow = FlowModel()
    flow.add_component(UniformFlow(strength=5.0, alpha=0))

    R = 2.0
    strength_doublet = 2 * np.pi * 5.0 * R ** 2
    flow.add_component(Doublet(strength=strength_doublet, dx=0, dy=0))
    flow.add_component(Vortex(strength=15.0, dx=0, dy=0))

    flow.plot_velocity_field(xlim=(-5, 5), ylim=(-5, 5), resolution=40)
    flow.plot_pressure_field(xlim=(-5, 5), ylim=(-5, 5), U_inf=5.0)

if __name__ == "__main__":
    #demo_single_vortex()
    #demo_vortex_pair()
    #demo_cylinder_flow()
    demo_cylinder_with_circulation()