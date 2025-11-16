from .thin_airfoil import ThinAirfoilTheory
from .discrete_vortex import DiscreteVortexMethod
from .diffusion_velocity import DiffusionVortex1D, DiffusionVortex2D

__all__ = ["ThinAirfoilTheory",
           "DiscreteVortexMethod",
           "DiffusionVortex1D", "DiffusionVortex2D"]
