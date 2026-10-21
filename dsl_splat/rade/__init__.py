"""RaDe-GS on a projected-Gaussian (3DGS-style) Warp backend."""

from .api import rasterize, RasterizationResult
from .filtering import apply_3d_filter
from . import ops

__all__ = ["rasterize", "RasterizationResult", "apply_3d_filter", "ops"]
