"""AAA-Gaussians rendering, geometry stages and method-specific adapters."""

from .api import RasterizationResult, rasterize
from .config import default_settings
from .legacy import GaussianRasterizer
from . import ops
from .settings import ExtendedSettings

__all__ = [
    "rasterize",
    "RasterizationResult",
    "default_settings",
    "ExtendedSettings",
    "GaussianRasterizer",
    "ops",
]
