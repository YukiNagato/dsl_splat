"""Warp Gaussian renderers; legacy top-level rendering exports select AAA.

Use aaa_warp.aaa or aaa_warp.gs3d for method-local APIs and adapters.
"""

from .preprocess import preprocess
from .binning import bin_and_sort
from .render import render_hierarchical_3d
from .rasterizer import GaussianRasterizer
from .gsplat import rasterization
from .functional import RasterizationResult, rasterize
from .cameras import PinholeCamera
from .config import default_settings
from . import ops
from .settings import (
    GaussianRasterizationSettings,
    ExtendedSettings,
    CullingSettings,
    SortSettings,
    SortQueueSizes,
    SortMode,
    GlobalSortOrder,
)

__all__ = [
    "preprocess",
    "bin_and_sort",
    "render_hierarchical_3d",
    "GaussianRasterizer",
    "rasterization",
    "rasterize",
    "RasterizationResult",
    "PinholeCamera",
    "default_settings",
    "ops",
    "GaussianRasterizationSettings",
    "ExtendedSettings",
    "CullingSettings",
    "SortSettings",
    "SortQueueSizes",
    "SortMode",
    "GlobalSortOrder",
]
