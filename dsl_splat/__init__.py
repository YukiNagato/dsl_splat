"""DSL Splat: Warp Gaussian renderers; top-level rendering exports select AAA.

Use dsl_splat.aaa or dsl_splat.gs3d for method-local APIs and adapters.
"""

from .aaa.preprocess import preprocess
from .aaa.binning import bin_and_sort
from .aaa.render import render_hierarchical_3d
from .aaa.legacy import GaussianRasterizer
from .aaa.gsplat import rasterization
from .aaa.api import RasterizationResult, rasterize
from .cameras import PinholeCamera
from .aaa.config import default_settings
from .aaa import ops
from .aaa.settings import (
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
