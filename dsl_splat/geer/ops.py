"""GEER stages for callers who explicitly compose or inspect the pipeline.

prepare_camera -> prepare_gaussians -> bound_gaussians -> bin_and_sort -> render

All stages use the caller's Torch stream. Geometry retains first-order gradients;
bounds and sorting are discrete. No persistent workspace or explicit bindings
object is required. See geer/README.md for the composition example.
"""

from .binning import ParticleBounds, bin_and_sort, bound_gaussians
from .cameras import prepare_camera
from .geometry import prepare_gaussians
from .render import render

__all__ = [
    "prepare_camera",
    "prepare_gaussians",
    "bound_gaussians",
    "bin_and_sort",
    "render",
    "ParticleBounds",
]
