"""PGSR plane geometry and supervision for the 3DGS and AAA renderers.

Tensor functions are ordinary PyTorch and may be wrapped with torch.compile.
Patch NCC uses Warp; training, dataset loading and debug image I/O stay in demo.
"""

from .geometry import (
    plane_features,
    plane_maps,
    depth_to_normal,
    pixel_rays,
    integer_center_intrinsics,
    project_points,
    sample_depth,
    multiview_geometry,
)
from .losses import MultiViewLoss, flatten_loss, single_view_loss, get_img_grad_weight
from .patch_ncc import warp_patch_ncc
from .types import PlaneMaps, Reprojection

__all__ = [
    "plane_features",
    "plane_maps",
    "depth_to_normal",
    "pixel_rays",
    "integer_center_intrinsics",
    "project_points",
    "sample_depth",
    "multiview_geometry",
    "MultiViewLoss",
    "flatten_loss",
    "single_view_loss",
    "get_img_grad_weight",
    "warp_patch_ncc",
    "PlaneMaps",
    "Reprojection",
]
