"""Plane-rendering tensors used by PGSR supervision."""

from typing import TypedDict

from torch import Tensor


class PlaneMaps(TypedDict):
    """One camera: scalar maps (1,H,W,1), camera-space normals (1,H,W,3)."""

    depths: Tensor
    normals: Tensor
    render_distances: Tensor
    depth_normals: Tensor
    depth_valid: Tensor
    depth_normal_valid: Tensor


class Reprojection(TypedDict):
    """Dense reference-to-neighbor consistency; flattened masks have H*W rows."""

    pixels: Tensor
    weights: Tensor
    valid: Tensor
    ncc_valid: Tensor
    loss: Tensor
