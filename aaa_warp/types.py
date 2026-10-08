"""Tensor contracts shared by the Python stages.

Shapes are documented here; TypedDict does not allocate wrappers or validate
GPU tensors at runtime. The legacy ``rgb`` field means arbitrary features.
"""

from __future__ import annotations

from typing import Protocol, TypedDict

from torch import Tensor

from .dispatch import FrameBindings
from .settings import GaussianRasterizationSettings


class GaussianInputs(TypedDict, total=False):
    """Activated inputs for one view; optional tensors may be None."""

    means3D: Tensor  # (N,3)
    means2D: Tensor | None  # auxiliary screen gradients
    opacities: Tensor  # (N,) or (N,1)
    filter3D: Tensor | None
    shs: Tensor | None  # (N,K,3)
    colors_precomp: Tensor | None  # (N,C)
    scales: Tensor | None  # (N,3)
    rotations: Tensor | None  # (N,4), wxyz
    cov3D_precomp: Tensor | None


class _Geometry(TypedDict):
    means2D: Tensor  # (N,2), bounding center
    rects2D: Tensor  # (N,2), bounding half extents
    depths: Tensor  # (N,)
    radii: Tensor  # int32 (N,); zero for rejected rows
    tiles_touched: Tensor  # int32 (N,)
    valid: Tensor  # bool (N,)
    rgb: Tensor  # (N,C)


class PreprocessedGaussians(_Geometry, total=False):
    """Only valid rows define geometry/features; culling sentinels always exist."""

    gauss2screen: Tensor  # 3D: (N,4,4)
    opacity: Tensor  # 3D: (N,)
    clamped: Tensor  # SH: bool (N,3)
    cov3D: Tensor  # 2D: (N,6)
    conic_opacity: Tensor  # 2D: (N,4)
    cov3D_inv: Tensor


class _TileRanges(TypedDict):
    point_list: Tensor  # int32 (pairs,)
    ranges: Tensor  # int32 (tiles,2)


class TileBins(_TileRanges, total=False):
    point_offsets: Tensor
    point_list_keys_unsorted: Tensor
    point_list_unsorted: Tensor
    point_list_keys: Tensor


class RenderOutput(TypedDict):
    color: Tensor  # (C,H,W)
    final_T: Tensor  # (H,W)
    contributors: Tensor  # int32 (H,W)


class RasterizationOutput(RenderOutput):
    radii: Tensor


class _RenderGradients(TypedDict):
    gauss2screen: Tensor
    rgb: Tensor
    opacity: Tensor


class RenderGradients(_RenderGradients, total=False):
    means2D: Tensor  # optional signed pixel-space gradients
    means2D_abs: Tensor  # optional sum of absolute per-pixel gradients


class GaussianGradients(TypedDict):
    means3D: Tensor
    means2D: Tensor
    scales: Tensor
    rotations: Tensor
    shs: Tensor
    colors_precomp: Tensor
    opacities: Tensor


class Renderer(Protocol):
    def __call__(
        self,
        preprocessed: PreprocessedGaussians,
        bins: TileBins,
        raster_settings: GaussianRasterizationSettings,
        *,
        output: RenderOutput | None = None,
        _bindings: FrameBindings | None = None,
    ) -> RenderOutput: ...
