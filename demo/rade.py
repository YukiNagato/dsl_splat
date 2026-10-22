"""RaDe-specific filtering, normals and strategy policy for the training demo."""

import math
from typing import Any, Sequence

import torch
from torch import Tensor

from dsl_splat.pgsr.geometry import (
    depth_to_normal,
    integer_center_intrinsics,
    pixel_rays,
    valid_depth,
)
from dsl_splat.strategy.default import DefaultStrategy
from dsl_splat.strategy import ImprovedGSStrategy
from dsl_splat.strategy.ops import reset_opa

if __package__:
    from .colmap_dataset import ImageFrame
else:
    from colmap_dataset import ImageFrame


@torch.no_grad()
def compute_3d_filter(
    means: Tensor, frames: Sequence[ImageFrame], near_plane: float = 0.2
) -> Tensor:
    """RaDe's sqrt(0.2)*min_visible_z/max_fx filter width, per Gaussian.

    Bounds extend the image by 7.5% on each side (the centered-camera 0.575
    rule), retaining arbitrary principal points. Unseen points use the largest
    visible depth. If all points are unseen, use the largest positive camera
    depth, or near_plane if all are behind the cameras. Empty inputs are valid.
    """
    if not frames:
        raise ValueError("3D filtering requires at least one training camera")
    if not len(means):
        return means.new_empty((0, 1))
    distance = means.new_full((len(means),), math.inf)
    fallback = means.new_tensor(near_plane)
    max_focal = max(frame.fx for frame in frames)
    for frame in frames:
        view = frame.w2c.to(means)
        points = means @ view[:3, :3].T + view[:3, 3]
        z = points[:, 2]
        valid = torch.isfinite(points).all(-1) & (z > near_plane)
        safe = torch.where(valid[:, None], points, 0.0)
        uv = safe[:, :2] / torch.where(valid, z, 1.0)[:, None]
        u, v = uv[:, 0] * frame.fx + frame.cx, uv[:, 1] * frame.fy + frame.cy
        inside = (
            valid
            & (u >= -0.075 * frame.width)
            & (u <= 1.075 * frame.width)
            & (v >= -0.075 * frame.height)
            & (v <= 1.075 * frame.height)
        )
        distance = torch.where(inside, torch.minimum(distance, z), distance)
        fallback = torch.maximum(fallback, torch.where(valid, z, 0.0).amax())
    visible = torch.isfinite(distance)
    farthest = torch.where(visible, distance, 0.0).amax()
    fallback = torch.where(visible.any(), farthest, fallback)
    distance = torch.where(visible, distance, fallback)
    return (distance * (math.sqrt(0.2) / max_focal))[:, None]


class _RaDeFilter:
    """Filtering shared by the default and Improved-GS population policies."""

    frames: Sequence[ImageFrame]
    filter_enabled: bool
    near_plane: float
    reset_opacity: float

    @torch.no_grad()
    def refresh_filter(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        state: dict[str, Any],
    ) -> None:
        state["filter_3d"] = (
            compute_3d_filter(params["means"], self.frames, self.near_plane)
            if self.filter_enabled
            else params["means"].new_zeros((len(params["means"]), 1))
        )

    def _reset_opacities(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Any],
        info: dict[str, Any],
    ) -> None:
        reset_opa(
            params,
            optimizers,
            state,
            self.reset_opacity,
            opacity_filter_variance=(
                state["filter_3d"].square() if self.filter_enabled else None
            ),
        )


class RaDeStrategy(_RaDeFilter, DefaultStrategy):
    """Keep filter rows aligned through growth/pruning and reset filtered opacity."""

    def __init__(
        self,
        frames: Sequence[ImageFrame],
        *,
        filter_3d: bool = True,
        near_plane: float = 0.2,
        reset_opacity: float = 0.01,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.frames, self.filter_enabled = frames, filter_3d
        self.near_plane, self.reset_opacity = near_plane, reset_opacity

    def _prune_gs(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Any],
        step: int,
    ) -> int:
        removed = super()._prune_gs(params, optimizers, state, step)
        # Called after growth; recompute even when a split leaves N unchanged.
        self.refresh_filter(params, state)
        return removed


class RaDeImprovedGSStrategy(_RaDeFilter, ImprovedGSStrategy):
    """Apply LAS/RAP while preserving RaDe's camera-dependent filter/reset."""

    def __init__(
        self,
        frames: Sequence[ImageFrame],
        *,
        filter_3d: bool = True,
        near_plane: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.frames, self.filter_enabled = frames, filter_3d
        self.near_plane = near_plane

    def _grow_with_scores(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Any],
        step: int,
        scores: Tensor | None,
    ) -> int:
        count = super()._grow_with_scores(params, optimizers, state, step, scores)
        if count:
            self.refresh_filter(params, state)
        return count


def geometry_maps(
    expected: Tensor,
    median: Tensor,
    normals: Tensor,
    alpha: Tensor,
    K: Tensor,
    depth_ratio: float = 0.0,
) -> dict[str, Tensor]:
    """Direct RaDe depth/normal supervision; no PGSR plane or alpha weighting."""
    expected, median, alpha = expected[0, ..., 0], median[0, ..., 0], alpha[0, ..., 0]
    normal = normals[0]
    finite = torch.isfinite(normal).all(-1) & torch.isfinite(alpha) & (alpha > 1e-6)
    safe_normal = torch.where(finite[..., None], normal, 0.0)
    expected_valid = valid_depth(expected) & finite
    _, rays = pixel_rays(integer_center_intrinsics(K), *expected.shape)
    depth_normal = torch.zeros_like(safe_normal)
    normal_valid = torch.zeros_like(expected_valid)
    loss = safe_normal.reshape(-1)[:0].sum()
    for weight, depth in ((1 - depth_ratio, expected), (depth_ratio, median)):
        if weight > 0:
            valid = valid_depth(depth) & finite
            dn, mask = depth_to_normal(depth, rays, valid)
            error = 1 - (safe_normal * dn).sum(-1)
            loss = loss + weight * torch.where(mask, error, 0.0).mean()
            depth_normal = depth_normal + weight * dn
            normal_valid = normal_valid | mask
    return dict(
        depths=torch.where(expected_valid, expected, 0.0)[None, ..., None],
        normals=safe_normal[None],
        depth_valid=expected_valid[None, ..., None],
        depth_normals=depth_normal[None],
        depth_normal_valid=normal_valid[None, ..., None],
        normal_loss=loss,
    )
