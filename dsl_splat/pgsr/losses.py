"""PGSR supervision built from tensor geometry and RaDe-GS patch NCC."""

# PGSR adaptations: Copyright 2023-2024, 3D Vision Group, Zhejiang University.
# See PGSR_LICENSE.txt and README.md for upstream terms and local changes.

import math
from typing import Callable

import torch
from torch import Tensor
import torch.nn.functional as F

from .geometry import multiview_geometry
from .patch_ncc import warp_patch_ncc
from .types import PlaneMaps


def flatten_loss(scales: Tensor) -> Tensor:
    """Penalize the smallest activated scale of caller-selected Gaussians."""
    if scales.shape[0] == 0:
        return scales.sum()
    return scales.amin(dim=-1).mean()


def get_img_grad_weight(image: Tensor) -> Tensor:
    """PGSR image-gradient weights for RGB HWC, including constant/tiny images."""
    height, width = image.shape[:2]
    if min(height, width) < 3:
        return image.new_ones((height, width))
    horizontal = (image[1:-1, 2:] - image[1:-1, :-2]).abs().mean(dim=-1)
    vertical = (image[:-2, 1:-1] - image[2:, 1:-1]).abs().mean(dim=-1)
    gradient = torch.maximum(horizontal, vertical)
    gradient = (gradient - gradient.min()) / (
        gradient.max() - gradient.min()
    ).clamp_min(1e-8)
    return F.pad(gradient, (1, 1, 1, 1), value=1.0)


def single_view_loss(image: Tensor, renders: PlaneMaps) -> Tensor:
    weights = (1 - get_img_grad_weight(image)).clamp(0, 1).detach().square()
    normal = renders["normals"][0]
    depth_normal = renders["depth_normals"][0]
    mask = renders["depth_normal_valid"][0, ..., 0]
    normal = torch.where(mask[..., None], normal, 0.0)
    depth_normal = torch.where(mask[..., None], depth_normal, 0.0)
    error = (depth_normal - normal).abs().sum(dim=-1)
    return torch.where(mask, weights * error, 0.0).mean()


class MultiViewLoss:
    """Geometric consistency and sampled patch NCC for two rendered views.

    Camera inputs are ordinary OpenCV w2c and K, with the same half-integer
    pixel-center convention as the renderers. Grayscale images are (H,W).
    K may remain on CPU, avoiding device-to-host reads for NCC's scalar camera
    arguments. Geometry is selected via geometry_fn, e.g. torch.compile of
    multiview_geometry; dynamic sampling and Warp dispatch remain in Python.
    """

    def __init__(
        self,
        pixel_noise_th: float = 1.0,
        sample_num: int = 102400,
        *,
        half_step: bool = False,
        geometry_fn: Callable = multiview_geometry,
    ):
        if pixel_noise_th <= 0 or not math.isfinite(pixel_noise_th) or sample_num <= 0:
            raise ValueError("pixel_noise_th and sample_num must be positive")
        self.pixel_noise_th = pixel_noise_th
        self.sample_num = sample_num
        self.half_step = half_step
        self.geometry_fn = geometry_fn

    def __call__(
        self,
        renders: PlaneMaps,
        near_renders: PlaneMaps,
        *,
        viewmat: Tensor,
        K: Tensor,
        near_viewmat: Tensor,
        near_K: Tensor,
        image: Tensor,
        near_image: Tensor,
        debug_info: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        height, width = image.shape
        depth = renders["depths"].reshape(height, width)
        neighbor_depth = near_renders["depths"].reshape(near_image.shape)
        normal = renders["normals"].reshape(height, width, 3)
        view, near = viewmat.to(depth), near_viewmat.to(depth)
        geometry = self.geometry_fn(
            depth,
            normal,
            neighbor_depth,
            view,
            K.to(depth),
            near,
            near_K.to(depth),
            renders.get("depth_valid"),
            near_renders.get("depth_valid"),
            self.pixel_noise_th,
        )
        if debug_info is not None:
            debug_info["geometry_weights"] = (
                geometry["weights"].reshape(height, width).detach()
            )
        valid_indices = geometry["ncc_valid"].nonzero(as_tuple=True)[0]
        if valid_indices.numel() > self.sample_num:
            selection = torch.randperm(valid_indices.numel(), device=depth.device)[
                : self.sample_num
            ]
            valid_indices = valid_indices[selection]
        zero = depth.reshape(-1)[:0].sum() + neighbor_depth.reshape(-1)[:0].sum()
        if valid_indices.numel() == 0:
            return zero, geometry["loss"]

        rotation = near[:3, :3] @ view[:3, :3].T
        translation = near[:3, 3] - rotation @ view[:3, 3]
        ref_k, near_k = K.detach().cpu().tolist(), near_K.detach().cpu().tolist()
        score, ncc_valid = warp_patch_ncc(
            depth.reshape(-1)[valid_indices],
            normal.reshape(-1, 3)[valid_indices],
            geometry["pixels"].reshape(-1, 2)[valid_indices],
            rotation.T.contiguous(),
            translation,
            image.to(depth),
            near_image.to(depth),
            ref_k[0][0],
            ref_k[1][1],
            ref_k[0][2] - 0.5,
            ref_k[1][2] - 0.5,
            near_k[0][0],
            near_k[1][1],
            near_k[0][2] - 0.5,
            near_k[1][2] - 0.5,
            half_step=self.half_step,
        )
        error = (1 - score).clamp(0, 2)
        mask = ncc_valid & torch.isfinite(score) & (error < 0.9)
        if not mask.any():
            return zero, geometry["loss"]
        ncc_loss = (error[mask] * geometry["weights"][valid_indices][mask]).mean()
        return ncc_loss, geometry["loss"]
