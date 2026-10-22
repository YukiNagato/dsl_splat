# Copyright (c) 2026 ImprovedGS authors and Zhejiang University.
# Adapted for DSL Splat: HWC edge maps, Warp LAS, generalized attributes and
# optimizer state. See IMPROVED_GS_LICENSE.txt and README.md for provenance.
"""Improved-GS geometry, edge images and per-view EAS score reduction."""

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import warp as wp

from ..runtime.interop import current_stream
from ._improved_gs_kernels import split_geometry
from .base import Params
from .ops import update_param_with_optimizer


@torch.no_grad()
def split_long_axis(
    params: Params,
    optimizers: dict[str, torch.optim.Optimizer],
    state: dict[str, Any],
    mask: Tensor,
    split_distance: float = 0.45,
    opacity_reduction: float = 0.6,
) -> None:
    """Replace selected rows by +/- longest-axis children, zeroing new moments.

    Survivor order is unchanged; positive children precede negative children.
    Extra attributes and their requires_grad/dtype are preserved. All tensor
    entries in ``state`` must be per-Gaussian, following DefaultStrategy's ops.
    """
    n_total = len(params["means"])
    if (
        mask.dtype != torch.bool
        or mask.shape != (n_total,)
        or mask.device != params["means"].device
    ):
        raise ValueError("mask must be a boolean vector with one entry per Gaussian")
    if not 0 < split_distance <= 0.5 or not 0 < opacity_reduction <= 1:
        raise ValueError(
            "require 0 < split_distance <= 0.5 and 0 < opacity_reduction <= 1"
        )
    selected, rest = torch.where(mask)[0], torch.where(~mask)[0]
    n = len(selected)
    if n == 0:
        return
    means = params["means"]
    for name, shape in (
        ("means", (n_total, 3)),
        ("quats", (n_total, 4)),
        ("scales", (n_total, 3)),
    ):
        if params[name].shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if params["opacities"].shape not in ((n_total,), (n_total, 1)):
        raise ValueError("opacities must have shape [N] or [N,1]")
    for name in ("means", "quats", "scales", "opacities"):
        if params[name].dtype != torch.float32 or params[name].device != means.device:
            raise ValueError("LAS geometry parameters must be float32 on one device")
    child_means = means.new_empty((2 * n, 3))
    child_scales = means.new_empty((2 * n, 3))
    child_logits = means.new_empty(2 * n)
    wp.init()
    wp.launch(
        split_geometry,
        dim=n,
        inputs=[
            *[
                wp.from_torch(t, requires_grad=False)
                for t in (
                    means,
                    params["quats"],
                    params["scales"],
                    params["opacities"].view(-1),
                    selected,
                )
            ],
            split_distance,
            opacity_reduction,
        ],
        outputs=[wp.from_torch(t) for t in (child_means, child_scales, child_logits)],
        device=wp.device_from_torch(means.device),
        stream=current_stream(means.device) if means.is_cuda else None,
    )
    geometry = dict(means=child_means, scales=child_scales, opacities=child_logits)

    def param_fn(name: str, param: Tensor) -> nn.Parameter:
        children = (
            geometry[name].reshape(2 * n, *param.shape[1:])
            if name in geometry
            else param[selected].repeat((2,) + (1,) * (param.ndim - 1))
        )
        return nn.Parameter(
            torch.cat((param[rest], children)), requires_grad=param.requires_grad
        )

    def optimizer_fn(key: str, value: Tensor) -> Tensor:
        return torch.cat((value[rest], value.new_zeros((2 * n, *value.shape[1:]))))

    update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    for key, value in state.items():
        if isinstance(value, Tensor):
            state[key] = torch.cat(
                (value[rest], value[selected].repeat((2,) + (1,) * (value.ndim - 1)))
            )


@torch.no_grad()
def compute_edge_map(image: Tensor) -> Tensor:
    """Pillow-style FIND_EDGES on RGB HWC [0,1], without a Pillow dependency.

    Matches the paper-era preprocessing: quantize RGB, convert to 8-bit L,
    clamp the 8-neighbor Laplacian and retain the unfiltered image border.
    """
    if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("image must be nonempty RGB HWC")
    rgb = (image.detach().clamp(0, 1) * 255).to(torch.int32)
    gray = ((rgb * rgb.new_tensor([19595, 38470, 7471])).sum(-1) + 32768) // 65536
    edge = gray.float()
    if min(image.shape[:2]) >= 3:
        kernel = edge.new_tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        edge[1:-1, 1:-1] = F.conv2d(edge[None, None], kernel[None, None])[0, 0].clamp(
            0, 255
        )
    return edge / 255.0


class EdgeScoreAccumulator:
    """Average per-view contribution sums, normalized by their positive mean.

    Feed the renderer's ``WeightedContributionSum`` result for each sampled
    view. Only accepted blends contribute; background pixels add no GS score.
    This object owns no cameras, renderer, or trainable Gaussian parameters.
    """

    def __init__(self, n: int, device: torch.device | str) -> None:
        self.total = torch.zeros(n, device=device)
        self.views = 0

    @torch.no_grad()
    def add(self, contributions: Tensor, visible: Tensor | None = None) -> None:
        if (
            contributions.shape != self.total.shape
            or contributions.device != self.total.device
        ):
            raise ValueError(
                "contributions must have N entries on the accumulator device"
            )
        values = torch.nan_to_num(
            contributions.detach().float(), nan=0, posinf=0, neginf=0
        ).clamp_min(0)
        positive = values > 0
        mean = values.sum() / positive.sum().clamp_min(1)
        normalized = values / torch.where(mean > 0, mean, 1.0)
        if visible is not None:
            if (
                visible.shape != values.shape
                or visible.dtype != torch.bool
                or visible.device != values.device
            ):
                raise ValueError(
                    "visible must be a boolean N-vector on the same device"
                )
            normalized = torch.where(visible, normalized, 0)
        self.total.add_(normalized)
        self.views += 1

    def result(self) -> Tensor:
        return self.total / max(self.views, 1)
