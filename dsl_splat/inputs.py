"""Batched Gaussian layouts, feature selection and differentiable SH evaluation."""

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor
import torch.nn.functional as F
from .primitives.spherical_harmonics import spherical_harmonics as _warp_sh

RenderMode = Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"]


def batch_item(values: Tensor, index: int) -> Tensor:
    """Use squeeze for singleton batches, avoiding a gradient scatter/copy."""
    return values.squeeze(0) if index == 0 and values.shape[0] == 1 else values[index]


@dataclass(slots=True)
class ViewFeatures:
    values: tuple[Tensor, ...]
    per_view: bool

    @property
    def channels(self) -> int:
        return self.values[0].shape[-1]

    def at(self, batch: int, camera: int) -> Tensor:
        values = self.values[batch]
        return batch_item(values, camera) if self.per_view else values


@dataclass(slots=True)
class Inputs:
    batch_dims: tuple[int, ...]
    batch_count: int
    n: int
    cameras: int
    device: torch.device
    means: tuple[Tensor, ...]
    quats: tuple[Tensor, ...]
    scales: tuple[Tensor, ...]
    opacities: tuple[Tensor, ...]
    viewmats: Tensor
    Ks: Tensor
    colors: ViewFeatures
    backgrounds: Tensor | None
    extras: ViewFeatures | None
    filters: Tensor | None
    requires_screen_grad: bool


def spherical_harmonics(
    coefficients: Tensor, directions: Tensor, degree: int
) -> Tensor:
    """SH0–3 with fused CUDA float32 gradients and a CPU/double Torch path."""
    if coefficients.is_cuda and coefficients.dtype == directions.dtype == torch.float32:
        return _warp_sh(coefficients, directions, degree)
    return _torch_spherical_harmonics(coefficients, directions, degree)


def _torch_spherical_harmonics(
    coefficients: Tensor, directions: Tensor, degree: int
) -> Tensor:
    """Torch reference, also used for CPU/double numerical differentiation."""
    direction = F.normalize(directions, dim=-1)
    x, y, z = direction.unbind(-1)
    basis = [torch.full_like(x, 0.28209479177387814)]
    if degree >= 1:
        basis += [
            -0.4886025119029199 * y,
            0.4886025119029199 * z,
            -0.4886025119029199 * x,
        ]
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        basis += [
            1.0925484305920792 * x * y,
            -1.0925484305920792 * y * z,
            0.31539156525252005 * (2 * zz - xx - yy),
            -1.0925484305920792 * x * z,
            0.5462742152960396 * (xx - yy),
        ]
    if degree >= 3:
        basis += [
            -0.5900435899266435 * y * (3 * xx - yy),
            2.890611442640554 * x * y * z,
            -0.4570457994644658 * y * (4 * zz - xx - yy),
            0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy),
            -0.4570457994644658 * x * (4 * zz - xx - yy),
            1.445305721320277 * z * (xx - yy),
            -0.5900435899266435 * x * (xx - 3 * yy),
        ]
    weights = torch.stack(basis, dim=-1)
    return (
        (coefficients[:, : len(basis)] * weights[..., None]).sum(-2) + 0.5
    ).clamp_min(0)


def check_tensor(
    name: str,
    tensor: Tensor,
    device: torch.device,
    shape: tuple[int, ...] | None = None,
) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device != device
        or tensor.dtype != torch.float32
    ):
        raise ValueError(f"{name} must be float32 on {device}")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )


def _batch_views(
    value: Tensor, batch_count: int, item_shape: tuple[int, ...]
) -> tuple[Tensor, ...]:
    """Keep unbatched inputs intact and share one unbind across real batches."""
    if batch_count == 1:
        return (value if value.shape == item_shape else value.reshape(item_shape),)
    shape = (batch_count,) + item_shape
    values = value if value.shape == shape else value.reshape(shape)
    return values.unbind(0)


def _view_values(
    name: str,
    value: Tensor,
    batch_dims: tuple[int, ...],
    batch_count: int,
    cameras: int,
    n: int,
    device: torch.device,
    *,
    sh: bool = False,
) -> ViewFeatures:
    check_tensor(name, value, device)
    trailing = 3 if sh else 2
    shared_rank = len(batch_dims) + trailing
    per_view = value.ndim == shared_rank + 1
    tail = tuple(value.shape[-trailing:])
    expected = batch_dims + ((cameras,) if per_view else ()) + tail
    if (
        value.ndim not in (shared_rank, shared_rank + 1)
        or tuple(value.shape) != expected
        or tail[0] != n
        or tail[-1] <= 0
    ):
        raise ValueError(
            f"{name} must have batch dimensions {batch_dims} and shape (...,N,D) or (...,C,N,D)"
            if not sh
            else f"{name} must be (...,N,K,3) or (...,C,N,K,3)"
        )
    if sh and tail[-1] != 3:
        raise ValueError("SH colors must have three channels")
    item_shape = ((cameras,) if per_view else ()) + tail
    return ViewFeatures(_batch_views(value, batch_count, item_shape), per_view)


def normalize_inputs(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor | None,
    viewmats: Tensor,
    Ks: Tensor,
    backgrounds: Tensor | None,
    extras: Tensor | None,
    filter3D: Tensor | None,
    sh_degree: int | None,
    *,
    normalize_quats: bool = True,
) -> Inputs:
    """Check layouts, optionally delegating quaternion normalization to a kernel."""
    extra_signals = extras
    if (
        not isinstance(means, torch.Tensor)
        or not means.is_cuda
        or means.ndim < 2
        or means.shape[-1] != 3
    ):
        raise ValueError("means must be a CUDA float32 tensor with shape (...,N,3)")
    device = means.device
    check_tensor("means", means, device)
    batch_dims = tuple(means.shape[:-2])
    batch_count, n = math.prod(batch_dims), means.shape[-2]
    if batch_count <= 0:
        raise ValueError("Gaussian batch dimensions must be nonempty")
    check_tensor("quats", quats, device, batch_dims + (n, 4))
    check_tensor("scales", scales, device, batch_dims + (n, 3))
    check_tensor("opacities", opacities, device, batch_dims + (n,))
    check_tensor("viewmats", viewmats, device)
    if (
        viewmats.ndim != len(batch_dims) + 3
        or viewmats.shape[-2:] != (4, 4)
        or tuple(viewmats.shape[:-3]) != batch_dims
    ):
        raise ValueError(
            "viewmats must have shape (...,C,4,4) with the Gaussian batch dimensions"
        )
    cameras = viewmats.shape[-3]
    if cameras <= 0:
        raise ValueError("at least one camera is required")
    check_tensor("Ks", Ks, device, batch_dims + (cameras, 3, 3))
    if colors is None:
        raise ValueError("colors is required for RGB/features rendering")
    color_bank = _view_values(
        "colors",
        colors,
        batch_dims,
        batch_count,
        cameras,
        n,
        device,
        sh=sh_degree is not None,
    )
    if sh_degree is not None and color_bank.values[0].shape[-2] < (sh_degree + 1) ** 2:
        raise ValueError("not enough SH coefficients for sh_degree")
    d = color_bank.channels
    background_bank = None
    if backgrounds is not None:
        check_tensor("backgrounds", backgrounds, device, batch_dims + (cameras, d))
        background_bank = backgrounds.reshape(batch_count, cameras, d)
    extra_bank = None
    if extra_signals is not None:
        extra_bank = _view_values(
            "extra_signals", extra_signals, batch_dims, batch_count, cameras, n, device
        )
    filters = None
    if filter3D is not None:
        check_tensor("filter3D", filter3D, device)
        if tuple(filter3D.shape) not in (batch_dims + (n,), batch_dims + (n, 1)):
            raise ValueError("filter3D must have shape (...,N) or (...,N,1)")
        filters = filter3D.reshape(batch_count, n)
    if torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (Ks, backgrounds)
    ):
        raise NotImplementedError("intrinsics/background gradients are not supported")
    return Inputs(
        batch_dims,
        batch_count,
        n,
        cameras,
        device,
        _batch_views(means, batch_count, (n, 3)),
        _batch_views(
            F.normalize(quats, dim=-1) if normalize_quats else quats,
            batch_count,
            (n, 4),
        ),
        _batch_views(scales, batch_count, (n, 3)),
        _batch_views(opacities, batch_count, (n,)),
        viewmats.reshape(batch_count, cameras, 4, 4),
        Ks.detach().reshape(batch_count, cameras, 3, 3),
        color_bank,
        background_bank,
        extra_bank,
        filters,
        torch.is_grad_enabled()
        and any(
            isinstance(value, Tensor) and value.requires_grad
            for value in (means, quats, scales, opacities, colors, extras, viewmats)
        ),
    )
