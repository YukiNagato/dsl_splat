"""gsplat input layouts, feature selection and differentiable SH evaluation."""

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor
import torch.nn.functional as F

RenderMode = Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"]


@dataclass(slots=True)
class ViewFeatures:
    values: Tensor
    per_view: bool

    @property
    def channels(self) -> int:
        return self.values.shape[-1]

    def at(self, batch: int, camera: int) -> Tensor:
        return self.values[batch, camera] if self.per_view else self.values[batch]


@dataclass(slots=True)
class Inputs:
    batch_dims: tuple[int, ...]
    batch_count: int
    n: int
    cameras: int
    device: torch.device
    means: Tensor
    quats: Tensor
    scales: Tensor
    opacities: Tensor
    viewmats: Tensor
    Ks: Tensor
    colors: ViewFeatures | None
    backgrounds: Tensor | None
    extras: ViewFeatures | None
    filters: Tensor | None
    requires_screen_grad: bool


def spherical_harmonics(
    coefficients: Tensor, directions: Tensor, degree: int
) -> Tensor:
    """Degree 0–3 real SH, with Torch derivatives through coefficients/direction."""
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
    shape = (batch_count,) + ((cameras,) if per_view else ()) + tail
    return ViewFeatures(value.reshape(shape), per_view)


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
    render_mode: RenderMode,
    sh_degree: int | None,
) -> Inputs:
    """Check layouts once and normalize quaternions with Torch gradients."""
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
        if "RGB" in render_mode:
            raise ValueError("colors is required for RGB render modes")
        if sh_degree is not None:
            raise ValueError("sh_degree requires colors")
        color_bank, d = None, 0
    else:
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
        if sh_degree is not None and color_bank.values.shape[-2] < (sh_degree + 1) ** 2:
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
        t is not None and t.requires_grad for t in (viewmats, Ks, backgrounds)
    ):
        raise NotImplementedError(
            "camera/intrinsics/background gradients are not supported"
        )
    return Inputs(
        batch_dims,
        batch_count,
        n,
        cameras,
        device,
        means.reshape(batch_count, n, 3),
        F.normalize(quats.reshape(batch_count, n, 4), dim=-1),
        scales.reshape(batch_count, n, 3),
        opacities.reshape(batch_count, n),
        viewmats.detach().reshape(batch_count, cameras, 4, 4),
        Ks.detach().reshape(batch_count, cameras, 3, 3),
        color_bank,
        background_bank,
        extra_bank,
        filters,
        torch.is_grad_enabled()
        and any(
            isinstance(value, Tensor) and value.requires_grad
            for value in (means, quats, scales, opacities, colors, extras)
        ),
    )
