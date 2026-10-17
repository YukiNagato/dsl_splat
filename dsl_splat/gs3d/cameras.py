"""GPU preparation of OpenCV cameras for the original-3DGS projection."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
import warp as wp

from .camera_kernels import prepare_kernel, backward_kernel
from ..cameras import PinholeCamera, validate_camera_options
from ..inputs import check_tensor
from ..runtime.dispatch import FrameBindings, kernel_arg, launch
from ..runtime.interop import current_stream


@dataclass(frozen=True, slots=True)
class ViewConfig:
    width: int
    height: int
    near: float
    far: float
    radius_clip: float


class _PrepareCameras(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        views: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        bindings: FrameBindings,
        grad_enabled: bool,
        with_centers: bool,
    ) -> tuple[Tensor, Tensor]:
        wp.init()
        full = torch.empty_like(views, memory_format=torch.contiguous_format)
        flat_views, flat_Ks = views.reshape(-1, 4, 4), Ks.reshape(-1, 3, 3)
        errors = torch.empty(len(flat_views), dtype=torch.int32, device=views.device)
        inverse = views.new_empty((len(flat_views) if with_centers else 0, 4, 4))
        centers = views.new_empty(views.shape[:-2] + (3,) if with_centers else (0, 3))
        launch(
            prepare_kernel,
            dim=len(flat_views),
            inputs=[
                kernel_arg(flat_views, bindings=bindings),
                kernel_arg(flat_Ks, bindings=bindings),
                width,
                height,
                int(with_centers),
            ],
            outputs=[
                kernel_arg(full.reshape(-1, 4, 4), bindings=bindings),
                kernel_arg(inverse, bindings=bindings),
                kernel_arg(centers.reshape(-1, 3), bindings=bindings),
                kernel_arg(errors, bindings=bindings),
            ],
            stream=current_stream(views.device, bindings=bindings),
            bindings=bindings,
        )
        # Preserve eager, recoverable Python exceptions. Only status codes cross
        # to the CPU; K values and all projection arithmetic stay on the GPU.
        codes = errors.tolist()
        if 2 in codes:
            raise ValueError("Ks must be finite with positive fx/fy")
        if 1 in codes:
            raise NotImplementedError("Ks must be a zero-skew pinhole intrinsic matrix")
        if 3 in codes:
            raise torch.linalg.LinAlgError(
                "viewmats must be invertible for SH directions"
            )
        if grad_enabled and ctx.needs_input_grad[0]:
            ctx.save_for_backward(Ks, views, inverse)
            ctx.width, ctx.height = width, height
            ctx.launch_cache = bindings.launch_cache
        ctx.set_materialize_grads(False)
        if not with_centers:
            ctx.mark_non_differentiable(centers)
        return full, centers

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any, gradient: Tensor | None, grad_centers: Tensor | None
    ) -> tuple[Tensor | None, ...]:
        if gradient is None and grad_centers is None:
            return (None,) * 7
        Ks, views, inverse = ctx.saved_tensors
        grad_views = torch.empty_like(views, memory_format=torch.contiguous_format)
        full_gradient = views.new_empty((0, 4, 4)) if gradient is None else gradient
        center_gradient = (
            views.new_empty((0, 3)) if grad_centers is None else grad_centers
        )
        bindings = FrameBindings(ctx.launch_cache)
        launch(
            backward_kernel,
            dim=Ks.numel() // 9,
            inputs=[
                kernel_arg(Ks.reshape(-1, 3, 3), bindings=bindings),
                kernel_arg(full_gradient.reshape(-1, 4, 4), bindings=bindings),
                ctx.width,
                ctx.height,
                kernel_arg(inverse, bindings=bindings),
                kernel_arg(center_gradient.reshape(-1, 3), bindings=bindings),
                int(gradient is not None),
                int(grad_centers is not None),
            ],
            outputs=[kernel_arg(grad_views.reshape(-1, 4, 4), bindings=bindings)],
            stream=current_stream(views.device, bindings=bindings),
            bindings=bindings,
        )
        return grad_views, None, None, None, None, None, None


def prepare_cameras(
    views: Tensor, Ks: Tensor, width: int, height: int, bindings: FrameBindings
) -> Tensor:
    """Build image-plane projection rows once per current K/view pair on GPU."""
    return _prepare_cameras(views, Ks, width, height, bindings, False)[0]


def prepare_cameras_with_centers(
    views: Tensor, Ks: Tensor, width: int, height: int, bindings: FrameBindings
) -> tuple[Tensor, Tensor]:
    """Build projection and SH camera centers in one launch and one pose VJP."""
    return _prepare_cameras(views, Ks, width, height, bindings, True)


def _prepare_cameras(
    views: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    bindings: FrameBindings,
    with_centers: bool,
) -> tuple[Tensor, Tensor]:
    if torch.is_grad_enabled() and Ks.requires_grad:
        raise NotImplementedError("intrinsics gradients are not supported")
    return _PrepareCameras.apply(
        views, Ks, width, height, bindings, torch.is_grad_enabled(), with_centers
    )


def prepare_camera(
    camera: PinholeCamera, radius_clip: float, bindings: FrameBindings | None = None
) -> tuple[Tensor, Tensor, Tensor, ViewConfig]:
    validate_camera_options(
        camera.width, camera.height, camera.near_plane, camera.far_plane, radius_clip
    )
    if not isinstance(camera.viewmat, Tensor) or not camera.viewmat.is_cuda:
        raise ValueError("camera tensors must be CUDA float32")
    check_tensor("viewmat", camera.viewmat, camera.viewmat.device, (4, 4))
    check_tensor("K", camera.K, camera.viewmat.device, (3, 3))
    if torch.is_grad_enabled() and camera.K.requires_grad:
        raise NotImplementedError("intrinsics gradients are not supported")
    intrinsic = camera.K.detach().contiguous()
    view = camera.viewmat.contiguous()
    projection = prepare_cameras(
        view,
        intrinsic,
        camera.width,
        camera.height,
        FrameBindings() if bindings is None else bindings,
    )
    config = ViewConfig(
        camera.width,
        camera.height,
        camera.near_plane,
        camera.far_plane,
        radius_clip,
    )
    return view, projection, intrinsic, config
