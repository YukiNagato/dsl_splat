"""Differentiable 2D SSIM using a fused Warp Gaussian filter."""

import math
from typing import Any, Literal

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
import warp as wp

from ._ssim_kernels import backward_kernel, forward_kernel
from ..runtime.dispatch import FrameBindings, LaunchCache, kernel_arg, launch_tiled
from ..runtime.interop import current_stream

_launch_cache = LaunchCache()


class _SSIM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, x: Tensor, y: Tensor, crop: int, data_range: float, grad_enabled: bool
    ) -> Tensor:
        wp.init()
        batch, channels, height, width = x.shape
        planes = batch * channels
        output_shape = (batch, channels, height - 2 * crop, width - 2 * crop)
        partial_shape = (planes, *output_shape[2:], 3)
        needs = tuple(grad_enabled and value for value in ctx.needs_input_grad[:2])
        output = x.new_empty(output_shape)
        partials = [x.new_empty(partial_shape) if need else None for need in needs]
        bindings = FrameBindings(_launch_cache)
        launch_tiled(
            forward_kernel(*needs),
            dim=(planes, (height + 15) // 16, (width + 15) // 16),
            inputs=[
                kernel_arg(x, bindings=bindings),
                kernel_arg(y, bindings=bindings),
                channels,
                crop,
                (0.01 * data_range) ** 2,
                (0.03 * data_range) ** 2,
            ],
            outputs=[
                kernel_arg(output, bindings=bindings),
                *[kernel_arg(p, wp.vec3, bindings) for p in partials],
            ],
            stream=current_stream(x.device, bindings=bindings),
            block_dim=256,
            bindings=bindings,
        )
        if any(needs):
            ctx.save_for_backward(x, y, *partials)
            ctx.needs, ctx.crop = needs, crop
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, upstream: Tensor | None) -> tuple[Tensor | None, ...]:
        if upstream is None:
            return (None,) * 5
        x, y, partial_x, partial_y = ctx.saved_tensors
        shape = x.shape
        batch, channels, height, width = shape
        gradients = [x.new_empty(shape) if need else None for need in ctx.needs]
        bindings = FrameBindings(_launch_cache)
        launch_tiled(
            backward_kernel(*ctx.needs),
            dim=(batch * channels, (height + 15) // 16, (width + 15) // 16),
            inputs=[
                kernel_arg(x, bindings=bindings),
                kernel_arg(y, bindings=bindings),
                kernel_arg(upstream, bindings=bindings),
                kernel_arg(partial_x, wp.vec3, bindings),
                kernel_arg(partial_y, wp.vec3, bindings),
                channels,
                ctx.crop,
            ],
            outputs=[kernel_arg(value, bindings=bindings) for value in gradients],
            stream=current_stream(x.device, bindings=bindings),
            block_dim=256,
            bindings=bindings,
        )
        return (*gradients, None, None, None)


def fused_ssim(
    image: Tensor,
    target: Tensor,
    padding: Literal["same", "valid"] = "same",
    *,
    data_format: Literal["NCHW", "NHWC"] = "NCHW",
    data_range: float = 1.0,
    reduction: Literal["mean", "none"] = "mean",
) -> Tensor:
    """SSIM similarity with an 11x11 Gaussian window (sigma=1.5).

    Inputs are equal-shaped CUDA float32 images in NCHW or NHWC format.
    Unbatched CHW/HWC images are also accepted with the corresponding format.
    Strided inputs are read directly, without layout copies. ``same`` uses zero
    padding; ``valid`` excludes a five-pixel border and requires H,W >= 11.
    Return the mean similarity, or a map in the input layout for ``none``.
    Variances are clamped to zero and SSIM to [-1,1], matching fused-ssim.
    Both inputs support first-order gradients; use ``1 - fused_ssim(...)`` as
    a training loss. Gradients and saved state are automatic, with no train flag.
    ``data_range`` is the positive intensity range (normally 1.0); inputs are
    expected to be finite and are not clipped or normalized.
    """
    if not isinstance(image, Tensor) or not isinstance(target, Tensor):
        raise TypeError("image and target must be Torch tensors")
    if (
        not image.is_cuda
        or image.dtype != torch.float32
        or target.device != image.device
        or target.dtype != image.dtype
    ):
        raise ValueError("image and target must be CUDA float32 on the same device")
    shape = image.shape
    if shape != target.shape or len(shape) not in (3, 4) or any(s == 0 for s in shape):
        raise ValueError("images must have equal, nonempty CHW/HWC or NCHW/NHWC shapes")
    if (
        data_format not in ("NCHW", "NHWC")
        or padding not in ("same", "valid")
        or reduction not in ("mean", "none")
    ):
        raise ValueError("invalid data_format, padding or reduction")
    data_range = float(data_range)
    if not math.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be finite and positive")
    unbatched = len(shape) == 3
    x, y = (value.unsqueeze(0) if unbatched else value for value in (image, target))
    if data_format == "NHWC":
        x, y = x.permute(0, 3, 1, 2), y.permute(0, 3, 1, 2)
    crop = 5 if padding == "valid" else 0
    if min(x.shape[2:]) <= 2 * crop:
        raise ValueError("valid padding requires image height and width >= 11")
    output = _SSIM.apply(x, y, crop, data_range, torch.is_grad_enabled())
    if reduction == "mean":
        return output.mean()
    if data_format == "NHWC":
        output = output.permute(0, 2, 3, 1)
    return output[0] if unbatched else output
