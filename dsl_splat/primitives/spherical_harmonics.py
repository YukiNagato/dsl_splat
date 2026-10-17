"""First-order Torch autograd bridge for fused CUDA float32 SH0--3."""

from numbers import Integral
from typing import Any

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
import warp as wp

from .sh import forward_kernel, backward_kernel
from ..runtime.dispatch import FrameBindings, LaunchCache, kernel_arg, launch
from ..runtime.interop import current_stream

_launch_cache = LaunchCache()


class _SphericalHarmonics(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        coefficients: Tensor,
        directions: Tensor,
        radii: Tensor | None,
        degree: int,
        bindings: FrameBindings,
        grad_enabled: bool,
    ) -> Tensor:
        wp.init()
        n = len(coefficients)
        masked = radii is not None
        if radii is None:
            radii = coefficients.new_empty((0,), dtype=torch.int32)
        needs = (ctx.needs_input_grad[0], degree > 0 and ctx.needs_input_grad[1])
        save = grad_enabled and any(needs)
        colors = coefficients.new_empty((n, 3))
        active = coefficients.new_empty((n if save else 0, 3), dtype=torch.bool)
        if n:
            launch(
                forward_kernel(degree, masked, save),
                dim=n,
                inputs=[
                    kernel_arg(coefficients, bindings=bindings),
                    kernel_arg(directions, bindings=bindings),
                    kernel_arg(radii, bindings=bindings),
                ],
                outputs=[
                    kernel_arg(colors, wp.vec3, bindings),
                    kernel_arg(active, bindings=bindings),
                ],
                stream=current_stream(coefficients.device, bindings=bindings),
                bindings=bindings,
            )
        if save:
            ctx.save_for_backward(coefficients, directions, radii, active)
            ctx.degree, ctx.masked, ctx.needs = degree, masked, needs
            ctx.launch_cache = bindings.launch_cache
        ctx.set_materialize_grads(False)
        return colors

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, gradient: Tensor | None) -> tuple[Tensor | None, ...]:
        if gradient is None:
            return (None,) * 6
        coefficients, directions, radii, active = ctx.saved_tensors
        n = len(coefficients)
        need_coefficients, need_directions = ctx.needs
        # Dense gradients must include zero culled rows and padded coefficients.
        # A contiguous fill is faster than strided per-Gaussian zero stores.
        grad_coefficients = coefficients.new_zeros(
            coefficients.shape if need_coefficients else (0, 0, 3)
        )
        grad_directions = directions.new_empty((n if need_directions else 0, 3))
        bindings = FrameBindings(ctx.launch_cache)
        if n:
            launch(
                backward_kernel(ctx.degree, ctx.masked, *ctx.needs),
                dim=n,
                inputs=[
                    kernel_arg(coefficients, bindings=bindings),
                    kernel_arg(directions, bindings=bindings),
                    kernel_arg(radii, bindings=bindings),
                    kernel_arg(active, bindings=bindings),
                    kernel_arg(gradient, bindings=bindings),
                ],
                outputs=[
                    kernel_arg(grad_coefficients, bindings=bindings),
                    kernel_arg(grad_directions, wp.vec3, bindings),
                ],
                stream=current_stream(coefficients.device, bindings=bindings),
                bindings=bindings,
            )
        return (
            grad_coefficients if need_coefficients else None,
            grad_directions if need_directions else None,
            None,
            None,
            None,
            None,
        )


def spherical_harmonics(
    coefficients: Tensor,
    directions: Tensor,
    degree: int,
    *,
    radii: Tensor | None = None,
    bindings: FrameBindings | None = None,
) -> Tensor:
    """Evaluate SH color with +0.5/lower clamp; radii<=0 rows return zero.

    Strided CUDA float32 coefficients (N,K,3) and directions (N,3) are accepted.
    Gradients include coefficients and directions; padded/culled entries are zero.
    """
    if (
        isinstance(degree, bool)
        or not isinstance(degree, Integral)
        or degree not in range(4)
    ):
        raise ValueError("SH degree must be an integer in [0,3]")
    if (
        not isinstance(coefficients, Tensor)
        or not coefficients.is_cuda
        or coefficients.dtype != torch.float32
        or coefficients.ndim != 3
        or coefficients.shape[-1] != 3
        or coefficients.shape[1] < (degree + 1) ** 2
    ):
        raise ValueError(
            "SH coefficients must be CUDA float32 (N,K,3) with enough coefficients"
        )
    if (
        not isinstance(directions, Tensor)
        or directions.device != coefficients.device
        or directions.dtype != torch.float32
        or directions.shape != (len(coefficients), 3)
    ):
        raise ValueError(
            "SH directions must be float32 (N,3) on the coefficients' device"
        )
    if radii is not None and (
        radii.device != coefficients.device
        or radii.dtype != torch.int32
        or radii.shape != (len(coefficients),)
    ):
        raise ValueError("SH radii must be int32 (N,) on the coefficients' device")
    # SH0 has no direction dependence, including no autograd edge to directions.
    if degree == 0:
        directions = directions.detach()
    return _SphericalHarmonics.apply(
        coefficients,
        directions,
        radii,
        int(degree),
        FrameBindings(_launch_cache) if bindings is None else bindings,
        torch.is_grad_enabled(),
    )
