"""GEER canonical transforms and their local Warp-generated derivatives.

Based on Bosch Research's 3DGEER, revision 3e889e8b19903507714fd48b93cf69a68795e640.
See README.md, GEER_LICENSE.txt and GEER_RASTERIZER_LICENSE.txt.
"""

from typing import Any

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
import warp as wp

from ..primitives.quaternions import covariance_rotation
from ..primitives.math import fused_multiply_add as _fma
from ..runtime.dispatch import FrameBindings, LaunchCache, kernel_arg, launch
from ..runtime.interop import current_stream

launch_cache = LaunchCache()


@wp.func
def load_view(view: wp.array2d(dtype=float)):
    return wp.mat44(
        view[0, 0],
        view[0, 1],
        view[0, 2],
        view[0, 3],
        view[1, 0],
        view[1, 1],
        view[1, 2],
        view[1, 3],
        view[2, 0],
        view[2, 1],
        view[2, 2],
        view[2, 3],
        view[3, 0],
        view[3, 1],
        view[3, 2],
        view[3, 3],
    )


@wp.func
def compensated_opacity(scale: wp.vec3, opacity: float):
    variance = wp.cw_mul(scale, scale) + wp.vec3(1.0e-7)
    factor = (
        scale[0]
        * scale[1]
        * scale[2]
        / wp.sqrt(variance[0] * variance[1] * variance[2])
    )
    return factor * opacity


@wp.func_grad(compensated_opacity)
def adj_compensated_opacity(scale: wp.vec3, opacity: float, gradient: float):
    # Match the CUDA training convention: compensation has no scale derivative.
    variance = wp.cw_mul(scale, scale) + wp.vec3(1.0e-7)
    factor = (
        scale[0]
        * scale[1]
        * scale[2]
        / wp.sqrt(variance[0] * variance[1] * variance[2])
    )
    wp.adjoint[opacity] += gradient * wp.max(factor, 0.005)


@wp.func
def canonical(
    mean: wp.vec3, quat: wp.vec4, scale: wp.vec3, opacity: float, view: wp.mat44
):
    q = quat / wp.max(wp.length(quat), 1.0e-12)
    rotation = wp.mat33(
        view[0, 0],
        view[0, 1],
        view[0, 2],
        view[1, 0],
        view[1, 1],
        view[1, 2],
        view[2, 0],
        view[2, 1],
        view[2, 2],
    )
    position = wp.vec3(
        view[0, 0] * mean[0] + view[0, 1] * mean[1] + view[0, 2] * mean[2] + view[0, 3],
        view[1, 0] * mean[0] + view[1, 1] * mean[1] + view[1, 2] * mean[2] + view[1, 3],
        view[2, 0] * mean[0] + view[2, 1] * mean[1] + view[2, 2] * mean[2] + view[2, 3],
    )
    variance = wp.cw_mul(scale, scale) + wp.vec3(1.0e-7)
    inv = wp.vec3(
        1.0 / wp.sqrt(variance[0]),
        1.0 / wp.sqrt(variance[1]),
        1.0 / wp.sqrt(variance[2]),
    )
    matrix = wp.diag(inv) * covariance_rotation(q) * wp.transpose(rotation)
    # Preserve the reference's arithmetic order: float-distance ties determine
    # blending order, so a one-ULP change can create a visible color difference.
    distance = wp.sqrt(
        _fma(
            position[1],
            position[1],
            _fma(position[0], position[0], position[2] * position[2]),
        )
    )
    return position, matrix, compensated_opacity(scale, opacity), distance


@wp.func
def canonical_loss(
    mean: wp.vec3,
    quat: wp.vec4,
    scale: wp.vec3,
    opacity: float,
    view: wp.mat44,
    gp: wp.vec3,
    gm: wp.mat33,
    go: float,
    gd: float,
):
    p, m, o, d = canonical(mean, quat, scale, opacity, view)
    value = wp.dot(p, gp) + o * go + d * gd
    for row in range(3):
        for col in range(3):
            value += m[row, col] * gm[row, col]
    return value


@wp.kernel(enable_backward=False)
def prepare(
    means: wp.array(dtype=wp.vec3),
    quats: wp.array(dtype=wp.vec4),
    scales: wp.array(dtype=wp.vec3),
    opacities: wp.array(dtype=float),
    view: wp.array2d(dtype=float),
    positions: wp.array(dtype=wp.vec3),
    matrices: wp.array(dtype=wp.mat33),
    filtered: wp.array(dtype=float),
    depths: wp.array(dtype=float),
):
    i = wp.tid()
    p, m, o, d = canonical(means[i], quats[i], scales[i], opacities[i], load_view(view))
    positions[i] = p
    matrices[i] = m
    filtered[i] = o
    depths[i] = d


@wp.kernel(enable_backward=False)
def prepare_backward(
    means: wp.array(dtype=wp.vec3),
    quats: wp.array(dtype=wp.vec4),
    scales: wp.array(dtype=wp.vec3),
    opacities: wp.array(dtype=float),
    view: wp.array2d(dtype=float),
    gp: wp.array(dtype=wp.vec3),
    gm: wp.array(dtype=wp.mat33),
    go: wp.array(dtype=float),
    gd: wp.array(dtype=float),
    grad_means: wp.array(dtype=wp.vec3),
    grad_quats: wp.array(dtype=wp.vec4),
    grad_scales: wp.array(dtype=wp.vec3),
    grad_opacities: wp.array(dtype=float),
    grad_view: wp.array2d(dtype=float),
    need_view: bool,
):
    i = wp.tid()
    a, b, c, d, e, unused_gp, unused_gm, unused_go, unused_gd = wp.grad(canonical_loss)(
        means[i],
        quats[i],
        scales[i],
        opacities[i],
        load_view(view),
        gp[i],
        gm[i],
        go[i],
        gd[i],
    )
    grad_means[i] = a
    grad_quats[i] = b
    grad_scales[i] = c
    grad_opacities[i] = d
    if need_view:
        for row in range(3):
            for col in range(4):
                wp.atomic_add(grad_view, row, col, e[row, col])


def arguments(values, bindings):
    return [
        kernel_arg(v, t, bindings)
        for v, t in zip(values, (wp.vec3, wp.vec4, wp.vec3, wp.float32, wp.float32))
    ]


class _Geometry(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        opacities: Tensor,
        view: Tensor,
        grad_enabled: bool,
    ):
        wp.init()
        n = len(means)
        output = (
            means.new_empty((n, 3)),
            means.new_empty((n, 3, 3)),
            means.new_empty(n),
            means.new_empty(n),
        )
        bindings = FrameBindings(launch_cache)
        if n:
            launch(
                prepare,
                dim=n,
                inputs=arguments((means, quats, scales, opacities, view), bindings),
                outputs=[
                    kernel_arg(v, t, bindings)
                    for v, t in zip(output, (wp.vec3, wp.mat33, wp.float32, wp.float32))
                ],
                stream=current_stream(means.device, bindings=bindings),
                bindings=bindings,
            )
        if grad_enabled and any(ctx.needs_input_grad[:5]):
            ctx.save_for_backward(means, quats, scales, opacities, view)
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, gp, gm, go, gd):
        values = ctx.saved_tensors
        means, quats, scales, opacities, view = values
        n = len(means)
        seeds = [
            g.contiguous() if g is not None else means.new_zeros(shape)
            for g, shape in zip((gp, gm, go, gd), ((n, 3), (n, 3, 3), (n,), (n,)))
        ]
        gradients = [torch.empty_like(v) for v in values[:4]] + [torch.zeros_like(view)]
        bindings = FrameBindings(launch_cache)
        if n:
            launch(
                prepare_backward,
                dim=n,
                inputs=arguments(values, bindings)
                + [
                    kernel_arg(v, t, bindings)
                    for v, t in zip(seeds, (wp.vec3, wp.mat33, wp.float32, wp.float32))
                ],
                outputs=arguments(gradients, bindings) + [ctx.needs_input_grad[4]],
                stream=current_stream(means.device, bindings=bindings),
                bindings=bindings,
            )
        return (
            *[
                g if need else None
                for g, need in zip(gradients, ctx.needs_input_grad[:5])
            ],
            None,
        )


def prepare_gaussians(
    means: Tensor, quats: Tensor, scales: Tensor, opacities: Tensor, view: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _Geometry.apply(
        means, quats, scales, opacities, view, torch.is_grad_enabled()
    )
