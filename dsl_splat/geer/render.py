"""GEER ray--Gaussian evaluation, alpha blending and reverse traversal in Warp.

Algorithm adapted from the official forward.cu/backward.cu; see README.md.
"""

from functools import lru_cache
from typing import Any

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
import warp as wp

from .geometry import launch_cache
from .binning import ParticleBounds
from ..primitives.tiling import TileBins
from ..runtime.dispatch import FrameBindings, kernel_arg, launch, launch_tiled
from ..runtime.interop import current_stream
from ..runtime.gradients import AbsGradientTarget


@wp.func
def _inside_pbf(ray: wp.vec3, b: wp.vec4, mode: int):
    valid = (
        wp.dot(ray, ray) > 0.0
        and wp.isfinite(ray[0])
        and wp.isfinite(ray[1])
        and wp.isfinite(ray[2])
    )
    if mode == 1:
        valid = valid and ray[2] != 0.0
        if valid:
            valid = (
                ray[0] / ray[2] >= b[0]
                and ray[0] / ray[2] <= b[1]
                and ray[1] / ray[2] >= b[2]
                and ray[1] / ray[2] <= b[3]
            )
    return valid


@lru_cache(maxsize=32)
def forward_kernel(channels: int):
    Feature = wp.types.vector(channels, wp.float32)

    @wp.kernel(module="unique", enable_backward=False)
    def forward(
        width: int,
        height: int,
        mode: int,
        positions: wp.array(dtype=wp.vec3),
        matrices: wp.array(dtype=wp.mat33),
        opacities: wp.array(dtype=float),
        depths: wp.array(dtype=float),
        features: wp.array(dtype=Feature),
        rays: wp.array(dtype=wp.vec3),
        pbf: wp.array(dtype=wp.vec4),
        ranges: wp.array2d(dtype=int),
        ids: wp.array(dtype=int),
        background: wp.array(dtype=Feature),
        image: wp.array(dtype=Feature),
        alpha_image: wp.array(dtype=float),
        inverse: wp.array(dtype=float),
        final_t: wp.array(dtype=float),
        contributors: wp.array(dtype=int),
    ):
        tile, lane = wp.tid()
        gx = (width + 15) // 16
        x, y = (tile % gx) * 16 + lane % 16, (tile // gx) * 16 + lane // 16
        pixel = y * width + x
        inside = x < width and y < height
        done = not inside
        ray = wp.vec3(0.0, 0.0, 1.0)
        if inside:
            ray = rays[pixel]
        color = Feature()
        transmittance, invdepth = float(1.0), float(0.0)
        first, last = ranges[tile, 0], ranges[tile, 1]
        last_contributor = int(0)
        shared_center = wp.tile_empty(shape=(256,), dtype=wp.vec3, storage="shared")
        shared_m = wp.tile_empty(shape=(256,), dtype=wp.mat33, storage="shared")
        shared_o = wp.tile_empty(shape=(256,), dtype=float, storage="shared")
        shared_b = wp.tile_empty(shape=(256,), dtype=wp.vec4, storage="shared")
        shared_id = wp.tile_empty(shape=(256,), dtype=int, storage="shared")
        for begin in range(first, last, 256):
            if wp.tile_sum(wp.tile(int(not done)))[0] == 0:
                break
            index = begin + lane
            gaussian = int(0)
            p, m, o, b = wp.vec3(), wp.mat33(), float(0.0), wp.vec4()
            if index < last:
                gaussian = ids[index]
                p, m, o, b = (
                    positions[gaussian] + wp.vec3(),
                    matrices[gaussian] + wp.mat33(),
                    opacities[gaussian] + 0.0,
                    pbf[gaussian] + wp.vec4(),
                )
            # The canonical center is shared by every pixel in this tile.
            # Transform it once when loading the batch, rather than per ray.
            wp.tile_scatter_masked(shared_center, lane, m * p, index < last)
            wp.tile_scatter_masked(shared_m, lane, m, index < last)
            wp.tile_scatter_masked(shared_o, lane, o, index < last)
            wp.tile_scatter_masked(shared_b, lane, b, index < last)
            wp.tile_scatter_masked(shared_id, lane, gaussian, index < last)
            for j in range(wp.min(256, last - begin)):
                if not done:
                    if _inside_pbf(ray, shared_b[j], mode):
                        p_obj = shared_center[j]
                        d_obj = shared_m[j] * ray
                        normal = wp.cross(d_obj, p_obj)
                        power = -0.5 * wp.dot(normal, normal) / wp.dot(d_obj, d_obj)
                        if power <= 0.0:
                            alpha = wp.min(0.99, shared_o[j] * wp.exp(power))
                            if alpha >= 1.0 / 255.0:
                                next_t = transmittance * (1.0 - alpha)
                                if next_t < 0.0001:
                                    done = True
                                else:
                                    gid = shared_id[j]
                                    value = features[gid]
                                    weight = alpha * transmittance
                                    color += value * weight
                                    invdepth += weight / depths[gid]
                                    transmittance = next_t
                                    last_contributor = begin + j - first + 1
            # All lanes must finish reading before the next shared batch is written.
            wp.tile_sum(wp.tile(int(done)))
        if inside:
            image[pixel] = color + background[0] * transmittance
            alpha_image[pixel] = 1.0 - transmittance
            inverse[pixel] = invdepth
            final_t[pixel] = transmittance
            contributors[pixel] = last_contributor

    return forward


@lru_cache(maxsize=32)
def backward_kernel(channels: int):
    Feature = wp.types.vector(channels, wp.float32)

    @wp.kernel(module="unique", enable_backward=False)
    def backward(
        width: int,
        height: int,
        mode: int,
        need_background: bool,
        positions: wp.array(dtype=wp.vec3),
        matrices: wp.array(dtype=wp.mat33),
        opacities: wp.array(dtype=float),
        depths: wp.array(dtype=float),
        features: wp.array(dtype=Feature),
        rays: wp.array(dtype=wp.vec3),
        pbf: wp.array(dtype=wp.vec4),
        ranges: wp.array2d(dtype=int),
        ids: wp.array(dtype=int),
        background: wp.array(dtype=Feature),
        final_t: wp.array(dtype=float),
        contributors: wp.array(dtype=int),
        gimage: wp.array(dtype=Feature),
        galpha: wp.array(dtype=float),
        ginverse: wp.array(dtype=float),
        gp: wp.array(dtype=wp.vec3),
        gm: wp.array(dtype=wp.mat33),
        go: wp.array(dtype=float),
        gd: wp.array(dtype=float),
        gf: wp.array(dtype=Feature),
        grays: wp.array(dtype=wp.vec3),
        gbackground: wp.array(dtype=Feature),
        absolute: wp.array(dtype=wp.vec3),
    ):
        tile, lane = wp.tid()
        gx = (width + 15) // 16
        x, y = tile % gx * 16 + lane % 16, tile // gx * 16 + lane // 16
        pixel = y * width + x
        inside = x < width and y < height
        ray, gradient = wp.vec3(), Feature()
        transmittance, final_transmittance = float(1.0), float(1.0)
        alpha_gradient, inverse_gradient = float(0.0), float(0.0)
        n_contributors = int(0)
        if inside:
            ray = rays[pixel] + wp.vec3()
            transmittance = final_t[pixel] + 0.0
            final_transmittance = transmittance
            gradient = gimage[pixel] + Feature()
            alpha_gradient, inverse_gradient = (
                galpha[pixel] + 0.0,
                ginverse[pixel] + 0.0,
            )
            n_contributors = contributors[pixel] + 0
            # A constant background needs no gradient. Updating this single
            # address from every pixel otherwise serializes the whole image.
            if need_background:
                wp.atomic_add(gbackground, 0, gradient * transmittance)
        background_gradient = wp.dot(background[0], gradient)
        tail, last_color = Feature(), Feature()
        last_alpha, tail_inverse, last_inverse = float(0.0), float(0.0), float(0.0)
        ray_gradient = wp.vec3()
        first = ranges[tile, 0]
        # Every lane participates, including padded image edges. Each tile
        # fetches candidates once, then all pixels traverse that shared batch.
        count = wp.tile_max(wp.tile(n_contributors))[0]
        shared_p = wp.tile_empty(shape=(256,), dtype=wp.vec3, storage="shared")
        shared_m = wp.tile_empty(shape=(256,), dtype=wp.mat33, storage="shared")
        shared_o = wp.tile_empty(shape=(256,), dtype=float, storage="shared")
        shared_b = wp.tile_empty(shape=(256,), dtype=wp.vec4, storage="shared")
        shared_id = wp.tile_empty(shape=(256,), dtype=int, storage="shared")
        for end in range(first + count - 1, first - 1, -256):
            index = end - lane
            gid = int(0)
            position, matrix, opacity, bound = (
                wp.vec3(),
                wp.mat33(),
                float(0.0),
                wp.vec4(),
            )
            if index >= first:
                gid = ids[index]
                position = positions[gid] + wp.vec3()
                matrix = matrices[gid] + wp.mat33()
                opacity = opacities[gid] + 0.0
                bound = pbf[gid] + wp.vec4()
            wp.tile_scatter_masked(shared_p, lane, position, index >= first)
            wp.tile_scatter_masked(shared_m, lane, matrix, index >= first)
            wp.tile_scatter_masked(shared_o, lane, opacity, index >= first)
            wp.tile_scatter_masked(shared_b, lane, bound, index >= first)
            wp.tile_scatter_masked(shared_id, lane, gid, index >= first)
            for j in range(wp.min(256, end - first + 1)):
                active = inside and end - j - first < n_contributors
                if active and _inside_pbf(ray, shared_b[j], mode):
                    gid = shared_id[j]
                    matrix, position = shared_m[j] + wp.mat33(), shared_p[j] + wp.vec3()
                    p, d = matrix * position, matrix * ray
                    normal = wp.cross(d, p)
                    denominator = wp.dot(d, d)
                    power = -0.5 * wp.dot(normal, normal) / denominator
                    gaussian = wp.exp(power)
                    alpha = wp.min(0.99, shared_o[j] * gaussian)
                    if power <= 0.0 and alpha >= 1.0 / 255.0:
                        transmittance = transmittance / (1.0 - alpha)
                        value = features[gid]
                        tail = last_alpha * last_color + (1.0 - last_alpha) * tail
                        inv_depth = 1.0 / depths[gid]
                        tail_inverse = (
                            last_alpha * last_inverse
                            + (1.0 - last_alpha) * tail_inverse
                        )
                        g_alpha = transmittance * (
                            wp.dot(value - tail, gradient)
                            + (inv_depth - tail_inverse) * inverse_gradient
                        )
                        g_alpha += (
                            final_transmittance
                            / (1.0 - alpha)
                            * (alpha_gradient - background_gradient)
                        )
                        last_alpha, last_color, last_inverse = alpha, value, inv_depth
                        weight = alpha * transmittance
                        wp.atomic_add(gf, gid, gradient * weight)
                        wp.atomic_add(
                            gd, gid, -weight * inverse_gradient * inv_depth * inv_depth
                        )
                        wp.atomic_add(go, gid, gaussian * g_alpha)
                        g_power = shared_o[j] * gaussian * g_alpha
                        g_normal = normal * (-g_power / denominator)
                        g_denom = -g_power * power / denominator
                        g_p = wp.cross(g_normal, d)
                        g_d = wp.cross(p, g_normal) + 2.0 * g_denom * d
                        wp.atomic_add(gp, gid, wp.transpose(matrix) * g_p)
                        wp.atomic_add(
                            gm, gid, wp.outer(g_p, position) + wp.outer(g_d, ray)
                        )
                        wp.atomic_add(
                            absolute,
                            gid,
                            wp.vec3(wp.abs(g_p[0]), wp.abs(g_p[1]), wp.abs(g_p[2])),
                        )
                        ray_gradient += wp.transpose(matrix) * g_d
            wp.tile_sum(wp.tile(int(inside)))
        if inside:
            grays[pixel] = ray_gradient

    return backward


@wp.kernel(enable_backward=False)
def _densification(
    matrices: wp.array(dtype=wp.mat33),
    depths: wp.array(dtype=float),
    absolute: wp.array(dtype=wp.vec3),
    proxy: wp.array(dtype=wp.vec2),
):
    i = wp.tid()
    value = (
        wp.transpose(matrices[i]) * absolute[i] * (depths[i] * depths[i] + 1.0) * 0.3
    )
    proxy[i] = wp.vec2(value[0], value[1])


def _args(values, channels, bindings):
    Feature = wp.types.vector(channels, wp.float32)
    types = (
        wp.vec3,
        wp.mat33,
        wp.float32,
        wp.float32,
        Feature,
        wp.vec3,
        wp.vec4,
        wp.int32,
        wp.int32,
        Feature,
    )
    return [kernel_arg(v, t, bindings) for v, t in zip(values, types)]


class _Pixels(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        positions,
        matrices,
        opacities,
        depths,
        features,
        rays,
        pbf,
        ranges,
        ids,
        background,
        screen,
        width,
        height,
        mode,
        grad_enabled,
        absolute_target,
    ):
        channels = features.shape[1]
        n_pixels = width * height
        image = features.new_empty((n_pixels, channels))
        alpha, inverse, transmittance = [features.new_empty(n_pixels) for _ in range(3)]
        contributors = features.new_empty(n_pixels, dtype=torch.int32)
        values = (
            positions,
            matrices,
            opacities,
            depths,
            features,
            rays,
            pbf,
            ranges,
            ids,
            background,
        )
        bindings = FrameBindings(launch_cache)
        launch_tiled(
            forward_kernel(channels),
            dim=(len(ranges),),
            inputs=[width, height, mode] + _args(values, channels, bindings),
            outputs=[
                kernel_arg(image, wp.types.vector(channels, wp.float32), bindings),
                kernel_arg(alpha, bindings=bindings),
                kernel_arg(inverse, bindings=bindings),
                kernel_arg(transmittance, bindings=bindings),
                kernel_arg(contributors, bindings=bindings),
            ],
            stream=current_stream(features.device, bindings=bindings),
            block_dim=256,
            bindings=bindings,
        )
        if grad_enabled and any(ctx.needs_input_grad[:11]):
            ctx.save_for_backward(*values, transmittance, contributors)
            ctx.options = width, height, mode, channels
            ctx.absolute_target = absolute_target
        ctx.set_materialize_grads(False)
        return image, alpha, inverse

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, image_gradient, alpha_gradient, inverse_gradient):
        *values, transmittance, contributors = ctx.saved_tensors
        p, m, o, d, f, rays, pbf, ranges, ids, bg = values
        width, height, mode, channels = ctx.options
        seeds = [
            g.contiguous() if g is not None else f.new_zeros(shape)
            for g, shape in zip(
                (image_gradient, alpha_gradient, inverse_gradient),
                ((width * height, channels), (width * height,), (width * height,)),
            )
        ]
        gradients = [torch.zeros_like(v) for v in (p, m, o, d, f)]
        ray_gradient = torch.empty_like(rays)
        background_gradient = torch.zeros_like(bg)
        absolute = torch.zeros_like(p)
        proxy = f.new_empty((len(p), 2))
        bindings = FrameBindings(launch_cache)
        Feature = wp.types.vector(channels, wp.float32)
        launch_tiled(
            backward_kernel(channels),
            dim=(len(ranges),),
            inputs=[width, height, mode, ctx.needs_input_grad[9]]
            + _args(values, channels, bindings)
            + [
                kernel_arg(transmittance, bindings=bindings),
                kernel_arg(contributors, bindings=bindings),
            ]
            + [
                kernel_arg(g, t, bindings)
                for g, t in zip(seeds, (Feature, wp.float32, wp.float32))
            ],
            outputs=[
                kernel_arg(g, t, bindings)
                for g, t in zip(
                    gradients + [ray_gradient, background_gradient, absolute],
                    (
                        wp.vec3,
                        wp.mat33,
                        wp.float32,
                        wp.float32,
                        Feature,
                        wp.vec3,
                        Feature,
                        wp.vec3,
                    ),
                )
            ],
            stream=current_stream(f.device, bindings=bindings),
            block_dim=256,
            bindings=bindings,
        )
        if len(p):
            launch(
                _densification,
                dim=len(p),
                inputs=[
                    kernel_arg(m, wp.mat33, bindings),
                    kernel_arg(d, bindings=bindings),
                    kernel_arg(absolute, wp.vec3, bindings),
                ],
                outputs=[kernel_arg(proxy, wp.vec2, bindings)],
                stream=current_stream(f.device, bindings=bindings),
                bindings=bindings,
            )
        all_gradients = gradients + [
            ray_gradient,
            None,
            None,
            None,
            background_gradient,
            proxy,
        ]
        if ctx.absolute_target is not None:
            ctx.absolute_target.write(proxy.abs())
        return (
            *[
                g if need else None
                for g, need in zip(all_gradients, ctx.needs_input_grad[:11])
            ],
            None,
            None,
            None,
            None,
            None,
        )


def render(
    positions: Tensor,
    matrices: Tensor,
    opacities: Tensor,
    depths: Tensor,
    features: Tensor,
    rays: Tensor,
    bounds: ParticleBounds,
    bins: TileBins,
    background: Tensor,
    screen: Tensor,
    width: int,
    height: int,
    mode: int,
    absolute_target: AbsGradientTarget | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    return _Pixels.apply(
        positions,
        matrices,
        opacities,
        depths,
        features,
        rays,
        bounds.tangents,
        bins["ranges"],
        bins["point_list"],
        background,
        screen,
        width,
        height,
        mode,
        torch.is_grad_enabled(),
        absolute_target,
    )
