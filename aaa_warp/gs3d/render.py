"""Tile-cooperative 3DGS alpha blending and reverse traversal in Warp."""

from dataclasses import dataclass
from functools import cache

import torch
from torch import Tensor
import warp as wp

from .projection import ProjectedFootprint
from ..primitives.blocks import (
    shared_ids,
    shared_centers,
    shared_conics,
    synchronize,
    all_done,
)
from ..primitives.tiling import TileBins
from ..runtime.dispatch import FrameBindings, kernel_arg, launch
from ..runtime.interop import current_stream, contiguous_features
from ..statistics import (
    DeviceStatistics,
    PreparedStatistics,
    StatisticsKey,
    make_observer,
)


@dataclass(slots=True)
class RenderedPixels:
    features: Tensor
    transmittance: Tensor
    contributors: Tensor


@cache
def _make_forward(channels: int, statistics_key: StatisticsKey = ()) -> wp.Kernel:
    Feature = wp.types.vector(channels, wp.float32)
    observe = make_observer(statistics_key)

    @wp.func
    def evaluate(
        tid: int,
        width: int,
        height: int,
        ranges: wp.array2d(dtype=int),
        ids: wp.array(dtype=int),
        centers: wp.array(dtype=wp.vec2),
        conics: wp.array(dtype=wp.vec4),
        features: wp.array(dtype=Feature),
        background: wp.array(dtype=float),
        image: wp.array(dtype=float),
        final_t: wp.array(dtype=float),
        contributors: wp.array(dtype=int),
        statistics: DeviceStatistics,
    ):
        tile, lane = tid // 256, tid % 256
        gx = (width + 15) // 16
        x, y = tile % gx * 16 + lane % 16, tile // gx * 16 + lane // 16
        pixel = y * width + x
        inside = x < width and y < height
        done = not inside
        first, last = ranges[tile, 0], ranges[tile, 1]
        loaded_ids, loaded_centers, loaded_conics = (
            shared_ids(),
            shared_centers(),
            shared_conics(),
        )
        transmittance = float(1.0)
        color = Feature()
        contributor, last_contributor = int(0), int(0)
        for begin in range(first, last, 256):
            if all_done(done):
                break
            index = begin + lane
            if index < last:
                gaussian = ids[index]
                loaded_ids[lane] = gaussian
                loaded_centers[lane] = centers[gaussian]
                loaded_conics[lane] = conics[gaussian]
            synchronize()
            for j in range(wp.min(256, last - begin)):
                if not done:
                    contributor += 1
                    delta = loaded_centers[j] - wp.vec2(float(x), float(y))
                    conic = loaded_conics[j]
                    power = (
                        -0.5
                        * (
                            conic[0] * delta[0] * delta[0]
                            + conic[2] * delta[1] * delta[1]
                        )
                        - conic[1] * delta[0] * delta[1]
                    )
                    if power <= 0.0:
                        alpha = wp.min(0.99, conic[3] * wp.exp(power))
                        if alpha >= 1.0 / 255.0:
                            next_t = transmittance * (1.0 - alpha)
                            if next_t < 0.0001:
                                done = True
                            else:
                                gaussian = loaded_ids[j]
                                value = features[gaussian]
                                for channel in range(wp.static(channels)):
                                    ch = wp.static(channel)
                                    color[ch] = (
                                        color[ch] + value[ch] * alpha * transmittance
                                    )
                                if wp.static(len(statistics_key) > 0):
                                    observe(
                                        statistics,
                                        gaussian,
                                        pixel,
                                        alpha * transmittance,
                                    )
                                transmittance = next_t
                                last_contributor = contributor
        if inside:
            for channel in range(wp.static(channels)):
                ch = wp.static(channel)
                image[ch * width * height + pixel] = (
                    color[ch] + transmittance * background[ch]
                )
            final_t[pixel] = transmittance
            contributors[pixel] = last_contributor

    if statistics_key:

        @wp.kernel(
            module="unique",
            enable_backward=False,
            launch_bounds=256,
            module_options={"max_unroll": 0},
        )
        def render(
            width: int,
            height: int,
            ranges: wp.array2d(dtype=int),
            ids: wp.array(dtype=int),
            centers: wp.array(dtype=wp.vec2),
            conics: wp.array(dtype=wp.vec4),
            features: wp.array(dtype=Feature),
            background: wp.array(dtype=float),
            image: wp.array(dtype=float),
            final_t: wp.array(dtype=float),
            contributors: wp.array(dtype=int),
            pixel_values: wp.array2d(dtype=float),
            counts: wp.array2d(dtype=int),
            sums: wp.array2d(dtype=float),
        ):
            statistics = DeviceStatistics()
            statistics.pixel_values = pixel_values
            statistics.counts = counts
            statistics.sums = sums
            evaluate(
                wp.tid(),
                width,
                height,
                ranges,
                ids,
                centers,
                conics,
                features,
                background,
                image,
                final_t,
                contributors,
                statistics,
            )

    else:

        @wp.kernel(
            module="unique",
            enable_backward=False,
            launch_bounds=256,
            module_options={"max_unroll": 0},
        )
        def render(
            width: int,
            height: int,
            ranges: wp.array2d(dtype=int),
            ids: wp.array(dtype=int),
            centers: wp.array(dtype=wp.vec2),
            conics: wp.array(dtype=wp.vec4),
            features: wp.array(dtype=Feature),
            background: wp.array(dtype=float),
            image: wp.array(dtype=float),
            final_t: wp.array(dtype=float),
            contributors: wp.array(dtype=int),
        ):
            evaluate(
                wp.tid(),
                width,
                height,
                ranges,
                ids,
                centers,
                conics,
                features,
                background,
                image,
                final_t,
                contributors,
                DeviceStatistics(),
            )

    return render


@cache
def _make_backward(channels: int, absolute: bool, constant_tail: int = 0) -> wp.Kernel:
    Feature = wp.types.vector(channels, wp.float32)

    @wp.kernel(
        module="unique",
        enable_backward=False,
        launch_bounds=256,
        module_options={"max_unroll": 0},
    )
    def backward(
        width: int,
        height: int,
        ranges: wp.array2d(dtype=int),
        ids: wp.array(dtype=int),
        centers: wp.array(dtype=wp.vec2),
        conics: wp.array(dtype=wp.vec4),
        features: wp.array(dtype=Feature),
        background: wp.array(dtype=float),
        final_t: wp.array(dtype=float),
        contributors: wp.array(dtype=int),
        upstream: wp.array(dtype=float),
        grad_centers: wp.array2d(dtype=float),
        grad_conics: wp.array2d(dtype=float),
        grad_features: wp.array2d(dtype=float),
        abs_centers: wp.array2d(dtype=float),
    ):
        tid = wp.tid()
        tile, lane = tid // 256, tid % 256
        gx = (width + 15) // 16
        x, y = tile % gx * 16 + lane % 16, tile // gx * 16 + lane // 16
        pixel = y * width + x
        inside = x < width and y < height
        first, last = ranges[tile, 0], ranges[tile, 1]
        loaded_ids, loaded_centers, loaded_conics = (
            shared_ids(),
            shared_centers(),
            shared_conics(),
        )
        transmittance, final_transmittance = float(0.0), float(0.0)
        last_contributor = int(0)
        gradient, accumulated, last_color = Feature(), Feature(), Feature()
        last_alpha, background_gradient = float(0.0), float(0.0)
        if inside:
            final_transmittance = final_t[pixel]
            transmittance = final_transmittance
            last_contributor = contributors[pixel]
            for channel in range(wp.static(channels)):
                initial_ch = wp.static(channel)
                gradient[initial_ch] = upstream[initial_ch * width * height + pixel]
                background_gradient += background[initial_ch] * gradient[initial_ch]
        contributor = last - first
        for begin in range(0, last - first, 256):
            synchronize()
            index = last - begin - lane - 1
            if index >= first:
                gaussian = ids[index]
                loaded_ids[lane] = gaussian
                loaded_centers[lane] = centers[gaussian]
                loaded_conics[lane] = conics[gaussian]
            synchronize()
            for j in range(wp.min(256, last - first - begin)):
                if inside:
                    contributor -= 1
                    if contributor < last_contributor:
                        delta = loaded_centers[j] - wp.vec2(float(x), float(y))
                        conic = loaded_conics[j]
                        power = (
                            -0.5
                            * (
                                conic[0] * delta[0] * delta[0]
                                + conic[2] * delta[1] * delta[1]
                            )
                            - conic[1] * delta[0] * delta[1]
                        )
                        if power <= 0.0:
                            gaussian_value = wp.exp(power)
                            alpha = wp.min(0.99, conic[3] * gaussian_value)
                            if alpha >= 1.0 / 255.0:
                                transmittance = transmittance / (1.0 - alpha)
                                gaussian = loaded_ids[j]
                                color = features[gaussian]
                                grad_alpha = float(0.0)
                                for channel in range(wp.static(channels)):
                                    ch = wp.static(channel)
                                    accumulated[ch] = (
                                        last_alpha * last_color[ch]
                                        + (1.0 - last_alpha) * accumulated[ch]
                                    )
                                    last_color[ch] = color[ch]
                                    grad_alpha += (
                                        color[ch] - accumulated[ch]
                                    ) * gradient[ch]
                                    # Constant attributes still contribute to
                                    # opacity/geometry gradients (e.g. alpha).
                                    if wp.static(ch < channels - constant_tail):
                                        wp.atomic_add(
                                            grad_features,
                                            gaussian,
                                            ch,
                                            alpha * transmittance * gradient[ch],
                                        )
                                grad_alpha = (
                                    grad_alpha * transmittance
                                    + (-final_transmittance / (1.0 - alpha))
                                    * background_gradient
                                )
                                last_alpha = alpha
                                grad_value = conic[3] * grad_alpha
                                gdx, gdy = (
                                    gaussian_value * delta[0],
                                    gaussian_value * delta[1],
                                )
                                dx = grad_value * (-gdx * conic[0] - gdy * conic[1])
                                dy = grad_value * (-gdy * conic[2] - gdx * conic[1])
                                wp.atomic_add(grad_centers, gaussian, 0, dx)
                                wp.atomic_add(grad_centers, gaussian, 1, dy)
                                if wp.static(absolute):
                                    wp.atomic_add(abs_centers, gaussian, 0, wp.abs(dx))
                                    wp.atomic_add(abs_centers, gaussian, 1, wp.abs(dy))
                                wp.atomic_add(
                                    grad_conics,
                                    gaussian,
                                    0,
                                    -0.5 * gdx * delta[0] * grad_value,
                                )
                                wp.atomic_add(
                                    grad_conics,
                                    gaussian,
                                    1,
                                    -gdx * delta[1] * grad_value,
                                )
                                wp.atomic_add(
                                    grad_conics,
                                    gaussian,
                                    2,
                                    -0.5 * gdy * delta[1] * grad_value,
                                )
                                wp.atomic_add(
                                    grad_conics,
                                    gaussian,
                                    3,
                                    gaussian_value * grad_alpha,
                                )

    return backward


@torch.no_grad()
def render(
    state: ProjectedFootprint,
    bins: TileBins,
    features: Tensor,
    background: Tensor,
    width: int,
    height: int,
    bindings: FrameBindings,
    statistics: PreparedStatistics | None = None,
) -> RenderedPixels:
    channels = features.shape[-1]
    output = RenderedPixels(
        features.new_empty((channels, height, width)),
        features.new_empty((height, width)),
        torch.empty((height, width), dtype=torch.int32, device=features.device),
    )
    arguments = [
        width,
        height,
        kernel_arg(bins["ranges"], bindings=bindings),
        kernel_arg(bins["point_list"], bindings=bindings),
        kernel_arg(state.centers, wp.vec2, bindings),
        kernel_arg(state.conics, wp.vec4, bindings),
        kernel_arg(
            contiguous_features(features),
            wp.types.vector(channels, wp.float32),
            bindings,
        ),
        kernel_arg(background, bindings=bindings),
        kernel_arg(output.features.view(-1), bindings=bindings),
        kernel_arg(output.transmittance.view(-1), bindings=bindings),
        kernel_arg(output.contributors.view(-1), bindings=bindings),
    ]
    if statistics is not None:
        arguments.extend(
            kernel_arg(value, bindings=bindings)
            for value in (statistics.pixel_values, statistics.counts, statistics.sums)
        )
    launch(
        _make_forward(channels, () if statistics is None else statistics.key),
        dim=256 * len(bins["ranges"]),
        inputs=arguments,
        outputs=[],
        stream=current_stream(features.device, bindings=bindings),
        bindings=bindings,
    )
    return output


@torch.no_grad()
def backward(
    state: ProjectedFootprint,
    bins: TileBins,
    features: Tensor,
    background: Tensor,
    final_t: Tensor,
    contributors: Tensor,
    gradient: Tensor,
    width: int,
    height: int,
    bindings: FrameBindings,
    *,
    absgrad: bool = False,
    constant_tail: int = 0,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    if not 0 <= constant_tail <= features.shape[-1]:
        raise ValueError("constant_tail must be within the feature channel count")
    centers, conics, colors = (
        torch.zeros_like(state.centers),
        torch.zeros_like(state.conics),
        torch.zeros_like(features),
    )
    absolute = (
        torch.zeros_like(state.centers) if absgrad else state.centers.new_empty((0, 2))
    )
    channels = features.shape[-1]
    launch(
        _make_backward(channels, absgrad, constant_tail),
        dim=256 * len(bins["ranges"]),
        inputs=[
            width,
            height,
            kernel_arg(bins["ranges"], bindings=bindings),
            kernel_arg(bins["point_list"], bindings=bindings),
            kernel_arg(state.centers, wp.vec2, bindings),
            kernel_arg(state.conics, wp.vec4, bindings),
            kernel_arg(
                contiguous_features(features),
                wp.types.vector(channels, wp.float32),
                bindings,
            ),
            kernel_arg(background, bindings=bindings),
            kernel_arg(final_t.view(-1), bindings=bindings),
            kernel_arg(contributors.view(-1), bindings=bindings),
            kernel_arg(gradient.contiguous().view(-1), bindings=bindings),
        ],
        outputs=[
            kernel_arg(value, bindings=bindings)
            for value in (centers, conics, colors, absolute)
        ],
        stream=current_stream(features.device, bindings=bindings),
        bindings=bindings,
    )
    return centers, conics, colors, absolute if absgrad else None
