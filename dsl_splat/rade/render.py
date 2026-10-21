"""RaDe-GS tile-cooperative RGB/features, depth and normal rendering.

Geometry shares the same contributor traversal as color. Gradients replay that
traversal in reverse; median selection and early termination are discrete.
"""

from dataclasses import dataclass
from functools import cache
import torch
from torch import Tensor
import warp as wp
from .types import ProjectedGaussians
from ..primitives.blocks import (
    shared_ids,
    shared_centers,
    shared_conics,
    synchronize,
    all_done,
)
from ..primitives.warp import any_active, sum_vector_lanes
from ..primitives.tiling import TileBins
from ..runtime.dispatch import FrameBindings, kernel_arg, launch
from ..runtime.interop import current_stream, contiguous_features


@dataclass(slots=True)
class RenderedPixels:
    features: Tensor
    auxiliary: Tensor
    contributors: Tensor

    def images(self) -> tuple[Tensor, ...]:
        """Feature, alpha, expected depth, median depth and normal HWC views."""
        image = self.features.permute(1, 2, 0)
        return (
            image[..., :-6],
            image[..., -6:-5],
            image[..., -5:-4],
            image[..., -4:-3],
            image[..., -3:],
        )


@wp.func
def pixel_gradient(
    gradient: wp.array3d(dtype=float), x: int, y: int, channel: int
) -> float:
    # Autograd passes None for unused outputs. A null array has shape zero.
    if gradient.shape[0] > 0:
        return gradient[y, x, channel]
    return 0.0


@cache
def _shared_attributes(channels: int):
    # Channel-major storage makes cooperative writes contiguous. Native code
    # only allocates storage; loading, traversal and reduction stay in Warp.
    # Keep the ordinary 48 KiB static shared-memory budget for arbitrary D.
    # Wide features retain direct global reads, with no public channel limit.
    cached_channels = channels if channels <= 32 else 1
    Feature = wp.types.vector(channels, wp.float32)

    @wp.func_native(
        f"__shared__ float values[{256 * cached_channels}]; "
        f"return wp::array_t<float>(values, {256 * cached_channels});"
    )
    def shared_features() -> wp.array(dtype=float): ...

    @wp.func_native(
        "__shared__ wp::vec_t<3,float> values[512]; "
        "return wp::array_t<wp::vec_t<3,float>>(values, 512);"
    )
    def shared_geometry() -> wp.array(dtype=wp.vec3): ...

    @wp.func
    def load_features(
        values: wp.array(dtype=float),
        features: wp.array(dtype=Feature),
        ids: wp.array(dtype=int),
        index: int,
    ) -> Feature:
        if wp.static(channels > 32):
            return features[ids[index]]
        result = Feature()
        for channel in range(wp.static(channels)):
            read_ch = wp.static(channel)
            result[read_ch] = values[read_ch * 256 + index]
        return result

    return shared_features, shared_geometry, load_features


@cache
def _make_forward(channels: int) -> wp.Kernel:
    Feature = wp.types.vector(channels, wp.float32)
    shared_features, shared_geometry, load_features = _shared_attributes(channels)

    @wp.kernel(
        module="unique",
        enable_backward=False,
        launch_bounds=256,
        module_options={"max_unroll": 0, "fast_math": True},
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
        auxiliary: wp.array(dtype=float),
        contributors: wp.array(dtype=int),
        planes: wp.array(dtype=wp.vec3),
        normals: wp.array(dtype=wp.vec3),
        intrinsics: wp.array2d(dtype=float),
    ):
        tid = wp.tid()
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
        loaded_features, loaded_geometry = shared_features(), shared_geometry()
        transmittance = float(1.0)
        depth, median = float(0.0), float(0.0)
        normal = wp.vec3(0.0)
        median_contributor = int(-1)
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
                if wp.static(channels <= 32):
                    value = features[gaussian]
                    for channel in range(wp.static(channels)):
                        cache_ch = wp.static(channel)
                        loaded_features[cache_ch * 256 + lane] = value[cache_ch]
                loaded_geometry[lane] = planes[gaussian]
                loaded_geometry[256 + lane] = normals[gaussian]
            synchronize()
            for j in range(wp.min(256, last - begin)):
                if done:
                    break
                contributor += 1
                delta = loaded_centers[j] - wp.vec2(float(x), float(y))
                conic = loaded_conics[j]
                power = (
                    -0.5
                    * (conic[0] * delta[0] * delta[0] + conic[2] * delta[1] * delta[1])
                    - conic[1] * delta[0] * delta[1]
                )
                if power <= 0.0:
                    alpha = wp.min(0.99, conic[3] * wp.exp(power))
                    if alpha >= 1.0 / 255.0:
                        next_t = transmittance * (1.0 - alpha)
                        if next_t < 0.0001:
                            done = True
                        else:
                            value = load_features(
                                loaded_features, features, loaded_ids, j
                            )
                            for channel in range(wp.static(channels)):
                                blend_ch = wp.static(channel)
                                color[blend_ch] = (
                                    color[blend_ch]
                                    + value[blend_ch] * alpha * transmittance
                                )
                            plane = loaded_geometry[j]
                            distance = (
                                plane[0] * delta[0] + plane[1] * delta[1] + plane[2]
                            )
                            weight = alpha * transmittance
                            depth += distance * weight
                            normal += loaded_geometry[256 + j] * weight
                            if transmittance > 0.5:
                                median = distance
                                median_contributor = contributor
                            transmittance = next_t
                            last_contributor = contributor
        if inside:
            for channel in range(wp.static(channels)):
                output_ch = wp.static(channel)
                image[output_ch * width * height + pixel] = (
                    color[output_ch] + transmittance * background[output_ch]
                )
            count = width * height
            u = (float(x) + 0.5 - intrinsics[0, 2]) / intrinsics[0, 0]
            v = (float(y) + 0.5 - intrinsics[1, 2]) / intrinsics[1, 1]
            rln = 1.0 / wp.sqrt(u * u + v * v + 1.0)
            alpha = 1.0 - transmittance
            expected = float(0.0)
            if last_contributor > 0:
                expected = depth * rln / alpha
            length = wp.length(normal)
            unit = normal / wp.max(length, 1.0e-12)
            image[wp.static(channels) * count + pixel] = alpha
            image[wp.static(channels + 1) * count + pixel] = expected
            image[wp.static(channels + 2) * count + pixel] = median * rln
            image[wp.static(channels + 3) * count + pixel] = unit[0]
            image[wp.static(channels + 4) * count + pixel] = unit[1]
            image[wp.static(channels + 5) * count + pixel] = unit[2]
            auxiliary[pixel] = depth * rln
            auxiliary[count + pixel] = length
            contributors[count + pixel] = median_contributor
            contributors[pixel] = last_contributor

    return render


@cache
def _make_backward(channels: int, absolute: bool) -> wp.Kernel:
    Feature = wp.types.vector(channels, wp.float32)
    shared_features, shared_geometry, load_features = _shared_attributes(channels)

    @wp.kernel(
        module="unique",
        enable_backward=False,
        launch_bounds=256,
        module_options={"max_unroll": 0, "fast_math": True},
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
        image: wp.array(dtype=float),
        auxiliary: wp.array(dtype=float),
        planes: wp.array(dtype=wp.vec3),
        normals: wp.array(dtype=wp.vec3),
        intrinsics: wp.array2d(dtype=float),
        contributors: wp.array(dtype=int),
        upstream_features: wp.array3d(dtype=float),
        upstream_alpha: wp.array3d(dtype=float),
        upstream_depth: wp.array3d(dtype=float),
        upstream_median: wp.array3d(dtype=float),
        upstream_normals: wp.array3d(dtype=float),
        grad_centers: wp.array(dtype=wp.vec2),
        grad_conics: wp.array(dtype=wp.vec4),
        grad_features: wp.array(dtype=Feature),
        abs_centers: wp.array(dtype=wp.vec2),
        grad_planes: wp.array(dtype=wp.vec3),
        grad_normals: wp.array(dtype=wp.vec3),
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
        loaded_features, loaded_geometry = shared_features(), shared_geometry()
        transmittance, final_transmittance = float(0.0), float(0.0)
        last_contributor, median_contributor = int(0), int(-1)
        depth_gradient, median_gradient = float(0.0), float(0.0)
        normal_gradient = wp.vec3(0.0)
        accumulated_normal, last_normal = wp.vec3(0.0), wp.vec3(0.0)
        accumulated_depth, last_depth = float(0.0), float(0.0)
        count = width * height
        gradient, accumulated, last_color = Feature(), Feature(), Feature()
        last_alpha, background_gradient = float(0.0), float(0.0)
        if inside:
            final_transmittance = 1.0 - image[wp.static(channels) * count + pixel]
            transmittance = final_transmittance
            last_contributor = contributors[pixel]
            for channel in range(wp.static(channels)):
                initial_ch = wp.static(channel)
                gradient[initial_ch] = pixel_gradient(
                    upstream_features, x, y, initial_ch
                )
                background_gradient += background[initial_ch] * gradient[initial_ch]
            background_gradient -= pixel_gradient(upstream_alpha, x, y, 0)
            median_contributor = contributors[count + pixel]
            alpha = image[wp.static(channels) * count + pixel]
            if alpha > 0.0:
                u = (float(x) + 0.5 - intrinsics[0, 2]) / intrinsics[0, 0]
                v = (float(y) + 0.5 - intrinsics[1, 2]) / intrinsics[1, 1]
                rln = 1.0 / wp.sqrt(u * u + v * v + 1.0)
                gd = pixel_gradient(upstream_depth, x, y, 0)
                background_gradient += gd * (auxiliary[pixel] / alpha) / alpha
                depth_gradient = gd / alpha * rln
                median_gradient = pixel_gradient(upstream_median, x, y, 0) * rln
                normal_gradient = wp.vec3(
                    pixel_gradient(upstream_normals, x, y, 0),
                    pixel_gradient(upstream_normals, x, y, 1),
                    pixel_gradient(upstream_normals, x, y, 2),
                )
                unit = wp.vec3(
                    image[wp.static(channels + 3) * count + pixel],
                    image[wp.static(channels + 4) * count + pixel],
                    image[wp.static(channels + 5) * count + pixel],
                )
                length = auxiliary[count + pixel]
                if length >= 1.0e-12:
                    normal_gradient -= unit * wp.dot(unit, normal_gradient)
                normal_gradient /= wp.max(length, 1.0e-12)
        contributor = last - first
        for begin in range(0, last - first, 256):
            synchronize()
            index = last - begin - lane - 1
            if index >= first:
                gaussian = ids[index]
                loaded_ids[lane] = gaussian
                loaded_centers[lane] = centers[gaussian]
                loaded_conics[lane] = conics[gaussian]
                if wp.static(channels <= 32):
                    value = features[gaussian]
                    for channel in range(wp.static(channels)):
                        cache_ch = wp.static(channel)
                        loaded_features[cache_ch * 256 + lane] = value[cache_ch]
                loaded_geometry[lane] = planes[gaussian]
                loaded_geometry[256 + lane] = normals[gaussian]
            synchronize()
            for j in range(wp.min(256, last - first - begin)):
                contributor -= 1
                delta = loaded_centers[j] - wp.vec2(float(x), float(y))
                conic = loaded_conics[j]
                power = (
                    -0.5
                    * (conic[0] * delta[0] * delta[0] + conic[2] * delta[1] * delta[1])
                    - conic[1] * delta[0] * delta[1]
                )
                gaussian_value = wp.exp(power)
                alpha = wp.min(0.99, conic[3] * gaussian_value)
                valid = (
                    inside
                    and contributor < last_contributor
                    and power <= 0.0
                    and alpha >= 1.0 / 255.0
                )
                if not any_active(valid):
                    continue
                # Every lane participates in reductions, including padded pixels
                # and rejected contributors. Absgrad sums per-pixel magnitudes.
                local_features, local_planes, local_normals = (
                    Feature(),
                    wp.vec3(0.0),
                    wp.vec3(0.0),
                )
                local_conics, local_centers, local_absolute = (
                    wp.vec4(0.0),
                    wp.vec2(0.0),
                    wp.vec2(0.0),
                )
                if valid:
                    transmittance = transmittance / (1.0 - alpha)
                    grad_alpha = float(0.0)
                    value = load_features(loaded_features, features, loaded_ids, j)
                    for channel in range(wp.static(channels)):
                        blend_ch = wp.static(channel)
                        color = value[blend_ch]
                        accumulated[blend_ch] = (
                            last_alpha * last_color[blend_ch]
                            + (1.0 - last_alpha) * accumulated[blend_ch]
                        )
                        last_color[blend_ch] = color
                        grad_alpha += (color - accumulated[blend_ch]) * gradient[
                            blend_ch
                        ]
                        local_features[blend_ch] = (
                            alpha * transmittance * gradient[blend_ch]
                        )
                    plane = loaded_geometry[j]
                    distance = plane[0] * delta[0] + plane[1] * delta[1] + plane[2]
                    normal = loaded_geometry[256 + j]
                    accumulated_depth = (
                        last_alpha * last_depth + (1.0 - last_alpha) * accumulated_depth
                    )
                    accumulated_normal = (
                        last_alpha * last_normal
                        + (1.0 - last_alpha) * accumulated_normal
                    )
                    grad_alpha += (
                        distance - accumulated_depth
                    ) * depth_gradient + wp.dot(
                        normal - accumulated_normal, normal_gradient
                    )
                    last_depth, last_normal = distance, normal
                    weight = alpha * transmittance
                    g_distance = weight * depth_gradient
                    if contributor == median_contributor - 1:
                        g_distance += median_gradient
                    local_planes = wp.vec3(
                        g_distance * delta[0], g_distance * delta[1], g_distance
                    )
                    local_normals = normal_gradient * weight
                    grad_alpha = (
                        grad_alpha * transmittance
                        + (-final_transmittance / (1.0 - alpha)) * background_gradient
                    )
                    last_alpha = alpha
                    grad_value = conic[3] * grad_alpha
                    gdx, gdy = gaussian_value * delta[0], gaussian_value * delta[1]
                    dx = grad_value * (-gdx * conic[0] - gdy * conic[1])
                    dy = grad_value * (-gdy * conic[2] - gdx * conic[1])
                    dx += g_distance * plane[0]
                    dy += g_distance * plane[1]
                    local_centers = wp.vec2(dx, dy)
                    if wp.static(absolute):
                        local_absolute = wp.vec2(wp.abs(dx), wp.abs(dy))
                    local_conics = wp.vec4(
                        -0.5 * gdx * delta[0] * grad_value,
                        -gdx * delta[1] * grad_value,
                        -0.5 * gdy * delta[1] * grad_value,
                        gaussian_value * grad_alpha,
                    )
                local_features = sum_vector_lanes(local_features)
                local_planes = sum_vector_lanes(local_planes)
                local_normals = sum_vector_lanes(local_normals)
                local_conics = sum_vector_lanes(local_conics)
                local_centers = sum_vector_lanes(local_centers)
                if wp.static(absolute):
                    local_absolute = sum_vector_lanes(local_absolute)
                if lane % 32 == 0:
                    gaussian = loaded_ids[j]
                    wp.atomic_add(grad_features, gaussian, local_features)
                    wp.atomic_add(grad_planes, gaussian, local_planes)
                    wp.atomic_add(grad_normals, gaussian, local_normals)
                    wp.atomic_add(grad_conics, gaussian, local_conics)
                    wp.atomic_add(grad_centers, gaussian, local_centers)
                    if wp.static(absolute):
                        wp.atomic_add(abs_centers, gaussian, local_absolute)

    return backward


@torch.no_grad()
def render(
    state: ProjectedGaussians,
    bins: TileBins,
    features: Tensor,
    background: Tensor,
    intrinsics: Tensor,
    width: int,
    height: int,
    bindings: FrameBindings,
) -> RenderedPixels:
    channels = features.shape[-1]
    result = RenderedPixels(
        features.new_empty((channels + 6, height, width)),
        features.new_empty((2, height, width)),
        torch.empty((2, height, width), dtype=torch.int32, device=features.device),
    )
    launch(
        _make_forward(channels),
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
            kernel_arg(result.features.view(-1), bindings=bindings),
            kernel_arg(result.auxiliary.view(-1), bindings=bindings),
            kernel_arg(result.contributors.view(-1), bindings=bindings),
            kernel_arg(state.ray_planes, wp.vec3, bindings),
            kernel_arg(state.normals, wp.vec3, bindings),
            kernel_arg(intrinsics, bindings=bindings),
        ],
        outputs=[],
        stream=current_stream(features.device, bindings=bindings),
        bindings=bindings,
    )
    return result


@torch.no_grad()
def backward(
    state: ProjectedGaussians,
    bins: TileBins,
    features: Tensor,
    background: Tensor,
    intrinsics: Tensor,
    output: RenderedPixels,
    gradients: tuple[Tensor | None, ...],
    width: int,
    height: int,
    bindings: FrameBindings,
    *,
    absgrad: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    # Vector atomics need contiguous channel storage even when the caller's
    # feature tensor is a transposed view.
    centers, conics, colors, planes, normals = (
        torch.zeros_like(t, memory_format=torch.contiguous_format)
        for t in (
            state.centers,
            state.conics,
            features,
            state.ray_planes,
            state.normals,
        )
    )
    absolute = (
        torch.zeros_like(state.centers) if absgrad else state.centers.new_empty((0, 2))
    )
    channels = features.shape[-1]
    launch(
        _make_backward(channels, absgrad),
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
            kernel_arg(output.features.contiguous().view(-1), bindings=bindings),
            kernel_arg(output.auxiliary.view(-1), bindings=bindings),
            kernel_arg(state.ray_planes, wp.vec3, bindings),
            kernel_arg(state.normals, wp.vec3, bindings),
            kernel_arg(intrinsics, bindings=bindings),
            kernel_arg(output.contributors.view(-1), bindings=bindings),
            *(kernel_arg(g, bindings=bindings) for g in gradients),
        ],
        outputs=[
            kernel_arg(centers, wp.vec2, bindings),
            kernel_arg(conics, wp.vec4, bindings),
            kernel_arg(colors, wp.types.vector(channels, wp.float32), bindings),
            kernel_arg(absolute, wp.vec2, bindings),
            kernel_arg(planes, wp.vec3, bindings),
            kernel_arg(normals, wp.vec3, bindings),
        ],
        stream=current_stream(features.device, bindings=bindings),
        bindings=bindings,
    )
    return centers, conics, colors, planes, normals, absolute if absgrad else None
