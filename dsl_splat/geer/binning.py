"""Exact GEER particle bounding frusta, pixel bounds and radial-distance sorting.

Ported from the official GEER forward.cu/rasterizer_impl.cu; see README.md.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
import warp as wp

from .cameras import RayCamera
from .geometry import launch_cache
from ..primitives.tiling import TileBins, scan_counts, sort_key, sort_tile_pairs
from ..runtime.dispatch import FrameBindings, kernel_arg, launch
from ..runtime.interop import current_stream


@dataclass(slots=True)
class ParticleBounds:
    tangents: Tensor
    pixels: Tensor
    radii: Tensor
    tile_counts: Tensor
    pixel_radii: Tensor


@wp.func
def _axis_bounds(mu: float, z: float, xx: float, xz: float, zz: float, limit: float):
    c22 = 9.0 * zz - z * z
    c02 = 9.0 * xz - mu * z
    c00 = 9.0 * xx - mu * mu
    disc = c02 * c02 - c22 * c00
    valid = c22 != 0.0 and disc >= 0.0
    lo, hi = float(0.0), float(0.0)
    if valid:
        center = c02 / c22
        half = wp.sqrt(disc) / wp.abs(c22)
        lo, hi = center - half, center + half
        a, b = wp.sqrt(1.0 + lo * lo), wp.sqrt(1.0 + hi * hi)
        values = wp.vec4(lo / (1.0 + a), lo / (1.0 - a), hi / (1.0 + b), hi / (1.0 - b))
        direction = mu / wp.max(z, 1.0e-6)
        target = direction / (1.0 + wp.sqrt(1.0 + direction * direction))
        lower, upper = float(-wp.inf), float(wp.inf)
        lower_index, upper_index = int(-1), int(-1)
        for j in range(4):
            if values[j] < target and values[j] >= lower:
                lower, lower_index = values[j], j
            if values[j] > target and values[j] <= upper:
                upper, upper_index = values[j], j
        fov = limit / (1.0 + wp.sqrt(1.0 + limit * limit))
        original_lo, original_hi = lo, hi
        if lower < -fov:
            lo = -limit
        elif lower > fov:
            lo = limit
        elif lower_index >= 0:
            lo = original_lo if lower_index < 2 else original_hi
        if upper < -fov:
            hi = -limit
        elif upper > fov:
            hi = limit
        elif upper_index >= 0:
            hi = original_lo if upper_index < 2 else original_hi
    return lo, hi, valid


@wp.func
def _lower_bound(values: wp.array(dtype=float), target: float):
    low, high = int(0), values.shape[0]
    while low < high:
        mid = (low + high) // 2
        if values[mid] < target:
            low = mid + 1
        else:
            high = mid
    return low


@wp.func
def _kb_uv(
    x: float, y: float, K: wp.array2d(dtype=float), coeffs: wp.array(dtype=float)
):
    radius = wp.sqrt(x * x + y * y)
    angle = wp.atan(radius)
    a2 = angle * angle
    factor = (
        angle
        * (
            1.0
            + a2 * (coeffs[0] + a2 * (coeffs[1] + a2 * (coeffs[2] + a2 * coeffs[3])))
        )
        / wp.max(radius, 1.0e-9)
    )
    return wp.vec2(K[0, 0] * x * factor + K[0, 2], K[1, 1] * y * factor + K[1, 2])


@wp.func
def _kb_bounds(
    b: wp.vec4,
    K: wp.array2d(dtype=float),
    coeffs: wp.array(dtype=float),
    width: int,
    height: int,
):
    lb, lt = _kb_uv(b[0], b[2], K, coeffs), _kb_uv(b[0], b[3], K, coeffs)
    rb, rt = _kb_uv(b[1], b[2], K, coeffs), _kb_uv(b[1], b[3], K, coeffs)
    lm, rm = _kb_uv(b[0], 0.0, K, coeffs), _kb_uv(b[1], 0.0, K, coeffs)
    mb, mt = _kb_uv(0.0, b[2], K, coeffs), _kb_uv(0.0, b[3], K, coeffs)
    x0, x1, y0, y1 = lm[0], rm[0], mb[1], mt[1]
    if (b[1] < 0.0 and b[2] > 0.0) or (b[0] > 0.0 and b[3] < 0.0):
        x0, x1, y0, y1 = lb[0], rt[0], lb[1], rt[1]
    elif (b[1] < 0.0 and b[3] < 0.0) or (b[0] > 0.0 and b[2] > 0.0):
        x0, x1, y0, y1 = lt[0], rb[0], rb[1], lt[1]
    elif b[0] < 0.0 and b[1] > 0.0 and b[2] > 0.0:
        x0, x1, y0, y1 = lb[0], rb[0], wp.min(lb[1], rb[1]), mt[1]
    elif b[0] < 0.0 and b[1] > 0.0 and b[3] < 0.0:
        x0, x1, y0, y1 = lt[0], rt[0], mb[1], wp.max(lt[1], rt[1])
    elif b[2] < 0.0 and b[3] > 0.0 and b[1] < 0.0:
        x0, x1, y0, y1 = lm[0], wp.max(rb[0], rt[0]), rb[1], rt[1]
    elif b[2] < 0.0 and b[3] > 0.0 and b[0] > 0.0:
        x0, x1, y0, y1 = wp.min(lb[0], lt[0]), rm[0], lb[1], lt[1]
    return wp.vec4i(
        wp.clamp(int(wp.floor(x0)), 0, width - 1),
        wp.clamp(int(wp.floor(x1)), 0, width - 1),
        wp.clamp(int(wp.floor(y0)), 0, height - 1),
        wp.clamp(int(wp.floor(y1)), 0, height - 1),
    )


@wp.func
def tile_bounds(pixels: wp.vec4i, gx: int, gy: int):
    return wp.vec4i(
        wp.clamp(int(float(pixels[0]) / 16.0), 0, gx),
        wp.clamp(int(float(pixels[2]) / 16.0), 0, gy),
        wp.clamp(int(float(pixels[1] + 15) / 16.0), 0, gx),
        wp.clamp(int(float(pixels[3] + 15) / 16.0), 0, gy),
    )


@wp.kernel(enable_backward=False)
def _bounds(
    positions: wp.array(dtype=wp.vec3),
    matrices: wp.array(dtype=wp.mat33),
    opacities: wp.array(dtype=float),
    width: int,
    height: int,
    near: float,
    far: float,
    mode: int,
    limits: wp.array(dtype=float),
    K: wp.array2d(dtype=float),
    tx: wp.array(dtype=float),
    ty: wp.array(dtype=float),
    coeffs: wp.array(dtype=float),
    tangents: wp.array(dtype=wp.vec4),
    pixels: wp.array(dtype=wp.vec4i),
    radii: wp.array(dtype=int),
    counts: wp.array(dtype=int),
    pixel_radii: wp.array(dtype=wp.vec2),
):
    i = wp.tid()
    radii[i] = 0
    counts[i] = 0
    pixel_radii[i] = wp.vec2()
    p = positions[i]
    if p[2] <= near or p[2] >= far or opacities[i] < 1.0 / 255.0:
        return
    inv = wp.inverse(matrices[i])
    cov = inv * wp.transpose(inv)
    x0, x1, vx = _axis_bounds(p[0], p[2], cov[0, 0], cov[0, 2], cov[2, 2], limits[0])
    y0, y1, vy = _axis_bounds(p[1], p[2], cov[1, 1], cov[1, 2], cov[2, 2], limits[1])
    if not vx or not vy or (x1 - x0) * (y1 - y0) == 0.0:
        return
    tangent = wp.vec4(x0, x1, y0, y1)
    rect = wp.vec4i()
    if mode == 0:
        rect = wp.vec4i(
            _lower_bound(tx, x0),
            _lower_bound(tx, x1),
            _lower_bound(ty, y0),
            _lower_bound(ty, y1),
        )
    elif mode == 1:
        rect = _kb_bounds(tangent, K, coeffs, width, height)
    else:
        rect = wp.vec4i(
            wp.clamp(int(wp.floor(K[0, 0] * x0 + K[0, 2] - 1.0)), -width, 2 * width),
            wp.clamp(int(wp.ceil(K[0, 0] * x1 + K[0, 2] + 1.0)), -width, 2 * width),
            wp.clamp(int(wp.floor(K[1, 1] * y0 + K[1, 2] - 1.0)), -height, 2 * height),
            wp.clamp(int(wp.ceil(K[1, 1] * y1 + K[1, 2] + 1.0)), -height, 2 * height),
        )
    b = tile_bounds(rect, (width + 15) // 16, (height + 15) // 16)
    count = (b[2] - b[0]) * (b[3] - b[1])
    if count <= 0 or (rect[1] - rect[0]) * (rect[3] - rect[2]) == 0:
        return
    tangents[i] = tangent
    pixels[i] = rect
    radii[i] = wp.max(b[2] - b[0], b[3] - b[1])
    counts[i] = count
    pixel_radii[i] = wp.vec2(float(rect[1] - rect[0]), float(rect[3] - rect[2])) * 0.5


@wp.kernel(enable_backward=False)
def _duplicate(
    pixels: wp.array(dtype=wp.vec4i),
    radii: wp.array(dtype=int),
    depths: wp.array(dtype=float),
    offsets: wp.array(dtype=int),
    gx: int,
    gy: int,
    keys: wp.array(dtype=wp.uint64),
    ids: wp.array(dtype=int),
):
    i = wp.tid()
    if radii[i] > 0:
        b = tile_bounds(pixels[i], gx, gy)
        offset = int(0)
        if i > 0:
            offset = offsets[i - 1]
        for y in range(b[1], b[3]):
            for x in range(b[0], b[2]):
                keys[offset] = sort_key(y * gx + x, depths[i])
                ids[offset] = i
                offset += 1


@torch.no_grad()
def bound_gaussians(
    positions: Tensor,
    matrices: Tensor,
    opacities: Tensor,
    camera: RayCamera,
    mode: int,
    limits: Tensor,
    bindings: FrameBindings | None = None,
) -> ParticleBounds:
    """Bound the canonical-space Gaussians for a prepared ray camera."""
    if bindings is None:
        bindings = FrameBindings(launch_cache)
    n = len(positions)
    bounds = ParticleBounds(
        positions.new_empty((n, 4)),
        positions.new_empty((n, 4), dtype=torch.int32),
        positions.new_empty(n, dtype=torch.int32),
        positions.new_empty(n, dtype=torch.int32),
        positions.new_empty((n, 2)),
    )
    if n:
        launch(
            _bounds,
            dim=n,
            inputs=[
                kernel_arg(positions, wp.vec3, bindings),
                kernel_arg(matrices, wp.mat33, bindings),
                kernel_arg(opacities, bindings=bindings),
                camera.width,
                camera.height,
                camera.near_plane,
                camera.far_plane,
                mode,
                kernel_arg(limits, bindings=bindings),
                kernel_arg(camera.K, bindings=bindings),
                kernel_arg(camera.tan_theta, bindings=bindings),
                kernel_arg(camera.tan_phi, bindings=bindings),
                kernel_arg(camera.distortion_coeffs, bindings=bindings),
            ],
            outputs=[
                kernel_arg(bounds.tangents, wp.vec4, bindings),
                kernel_arg(bounds.pixels, wp.vec4i, bindings),
                kernel_arg(bounds.radii, bindings=bindings),
                kernel_arg(bounds.tile_counts, bindings=bindings),
                kernel_arg(bounds.pixel_radii, wp.vec2, bindings),
            ],
            stream=current_stream(positions.device, bindings=bindings),
            bindings=bindings,
        )
    return bounds


@torch.no_grad()
def bin_and_sort(
    bounds: ParticleBounds,
    depths: Tensor,
    width: int,
    height: int,
    bindings: FrameBindings | None = None,
) -> TileBins:
    """Duplicate into PBF tiles and stably sort by tile/radial distance."""
    if bindings is None:
        bindings = FrameBindings(launch_cache)
    stream = current_stream(depths.device, bindings=bindings)
    offsets, count = scan_counts(bounds.tile_counts, stream)
    keys = torch.empty(2 * count, dtype=torch.int64, device=depths.device)
    ids = torch.empty(2 * count, dtype=torch.int32, device=depths.device)
    gx, gy = (width + 15) // 16, (height + 15) // 16
    ranges = torch.zeros((gx * gy, 2), dtype=torch.int32, device=depths.device)
    if count:
        launch(
            _duplicate,
            dim=len(depths),
            inputs=[
                kernel_arg(bounds.pixels, wp.vec4i, bindings),
                kernel_arg(bounds.radii, bindings=bindings),
                kernel_arg(depths, bindings=bindings),
                kernel_arg(offsets, bindings=bindings),
                gx,
                gy,
            ],
            outputs=[
                kernel_arg(keys[:count], wp.uint64, bindings),
                kernel_arg(ids[:count], bindings=bindings),
            ],
            stream=stream,
            bindings=bindings,
        )
        sort_tile_pairs(keys, ids, count, ranges, stream, bindings)
    return dict(point_list=ids[:count], ranges=ranges)
