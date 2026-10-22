"""SnugBox/AccuTile intersections for alpha-thresholded projected Gaussians.

Adapted from Speedy-Splat's auxiliary.h, rasterizer b38abb6da7a7217999f8c5ec0e9219284464ffe0.
Traversal follows the shorter bounding-box axis. Counting and emission use the
same function, so prefix-scan allocations and written key counts cannot drift.
See ../NOTICE.md for source and Gaussian-Splatting license attribution.
"""

# Copyright (C) 2023, Inria / GRAPHDECO research group.
# Adapted from Speedy-Splat's Inria-derived rasterizer; the Gaussian-Splatting
# research-use license and attribution are retained in ../LICENSE.md and NOTICE.md.

from functools import cache
from typing import Literal

import warp as wp

from .geometry import tile_bounds
from ..primitives.tiling import sort_key

TileMode = Literal["radius", "snugbox", "accutile"]


def validate_tile_mode(mode: TileMode) -> None:
    if mode not in ("radius", "snugbox", "accutile"):
        raise ValueError("tile_mode must be 'radius', 'snugbox' or 'accutile'")


@wp.struct
class TilePairs:
    # A default-constructed/null array selects counting without device storage.
    keys: wp.array(dtype=wp.uint64)
    ids: wp.array(dtype=int)


@wp.struct
class Ellipse:
    center: wp.vec2
    conic: wp.vec3
    level: float
    determinant: float
    extent: wp.vec2
    bounds: wp.vec4i
    precise: bool


@wp.func
def footprint(
    center: wp.vec2, conic: wp.vec4, radius: int, gx: int, gy: int
) -> Ellipse:
    result = Ellipse()
    result.center = center
    result.conic = wp.vec3(conic[0], conic[1], conic[2])
    result.bounds = tile_bounds(center, radius, gx, gy)
    a, b, c = conic[0], conic[1], conic[2]
    det = float(wp.float64(a) * wp.float64(c) - wp.float64(b) * wp.float64(b))
    if conic[3] < 1.0 / 255.0:
        result.bounds = wp.vec4i(0)
        return result
    # Ill-conditioned/invalid ellipses retain the original conservative square.
    if not (a > 0.0 and c > 0.0 and det > 0.0):
        return result
    level = 2.0 * wp.log(conic[3] * 255.0)
    # Bound float32 quadratic/exp rounding over the original square, including
    # its boundary tiles. This retains pixels accepted by the scalar renderer
    # near the threshold, especially for long, rotated, thin ellipses.
    reach = float(radius) + 32.0
    roundoff = float(8.0 * 1.1920928955078125e-7) * (
        (a + 2.0 * wp.abs(b) + c) * reach * reach + wp.abs(level) + 1.0
    )
    result.level = wp.max(0.0, level) + roundoff
    result.determinant = det
    result.extent = wp.vec2(
        wp.sqrt(result.level * c / det), wp.sqrt(result.level * a / det)
    )
    if not (wp.isfinite(result.extent[0]) and wp.isfinite(result.extent[1])):
        return result
    # Clamp in floating point before converting to int, including huge extents.
    lo_x = int(
        wp.clamp(
            wp.floor((center[0] - result.extent[0]) / 16.0),
            0.0,
            float(gx),
        )
    )
    lo_y = int(
        wp.clamp(
            wp.floor((center[1] - result.extent[1]) / 16.0),
            0.0,
            float(gy),
        )
    )
    hi_x = int(
        wp.clamp(
            wp.floor((center[0] + result.extent[0]) / 16.0) + 1.0,
            0.0,
            float(gx),
        )
    )
    hi_y = int(
        wp.clamp(
            wp.floor((center[1] + result.extent[1]) / 16.0) + 1.0,
            0.0,
            float(gy),
        )
    )
    # Never introduce contributors outside original 3DGS's candidate tiles.
    result.bounds = wp.vec4i(
        wp.max(result.bounds[0], lo_x),
        wp.max(result.bounds[1], lo_y),
        wp.min(result.bounds[2], hi_x),
        wp.min(result.bounds[3], hi_y),
    )
    result.precise = True
    return result


@wp.func
def line_intersection(e: Ellipse, line: int, is_y: bool) -> wp.vec2:
    u, v = e.center[0], e.center[1]
    b, c = e.conic[1], e.conic[2]
    extent_u = e.extent[0]
    if is_y:
        u, v = v, u
        c = e.conic[0]
        extent_u = e.extent[1]
    h = wp.clamp(float(line * 16) - u, -extent_u, extent_u)
    root = wp.sqrt(wp.max(0.0, e.level * c - e.determinant * h * h))
    return wp.vec2(v + (-b * h - root) / c, v + (-b * h + root) / c)


@wp.func
def strip_span(
    e: Ellipse, row: int, is_y: bool, low: wp.vec2, high: wp.vec2
) -> wp.vec2i:
    # Use (u,v) = (y,x) for horizontal strips, (x,y) otherwise.
    u, v = e.center[0], e.center[1]
    a, b = e.conic[0], e.conic[1]
    extent = e.extent[1]
    lower, upper = e.bounds[1], e.bounds[3]
    if is_y:
        u, v = v, u
        a = e.conic[2]
        extent = e.extent[0]
        lower, upper = e.bounds[0], e.bounds[2]
    lo, hi = wp.min(low[0], high[0]), wp.max(low[1], high[1])
    # Include v extrema lying strictly between the strip's boundary lines.
    arg_min, arg_max = b * extent / a, -b * extent / a
    h0, h1 = float(row * 16) - u, float((row + 1) * 16) - u
    if h0 <= arg_min and arg_min <= h1:
        lo = v - extent
    if h0 <= arg_max and arg_max <= h1:
        hi = v + extent
    lo_tile = int(wp.clamp(wp.floor(lo / 16.0), float(lower), float(upper)))
    hi_tile = int(wp.clamp(wp.floor(hi / 16.0) + 1.0, float(lower), float(upper)))
    return wp.vec2i(lo_tile, hi_tile)


@cache
def make_intersector(mode: TileMode):
    validate_tile_mode(mode)

    @wp.func
    def visit(
        center: wp.vec2,
        conic: wp.vec4,
        radius: int,
        gx: int,
        gy: int,
        gaussian: int,
        depth: float,
        offset: int,
        output: TilePairs,
    ) -> int:
        e = Ellipse()
        e.bounds = tile_bounds(center, radius, gx, gy)
        if wp.static(mode != "radius"):
            e = footprint(center, conic, radius, gx, gy)
        bounds = e.bounds
        nx, ny = wp.max(0, bounds[2] - bounds[0]), wp.max(0, bounds[3] - bounds[1])
        if nx == 0 or ny == 0:
            return 0
        is_y = bool(True)
        if wp.static(mode == "accutile"):
            is_y = ny < nx
        start, end = bounds[1], bounds[3]
        if not is_y:
            start, end = bounds[0], bounds[2]
        count = int(0)
        # Box-only counting is constant time. Emission and AccuTile counting
        # share the exact same strip bounds and rounding.
        if wp.static(mode != "accutile"):
            if output.keys.shape[0] == 0:
                return nx * ny
        low = wp.vec2(0.0)
        if wp.static(mode == "accutile"):
            if e.precise:
                low = line_intersection(e, start, is_y)
        for row in range(start, end):
            span = wp.vec2i(bounds[0], bounds[2])
            if not is_y:
                span = wp.vec2i(bounds[1], bounds[3])
            if wp.static(mode == "accutile"):
                if e.precise:
                    high = line_intersection(e, row + 1, is_y)
                    span = strip_span(e, row, is_y, low, high)
                    low = high
            if output.keys.shape[0] > 0:
                for col in range(span[0], span[1]):
                    tile = col * gx + row
                    if is_y:
                        tile = row * gx + col
                    output.keys[offset + count + col - span[0]] = sort_key(tile, depth)
                    output.ids[offset + count + col - span[0]] = gaussian
            count += wp.max(0, span[1] - span[0])
        return count

    return visit
