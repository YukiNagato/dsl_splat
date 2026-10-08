"""AAA geometry from consistent_common.cuh and stopthepop_common.cuh.

Matrices here have ordinary mathematical rows. The CUDA GLM helpers store the
transpose, so g2s[col] in CUDA corresponds to g2s[row] here.
"""

import warp as wp
from .native_geometry import (
    frustum_depth_native,
    bound_axis_native,
    screen_bounds_native,
)


@wp.func
def rotation_matrix(q: wp.vec4):
    r, x, y, z = q[0], q[1], q[2], q[3]
    return wp.mat33(
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + r * z),
        2.0 * (x * z - r * y),
        2.0 * (x * y - r * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + r * x),
        2.0 * (x * z + r * y),
        2.0 * (y * z - r * x),
        1.0 - 2.0 * (x * x + y * y),
    )


@wp.func
def load_camera(a: wp.array2d(dtype=float)):
    return wp.mat44(
        a[0, 0],
        a[1, 0],
        a[2, 0],
        a[3, 0],
        a[0, 1],
        a[1, 1],
        a[2, 1],
        a[3, 1],
        a[0, 2],
        a[1, 2],
        a[2, 2],
        a[3, 2],
        a[0, 3],
        a[1, 3],
        a[2, 3],
        a[3, 3],
    )


@wp.func
def frustum_minimum(lo: wp.vec2, hi: wp.vec2, g2s: wp.mat44):
    return frustum_depth_native(g2s, lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1], False)[
        0
    ]


@wp.func
def bound_axis(g2v: wp.mat44, mean: wp.vec3, cutoff: float, axis: int):
    return bound_axis_native(g2v, mean, cutoff, axis)


@wp.func
def screen_bounds(g2s: wp.mat44, cutoff: float):
    bounds = screen_bounds_native(g2s, cutoff)
    return (
        bounds[2] >= 0.0,
        wp.vec2(bounds[0], bounds[1]),
        wp.vec2(bounds[2], bounds[3]),
    )


@wp.func
def tile_minimum_2d(co: wp.vec4, mean: wp.vec2, lo: wp.vec2, hi: wp.vec2):
    xdiff, ydiff = lo[0] - mean[0], lo[1] - mean[1]
    left, above = float(xdiff > 0.0), float(ydiff > 0.0)
    notx = left + float(mean[0] > hi[0])
    noty = above + float(mean[1] > hi[1])
    result = float(0.0)
    if notx + noty > 0.0:
        px = left * lo[0] + (1.0 - left) * hi[0]
        py = above * lo[1] + (1.0 - above) * hi[1]
        dx, dy = wp.copysign(15.0, xdiff), wp.copysign(15.0, ydiff)
        diffx, diffy = mean[0] - px, mean[1] - py
        tx = noty * wp.clamp(
            (dx * co[0] * diffx + dx * co[1] * diffy) * (1.0 / (225.0 * co[0])),
            0.0,
            1.0,
        )
        ty = notx * wp.clamp(
            (dy * co[1] * diffx + dy * co[2] * diffy) * (1.0 / (225.0 * co[2])),
            0.0,
            1.0,
        )
        mx, my = mean[0] - (px + tx * dx), mean[1] - (py + ty * dy)
        result = 0.5 * (co[0] * mx * mx + co[2] * my * my) + co[1] * mx * my
    return result
