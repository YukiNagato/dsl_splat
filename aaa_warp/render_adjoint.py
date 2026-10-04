"""Analytic 3D ray gradients and buffers for hierarchical backward replay.

Follows AAA/StopThePop hierarchical_render.cuh (MIT, Graz UT, 2024).
Sorting/culling are held fixed; alpha's 0.99 clamp uses CUDA's straight-through
gradient. Equations use Python/Warp; a small primitive specifies global-memory
atomics for the gradient buffers.
"""

import warp as wp

from .backward_intrinsics import global_atomic_add
from .render_warp_geometry import dot3, reciprocal
from .render_warp_intrinsics import load_matrix


@wp.struct
class RenderAdjoint:
    transforms: wp.array(dtype=wp.mat44)
    opacity: wp.array(dtype=float)
    background: wp.array(dtype=float)
    pixel_gradient: wp.array(dtype=float)
    transform_gradient: wp.array(dtype=float)
    opacity_gradient: wp.array(dtype=float)
    color_gradient: wp.array(dtype=float)


@wp.func
def accumulate_ray_gradient(
    context: RenderAdjoint,
    id: int,
    x: float,
    y: float,
    gaussian: float,
    alpha_gradient: float,
):
    g = load_matrix(context.transforms, id)
    a = g[0] - g[3] * x
    b = g[1] - g[3] * y
    nx = wp.vec3(a[0], a[1], a[2])
    ny = wp.vec3(b[0], b[1], b[2])
    dx, dy = a[3], b[3]
    moment = dx * ny - dy * nx
    direction = wp.cross(nx, ny)
    dd, mm = dot3(direction, direction), dot3(moment, moment)
    xx, yy, xy = dot3(nx, nx), dot3(ny, ny), dot3(nx, ny)
    ym, xm = dot3(ny, moment), dot3(nx, moment)
    dd0 = yy * nx - xy * ny
    dd1 = xx * ny - xy * nx
    dd3 = xy * (y * nx + x * ny) - x * yy * nx - y * xx * ny
    dm0, dm1 = -dy * moment, dx * moment
    dm3 = (dy * x - dx * y) * moment
    factor = context.opacity[id] * alpha_gradient * gaussian
    inverse_dd = reciprocal(dd)
    ratio = mm / dd * inverse_dd
    # Warp rows correspond to the GLM columns loaded by CUDA. CUDA writes
    # four float4 blocks; the screen-depth component (column 2) stays zero.
    for index in range(3):
        axis = wp.static(index)
        derivative = factor * (
            ratio * wp.vec4(dd0[axis], dd1[axis], 0.0, dd3[axis])
            - wp.vec4(dm0[axis], dm1[axis], 0.0, dm3[axis]) * inverse_dd
        )
        global_atomic_add(context.transform_gradient, id * 16 + axis * 4, derivative[0])
        global_atomic_add(
            context.transform_gradient, id * 16 + axis * 4 + 1, derivative[1]
        )
        global_atomic_add(
            context.transform_gradient, id * 16 + axis * 4 + 3, derivative[3]
        )
    translation = factor * -wp.vec4(ym, -xm, 0.0, y * xm - x * ym) * inverse_dd
    global_atomic_add(context.transform_gradient, id * 16 + 12, translation[0])
    global_atomic_add(context.transform_gradient, id * 16 + 13, translation[1])
    global_atomic_add(context.transform_gradient, id * 16 + 15, translation[3])
    global_atomic_add(context.opacity_gradient, id, gaussian * alpha_gradient)
