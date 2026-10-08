"""Analytic 3D ray gradients and buffers for hierarchical backward replay.

Follows AAA/StopThePop hierarchical_render.cuh (MIT, Graz UT, 2024).
Sorting/culling are held fixed; alpha's 0.99 clamp uses CUDA's straight-through
gradient. Equations use Python/Warp; a small primitive specifies global-memory
atomics for the gradient buffers.
"""

from functools import cache

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
    screen_gradient: wp.array(dtype=float)
    screen_abs_gradient: wp.array(dtype=float)


@cache
def _make_accumulate_ray_gradient(
    screen_grad: bool = False, absgrad: bool = False
) -> wp.Function:
    """Specialize optional screen-space statistics out of the ordinary backward."""

    @wp.func
    def accumulate(
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
        screen_x, screen_y = float(0.0), float(0.0)
        for index in range(3):
            axis = wp.static(index)
            derivative = factor * (
                ratio * wp.vec4(dd0[axis], dd1[axis], 0.0, dd3[axis])
                - wp.vec4(dm0[axis], dm1[axis], 0.0, dm3[axis]) * inverse_dd
            )
            if wp.static(screen_grad):
                screen_x += derivative[0] * g[3, axis]
                screen_y += derivative[1] * g[3, axis]
            global_atomic_add(
                context.transform_gradient, id * 16 + axis * 4, derivative[0]
            )
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
        if wp.static(screen_grad):
            # A rigid footprint translation adds delta_x*g[3] to g[0], and
            # delta_y*g[3] to g[1]. Contract this pixel's transform derivative
            # before any reduction; abs(sum(pixel_grad)) would lose AbsGS information.
            screen_x += translation[0] * g[3, 3]
            screen_y += translation[1] * g[3, 3]
            global_atomic_add(context.screen_gradient, id * 2, screen_x)
            global_atomic_add(context.screen_gradient, id * 2 + 1, screen_y)
            if wp.static(absgrad):
                global_atomic_add(context.screen_abs_gradient, id * 2, wp.abs(screen_x))
                global_atomic_add(
                    context.screen_abs_gradient, id * 2 + 1, wp.abs(screen_y)
                )

    return accumulate
