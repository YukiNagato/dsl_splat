"""Degree 0--3 spherical harmonics, matching AAA forward_common.h."""

import warp as wp
from functools import cache


@wp.func
def basis(direction: wp.vec3, coefficient: int):
    """One real SH basis function; Warp differentiates this local polynomial."""
    x, y, z = direction[0], direction[1], direction[2]
    if coefficient == 0:
        return 0.28209479177387814
    if coefficient == 1:
        return -0.4886025119029199 * y
    if coefficient == 2:
        return 0.4886025119029199 * z
    if coefficient == 3:
        return -0.4886025119029199 * x
    if coefficient == 4:
        return 1.0925484305920792 * x * y
    if coefficient == 5:
        return -1.0925484305920792 * y * z
    if coefficient == 6:
        return 0.31539156525252005 * (2.0 * z * z - x * x - y * y)
    if coefficient == 7:
        return -1.0925484305920792 * x * z
    if coefficient == 8:
        return 0.5462742152960396 * (x * x - y * y)
    if coefficient == 9:
        return -0.5900435899266435 * y * (3.0 * x * x - y * y)
    if coefficient == 10:
        return 2.890611442640554 * x * y * z
    if coefficient == 11:
        return -0.4570457994644658 * y * (4.0 * z * z - x * x - y * y)
    if coefficient == 12:
        return 0.3731763325901154 * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y)
    if coefficient == 13:
        return -0.4570457994644658 * x * (4.0 * z * z - x * x - y * y)
    if coefficient == 14:
        return 1.445305721320277 * z * (x * x - y * y)
    return -0.5900435899266435 * x * (x * x - 3.0 * y * y)


@cache
def forward_kernel(degree: int, masked: bool, save_clamp: bool) -> wp.Kernel:
    count = (degree + 1) ** 2

    @wp.kernel(
        module="unique", enable_backward=False, module_options={"fuse_fp": False}
    )
    def evaluate(
        coefficients: wp.array3d(dtype=float),
        directions: wp.array2d(dtype=float),
        radii: wp.array(dtype=int),
        colors: wp.array(dtype=wp.vec3),
        active_channels: wp.array2d(dtype=bool),
    ):
        i = wp.tid()
        visible = bool(True)
        if wp.static(masked):
            visible = radii[i] > 0
        value = wp.vec3(0.0)
        if visible:
            unit = wp.vec3(0.0)
            if wp.static(degree > 0):
                d = wp.vec3(directions[i, 0], directions[i, 1], directions[i, 2])
                unit = d / wp.max(wp.length(d), 1.0e-12)
            for k in range(wp.static(count)):
                coefficient = wp.vec3(
                    coefficients[i, k, 0], coefficients[i, k, 1], coefficients[i, k, 2]
                )
                value += basis(unit, k) * coefficient
            value += wp.vec3(0.5)
        colors[i] = wp.max(value, wp.vec3(0.0))
        if wp.static(save_clamp):
            for ch in range(3):
                active_channels[i, ch] = visible and value[ch] >= 0.0

    return evaluate


@cache
def backward_kernel(
    degree: int, masked: bool, need_coefficients: bool, need_directions: bool
) -> wp.Kernel:
    count = (degree + 1) ** 2

    @wp.kernel(
        module="unique", enable_backward=False, module_options={"fuse_fp": False}
    )
    def backward(
        coefficients: wp.array3d(dtype=float),
        directions: wp.array2d(dtype=float),
        radii: wp.array(dtype=int),
        active_channels: wp.array2d(dtype=bool),
        upstream: wp.array2d(dtype=float),
        grad_coefficients: wp.array3d(dtype=float),
        grad_directions: wp.array(dtype=wp.vec3),
    ):
        i = wp.tid()
        visible = bool(True)
        if wp.static(masked):
            visible = radii[i] > 0
        direction_gradient = wp.vec3(0.0)
        if visible:
            unit, color_gradient = wp.vec3(0.0), wp.vec3(0.0)
            length = float(1.0)
            for ch in range(3):
                if active_channels[i, ch]:
                    color_gradient[ch] = upstream[i, ch]
            if wp.static(degree > 0):
                d = wp.vec3(directions[i, 0], directions[i, 1], directions[i, 2])
                length = wp.length(d)
                unit = d / wp.max(length, 1.0e-12)
            unit_gradient = wp.vec3(0.0)
            for k in range(wp.static(count)):
                if wp.static(need_coefficients):
                    coefficient_gradient = basis(unit, k) * color_gradient
                    for ch in range(3):
                        grad_coefficients[i, k, ch] = coefficient_gradient[ch]
                if wp.static(need_directions):
                    coefficient = wp.vec3(
                        coefficients[i, k, 0],
                        coefficients[i, k, 1],
                        coefficients[i, k, 2],
                    )
                    basis_gradient, unused = wp.grad(basis)(unit, k)
                    unit_gradient += (
                        wp.dot(coefficient, color_gradient) * basis_gradient
                    )
            if wp.static(need_directions):
                direction_gradient = unit_gradient / wp.max(length, 1.0e-12)
                if length >= 1.0e-12:
                    direction_gradient -= unit * (wp.dot(unit, unit_gradient) / length)
        if wp.static(need_directions):
            grad_directions[i] = direction_gradient

    return backward


@wp.func
def evaluate_sh(
    i: int, degree: int, mean: wp.vec3, camera: wp.vec3, sh: wp.array2d(dtype=wp.vec3)
):
    d = (mean - camera) / wp.length(mean - camera)
    x, y, z = d[0], d[1], d[2]
    result = 0.28209479177387814 * sh[i, 0]
    if degree > 0:
        result = (
            result
            - 0.4886025119029199 * y * sh[i, 1]
            + 0.4886025119029199 * z * sh[i, 2]
            - 0.4886025119029199 * x * sh[i, 3]
        )
        if degree > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + 1.0925484305920792 * xy * sh[i, 4]
                - 1.0925484305920792 * yz * sh[i, 5]
                + 0.31539156525252005 * (2.0 * zz - xx - yy) * sh[i, 6]
                - 1.0925484305920792 * xz * sh[i, 7]
                + 0.5462742152960396 * (xx - yy) * sh[i, 8]
            )
            if degree > 2:
                result = (
                    result
                    - 0.5900435899266435 * y * (3.0 * xx - yy) * sh[i, 9]
                    + 2.890611442640554 * xy * z * sh[i, 10]
                    - 0.4570457994644658 * y * (4.0 * zz - xx - yy) * sh[i, 11]
                    + 0.3731763325901154
                    * z
                    * (2.0 * zz - 3.0 * xx - 3.0 * yy)
                    * sh[i, 12]
                    - 0.4570457994644658 * x * (4.0 * zz - xx - yy) * sh[i, 13]
                    + 1.445305721320277 * z * (xx - yy) * sh[i, 14]
                    - 0.5900435899266435 * x * (xx - 3.0 * yy) * sh[i, 15]
                )
    return result + wp.vec3(0.5)
