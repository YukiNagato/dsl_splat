"""AAA CUDA-compatible analytic 3D preprocessing adjoint, in Python/Warp.

Matches backward.cu: adaptive filter variance is held fixed and recomputed
from view depth/focal_x. CUDA does not differentiate filter3D or scale_modifier,
and its geometric mean gradient overwrites the earlier SH direction gradient.
These are reference semantics, not automatic differentiation of every forward
operation. See BACKWARD_IMPLEMENTATION.md for their implications.
"""

import torch
import warp as wp

from .dispatch import kernel_arg, launch
from .geometry import load_camera
from .interop import current_stream


@wp.func
def sh_coefficient(d: wp.vec3, coefficient: int):
    x, y, z = d[0], d[1], d[2]
    if coefficient == 0:
        return 0.28209479177387814
    if coefficient == 1:
        return -0.4886025119029199 * y
    if coefficient == 2:
        return 0.4886025119029199 * z
    if coefficient == 3:
        return -0.4886025119029199 * x
    xx, yy, zz = x * x, y * y, z * z
    if coefficient == 4:
        return 1.0925484305920792 * x * y
    if coefficient == 5:
        return -1.0925484305920792 * y * z
    if coefficient == 6:
        return 0.31539156525252005 * (2.0 * zz - xx - yy)
    if coefficient == 7:
        return -1.0925484305920792 * x * z
    if coefficient == 8:
        return 0.5462742152960396 * (xx - yy)
    if coefficient == 9:
        return -0.5900435899266435 * y * (3.0 * xx - yy)
    if coefficient == 10:
        return 2.890611442640554 * x * y * z
    if coefficient == 11:
        return -0.4570457994644658 * y * (4.0 * zz - xx - yy)
    if coefficient == 12:
        return 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)
    if coefficient == 13:
        return -0.4570457994644658 * x * (4.0 * zz - xx - yy)
    if coefficient == 14:
        return 1.445305721320277 * z * (xx - yy)
    return -0.5900435899266435 * x * (xx - 3.0 * yy)


@wp.kernel(enable_backward=False)
def sh_adjoint(
    means: wp.array(dtype=wp.vec3),
    camera: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=int),
    clamped: wp.array2d(dtype=bool),
    color_gradient: wp.array(dtype=wp.vec3),
    degree: int,
    coefficients: int,
    gradient: wp.array(dtype=wp.vec3),
):
    # Consecutive threads own consecutive RGB outputs, including rejected
    # rows and unused SH coefficients. No separate SH zero-fill is needed.
    index = wp.tid()
    i = index // coefficients
    coefficient = index % coefficients
    value = wp.vec3(0.0)
    if radii[i] > 0 and coefficient < (degree + 1) * (degree + 1):
        ray = means[i] - camera[0]
        direction = ray / wp.length(ray)
        value = sh_coefficient(direction, coefficient) * color_gradient[i]
        for channel in range(3):
            ch = wp.static(channel)
            if clamped[i, ch]:
                value[ch] = 0.0
    gradient[index] = value


@wp.func
def quaternion_rotation(q: wp.vec4):
    r, x, y, z = q[0], q[1], q[2], q[3]
    # Rows are GLM columns, as in CUDA's quaternion backward equations.
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
def quaternion_gradient(q: wp.vec4, d: wp.mat33):
    r, x, y, z = q[0], q[1], q[2], q[3]
    return wp.vec4(
        2.0 * z * (d[0, 1] - d[1, 0])
        + 2.0 * y * (d[2, 0] - d[0, 2])
        + 2.0 * x * (d[1, 2] - d[2, 1]),
        2.0 * y * (d[1, 0] + d[0, 1])
        + 2.0 * z * (d[2, 0] + d[0, 2])
        + 2.0 * r * (d[1, 2] - d[2, 1])
        - 4.0 * x * (d[2, 2] + d[1, 1]),
        2.0 * x * (d[1, 0] + d[0, 1])
        + 2.0 * r * (d[2, 0] - d[0, 2])
        + 2.0 * z * (d[1, 2] + d[2, 1])
        - 4.0 * y * (d[2, 2] + d[0, 0]),
        2.0 * r * (d[0, 1] - d[1, 0])
        + 2.0 * x * (d[2, 0] + d[0, 2])
        + 2.0 * y * (d[1, 2] + d[2, 1])
        - 4.0 * z * (d[1, 1] + d[0, 0]),
    )


@wp.kernel(enable_backward=False)
def preprocess_adjoint(
    means: wp.array(dtype=wp.vec3),
    scales: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.vec4),
    opacity: wp.array(dtype=float),
    radii: wp.array(dtype=int),
    view: wp.array2d(dtype=float),
    proj: wp.array2d(dtype=float),
    camera: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    focal_x: float,
    ewa: bool,
    matrix_gradient: wp.array(dtype=wp.mat44),
    render_opacity_gradient: wp.array(dtype=float),
    opacity_gradient: wp.array(dtype=float),
    mean_gradient: wp.array(dtype=wp.vec3),
    scale_gradient: wp.array(dtype=wp.vec3),
    rotation_gradient: wp.array(dtype=wp.vec4),
):
    i = wp.tid()
    opacity_gradient[i] = render_opacity_gradient[i]
    if radii[i] <= 0:
        mean_gradient[i] = wp.vec3(0.0)
        scale_gradient[i] = wp.vec3(0.0)
        rotation_gradient[i] = wp.vec4(0.0)
        return
    mean = means[i]
    v = load_camera(view)
    mean_view_z = v[2, 0] * mean[0] + v[2, 1] * mean[1] + v[2, 2] * mean[2] + v[2, 3]
    filter_scale = wp.abs(mean_view_z) / focal_x * wp.sqrt(0.3)
    variance = filter_scale * filter_scale
    scale = scales[i]
    squared = wp.cw_mul(scale, scale)
    dilated_squared = squared + wp.vec3(variance)
    dilated = wp.vec3(
        wp.sqrt(dilated_squared[0]),
        wp.sqrt(dilated_squared[1]),
        wp.sqrt(dilated_squared[2]),
    )
    rotation = quaternion_rotation(rotations[i])
    viewport = wp.mat44(
        float(width) / 2.0,
        0.0,
        0.0,
        float(width) / 2.0 - 0.5,
        0.0,
        float(height) / 2.0,
        0.0,
        float(height) / 2.0 - 0.5,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    world_to_screen = viewport * load_camera(proj)
    # Render gradients follow CUDA's transposed storage convention.
    world_grad = wp.transpose(world_to_screen) * wp.transpose(matrix_gradient[i])
    columns = wp.mat33(
        world_grad[0, 0],
        world_grad[1, 0],
        world_grad[2, 0],
        world_grad[0, 1],
        world_grad[1, 1],
        world_grad[2, 1],
        world_grad[0, 2],
        world_grad[1, 2],
        world_grad[2, 2],
    )
    mean_grad = wp.vec3(world_grad[0, 3], world_grad[1, 3], world_grad[2, 3])
    compensation_scale = wp.vec3(0.0)
    compensation_direction = wp.vec3(0.0)
    direction = wp.vec3(0.0)
    if ewa:
        ray = mean - camera[0]
        direction = ray / wp.length(ray)
        rotated = rotation * direction
        ray_squared = wp.cw_mul(rotated, rotated)
        before = wp.vec3(
            squared[1] * squared[2], squared[0] * squared[2], squared[0] * squared[1]
        )
        after = wp.vec3(
            dilated_squared[1] * dilated_squared[2],
            dilated_squared[0] * dilated_squared[2],
            dilated_squared[0] * dilated_squared[1],
        )
        a, b = wp.dot(ray_squared, before), wp.dot(ray_squared, after)
        compensation = wp.sqrt(a / b)
        factor = render_opacity_gradient[i] * opacity[i] * compensation / (a * b)
        opacity_gradient[i] = render_opacity_gradient[i] * compensation
        da = wp.vec3(
            scale[0] * (ray_squared[1] * squared[2] + ray_squared[2] * squared[1]),
            scale[1] * (ray_squared[0] * squared[2] + ray_squared[2] * squared[0]),
            scale[2] * (ray_squared[0] * squared[1] + ray_squared[1] * squared[0]),
        )
        db = wp.vec3(
            scale[0]
            * (
                ray_squared[1] * dilated_squared[2]
                + ray_squared[2] * dilated_squared[1]
            ),
            scale[1]
            * (
                ray_squared[0] * dilated_squared[2]
                + ray_squared[2] * dilated_squared[0]
            ),
            scale[2]
            * (
                ray_squared[0] * dilated_squared[1]
                + ray_squared[1] * dilated_squared[0]
            ),
        )
        compensation_scale = factor * (b * da - a * db)
        compensation_direction = factor * (
            b * wp.cw_mul(rotated, before) - a * wp.cw_mul(rotated, after)
        )
        d_direction = wp.transpose(rotation) * compensation_direction
        mean_grad += (
            d_direction - direction * wp.dot(direction, d_direction)
        ) / wp.length(ray)
    scale_grad = wp.vec3(0.0)
    for axis in range(3):
        scale_grad[axis] = (
            scale[axis] * wp.dot(rotation[axis], columns[axis]) / dilated[axis]
            + compensation_scale[axis]
        )
        columns[axis] = (
            columns[axis] * dilated[axis] + direction * compensation_direction[axis]
        )
    mean_gradient[i] = mean_grad
    scale_gradient[i] = scale_grad
    rotation_gradient[i] = quaternion_gradient(rotations[i], columns)


@torch.no_grad()
def backward_preprocess(
    inputs, state, render_gradients, raster_settings, *, _bindings=None
):
    n = inputs["means3D"].shape[0]
    device = inputs["means3D"].device
    sh = inputs.get("shs")
    m = 0 if sh is None else sh.shape[1]
    allocation = dict(device=device, dtype=torch.float32)
    gradients = dict(
        means3D=torch.empty((n, 3), **allocation),
        means2D=torch.zeros((n, 3), **allocation),
        scales=torch.empty((n, 3), **allocation),
        rotations=torch.empty((n, 4), **allocation),
        shs=torch.empty((n, m, 3), **allocation),
        colors_precomp=render_gradients["rgb"],
        opacities=torch.empty(n, **allocation),
    )
    if n == 0:
        return gradients

    def arg(tensor, dtype=wp.float32):
        return kernel_arg(tensor.contiguous(), dtype, _bindings)

    config = raster_settings
    stream = current_stream(device, bindings=_bindings)
    camera = arg(config.campos.view(1, 3), wp.vec3)
    if sh is not None:
        launch(
            sh_adjoint,
            dim=n * m,
            inputs=[
                arg(inputs["means3D"], wp.vec3),
                camera,
                arg(state["radii"], wp.int32),
                arg(state["clamped"], wp.bool),
                arg(render_gradients["rgb"], wp.vec3),
                config.sh_degree,
                m,
            ],
            outputs=[arg(gradients["shs"].view(-1, 3), wp.vec3)],
            stream=stream,
            bindings=_bindings,
        )
    launch(
        preprocess_adjoint,
        dim=n,
        inputs=[
            arg(inputs["means3D"], wp.vec3),
            arg(inputs["scales"], wp.vec3),
            arg(inputs["rotations"], wp.vec4),
            arg(inputs["opacities"].view(-1)),
            arg(state["radii"], wp.int32),
            arg(config.viewmatrix),
            arg(config.projmatrix),
            camera,
            config.image_width,
            config.image_height,
            config.image_width / (2.0 * config.tanfovx),
            config.settings.proper_ewa_scaling,
            arg(render_gradients["gauss2screen"], wp.mat44),
            arg(render_gradients["opacity"]),
        ],
        outputs=[
            arg(gradients["opacities"]),
            arg(gradients["means3D"], wp.vec3),
            arg(gradients["scales"], wp.vec3),
            arg(gradients["rotations"], wp.vec4),
        ],
        stream=stream,
        bindings=_bindings,
    )
    return gradients
