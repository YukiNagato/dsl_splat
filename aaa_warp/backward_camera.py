"""Camera adjoint for AAA's continuous 3D evaluation, in Python/Warp.

The existing Gaussian adjoint keeps its CUDA conventions. Camera derivatives
also include adaptive filter variance, opacity compensation and SH direction.
Sorting, culling and alpha thresholds stay fixed, as in ordinary GS backward.
"""

import torch
import warp as wp

from .backward_preprocess import quaternion_rotation
from .dispatch import FrameBindings, kernel_arg, launch
from .geometry import load_camera
from .interop import current_stream
from .settings import GaussianRasterizationSettings
from .types import GaussianInputs, PreprocessedGaussians, RenderGradients


@wp.func
def sh_basis_gradient(d: wp.vec3, coefficient: int):
    x, y, z = d[0], d[1], d[2]
    if coefficient == 0:
        return wp.vec3(0.0)
    if coefficient == 1:
        return wp.vec3(0.0, -0.4886025119029199, 0.0)
    if coefficient == 2:
        return wp.vec3(0.0, 0.0, 0.4886025119029199)
    if coefficient == 3:
        return wp.vec3(-0.4886025119029199, 0.0, 0.0)
    if coefficient == 4:
        return 1.0925484305920792 * wp.vec3(y, x, 0.0)
    if coefficient == 5:
        return -1.0925484305920792 * wp.vec3(0.0, z, y)
    if coefficient == 6:
        return 0.31539156525252005 * wp.vec3(-2.0 * x, -2.0 * y, 4.0 * z)
    if coefficient == 7:
        return -1.0925484305920792 * wp.vec3(z, 0.0, x)
    if coefficient == 8:
        return 0.5462742152960396 * wp.vec3(2.0 * x, -2.0 * y, 0.0)
    if coefficient == 9:
        return -0.5900435899266435 * wp.vec3(6.0 * x * y, 3.0 * (x * x - y * y), 0.0)
    if coefficient == 10:
        return 2.890611442640554 * wp.vec3(y * z, x * z, x * y)
    if coefficient == 11:
        return 0.4570457994644658 * wp.vec3(
            2.0 * x * y, x * x + 3.0 * y * y - 4.0 * z * z, -8.0 * y * z
        )
    if coefficient == 12:
        return 0.3731763325901154 * wp.vec3(
            -6.0 * x * z, -6.0 * y * z, 6.0 * z * z - 3.0 * (x * x + y * y)
        )
    if coefficient == 13:
        return 0.4570457994644658 * wp.vec3(
            3.0 * x * x + y * y - 4.0 * z * z, 2.0 * x * y, -8.0 * x * z
        )
    if coefficient == 14:
        return 1.445305721320277 * wp.vec3(2.0 * x * z, -2.0 * y * z, x * x - y * y)
    return -0.5900435899266435 * wp.vec3(3.0 * (x * x - y * y), -6.0 * x * y, 0.0)


@wp.kernel(enable_backward=False)
def camera_adjoint(
    means: wp.array(dtype=wp.vec3),
    scales: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.vec4),
    filters: wp.array(dtype=float),
    has_filter: bool,
    radii: wp.array(dtype=int),
    effective_opacity: wp.array(dtype=float),
    view: wp.array2d(dtype=float),
    proj: wp.array2d(dtype=float),
    camera: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    focal: float,
    modifier: float,
    ewa: bool,
    matrix_gradient: wp.array(dtype=wp.mat44),
    opacity_gradient: wp.array(dtype=float),
    sh: wp.array2d(dtype=wp.vec3),
    use_sh: bool,
    degree: int,
    clamped: wp.array2d(dtype=bool),
    color_gradient: wp.array(dtype=float),
    contributions: wp.array2d(dtype=float),
):
    i = wp.tid()
    if radii[i] <= 0:
        for component in range(23):
            contributions[i, component] = 0.0
        return
    mean = means[i]
    v = load_camera(view)
    depth = wp.dot(v[2], wp.vec4(mean[0], mean[1], mean[2], 1.0))
    adaptive_variance = 0.3 * (depth / focal) * (depth / focal)
    variance = adaptive_variance
    filter_active = True
    if has_filter:
        fixed_variance = filters[i] * filters[i]
        variance = wp.max(variance, fixed_variance)
        filter_active = adaptive_variance > fixed_variance
    squared = wp.cw_mul(scales[i], scales[i])
    dilated_squared = squared + wp.vec3(variance)
    dilated = wp.vec3(
        wp.sqrt(dilated_squared[0]),
        wp.sqrt(dilated_squared[1]),
        wp.sqrt(dilated_squared[2]),
    )
    rotation = quaternion_rotation(rotations[i])
    root_modifier = wp.sqrt(modifier)
    linear = (
        wp.transpose(
            wp.mat33(dilated[0], 0.0, 0.0, 0.0, dilated[1], 0.0, 0.0, 0.0, dilated[2])
            * rotation
        )
        * root_modifier
    )
    world = wp.mat44(
        linear[0, 0],
        linear[0, 1],
        linear[0, 2],
        mean[0],
        linear[1, 0],
        linear[1, 1],
        linear[1, 2],
        mean[1],
        linear[2, 0],
        linear[2, 1],
        linear[2, 2],
        mean[2],
        0.0,
        0.0,
        0.0,
        1.0,
    )
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
    screen_gradient = wp.transpose(matrix_gradient[i])
    projection_gradient = wp.transpose(viewport) * screen_gradient * wp.transpose(world)
    world_gradient = wp.transpose(viewport * load_camera(proj)) * screen_gradient
    variance_gradient = float(0.0)
    for axis in range(3):
        column = wp.vec3(
            world_gradient[0, axis], world_gradient[1, axis], world_gradient[2, axis]
        )
        variance_gradient += (
            wp.dot(column, rotation[axis]) * root_modifier / (2.0 * dilated[axis])
        )
    camera_gradient = wp.vec3(0.0)
    ray = mean - camera[0]
    distance = wp.length(ray)
    direction = ray / distance
    if ewa:
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
        factor = opacity_gradient[i] * effective_opacity[i]
        rotated_gradient = factor * (
            wp.cw_mul(rotated, before) / a - wp.cw_mul(rotated, after) / b
        )
        direction_gradient = wp.transpose(rotation) * rotated_gradient
        camera_gradient -= (
            direction_gradient - direction * wp.dot(direction, direction_gradient)
        ) / distance
        area_variance_gradient = wp.vec3(
            dilated_squared[1] + dilated_squared[2],
            dilated_squared[0] + dilated_squared[2],
            dilated_squared[0] + dilated_squared[1],
        )
        variance_gradient -= (
            0.5 * factor * wp.dot(ray_squared, area_variance_gradient) / b
        )
    if use_sh:
        color_grad = wp.vec3(0.0)
        for channel in range(3):
            if not clamped[i, channel]:
                color_grad[channel] = color_gradient[i * 3 + channel]
        direction_grad = wp.vec3(0.0)
        for coefficient in range((degree + 1) * (degree + 1)):
            direction_grad += wp.dot(
                sh[i, coefficient], color_grad
            ) * sh_basis_gradient(direction, coefficient)
        camera_gradient -= (
            direction_grad - direction * wp.dot(direction, direction_grad)
        ) / distance
    depth_gradient = float(0.0)
    if filter_active:
        depth_gradient = variance_gradient * 0.6 * depth / (focal * focal)
    for row in range(4):
        for col in range(4):
            # Public settings store projection/view matrices transposed.
            contributions[i, row * 4 + col] = projection_gradient[col, row]
        value = float(1.0)
        if row < 3:
            value = mean[row]
        contributions[i, 16 + row] = depth_gradient * value
    for axis in range(3):
        contributions[i, 20 + axis] = camera_gradient[axis]


@torch.no_grad()
def backward_camera(
    inputs: GaussianInputs,
    state: PreprocessedGaussians,
    render_gradients: RenderGradients,
    config: GaussianRasterizationSettings,
    *,
    _bindings: FrameBindings | None = None,
) -> dict[str, torch.Tensor]:
    """Reduce per-Gaussian camera contributions without contended atomics."""
    means = inputs["means3D"]
    n, device = means.shape[0], means.device
    gradients = {
        name: torch.zeros_like(getattr(config, name))
        for name in ("viewmatrix", "projmatrix", "campos")
    }
    if n == 0:
        return gradients
    rows = torch.empty((n, 23), device=device, dtype=torch.float32)
    empty = torch.empty(0, device=device, dtype=torch.float32)
    filters = inputs.get("filter3D")
    sh = inputs.get("shs")

    def arg(tensor, dtype=wp.float32):
        return kernel_arg(tensor.contiguous(), dtype, _bindings)

    launch(
        camera_adjoint,
        dim=n,
        inputs=[
            arg(means, wp.vec3),
            arg(inputs["scales"], wp.vec3),
            arg(inputs["rotations"], wp.vec4),
            arg(empty if filters is None else filters.view(-1)),
            filters is not None,
            arg(state["radii"], wp.int32),
            arg(state["opacity"]),
            arg(config.viewmatrix),
            arg(config.projmatrix),
            arg(config.campos.view(1, 3), wp.vec3),
            config.image_width,
            config.image_height,
            max(
                config.image_width / (2 * config.tanfovx),
                config.image_height / (2 * config.tanfovy),
            ),
            config.scale_modifier,
            config.settings.proper_ewa_scaling,
            arg(render_gradients["gauss2screen"], wp.mat44),
            arg(render_gradients["opacity"]),
            arg(empty.view(0, 0, 3) if sh is None else sh, wp.vec3),
            sh is not None,
            config.sh_degree,
            arg(
                state.get(
                    "clamped", torch.empty((0, 3), device=device, dtype=torch.bool)
                ),
                wp.bool,
            ),
            arg(render_gradients["rgb"].view(-1)),
        ],
        outputs=[arg(rows)],
        stream=current_stream(device, bindings=_bindings),
        bindings=_bindings,
    )
    total = rows.sum(0)
    gradients["projmatrix"] = total[:16].view(4, 4)
    gradients["viewmatrix"][:, 2] = total[16:20]
    gradients["campos"] = total[20:23]
    return gradients
