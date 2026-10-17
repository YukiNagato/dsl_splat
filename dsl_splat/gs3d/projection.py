"""Original 3DGS projection and local Warp-generated VJPs.

The custom inverse-covariance and clamp adjoints preserve the CUDA reference's
regularization and saturated-Jacobian convention. Culling and tile membership
are discrete; only visible rows define projected geometry. Camera gradients
are reduced within each thread block before its leader updates the shared matrices.
"""

from dataclasses import dataclass
from functools import cache
from typing import Any

from .cameras import ViewConfig

import torch
from torch import Tensor
import warp as wp

from .math import point_row, reference_matmul, reference_rotation
from ..runtime.dispatch import FrameBindings, kernel_arg, launch
from ..runtime.interop import current_stream

_BLOCK_DIM = 256


@dataclass(slots=True)
class ProjectedFootprint:
    centers: Tensor
    conics: Tensor


@dataclass(slots=True)
class ProjectedGaussians(ProjectedFootprint):
    radii: Tensor
    tile_counts: Tensor
    depths: Tensor


@wp.func
def tile_bounds(center: wp.vec2, radius: int, gx: int, gy: int):
    # Keep CUDA getRect's integer truncation and +BLOCK-1 arithmetic.
    return wp.vec4i(
        wp.clamp(int((center[0] - float(radius)) / 16.0), 0, gx),
        wp.clamp(int((center[1] - float(radius)) / 16.0), 0, gy),
        wp.clamp(int(((center[0] + float(radius)) + 16.0 - 1.0) / 16.0), 0, gx),
        wp.clamp(int(((center[1] + float(radius)) + 16.0 - 1.0) / 16.0), 0, gy),
    )


@wp.func
def clamp_point(point: wp.vec3, limx: float, limy: float):
    return wp.vec3(
        wp.clamp(point[0] / point[2], -limx, limx) * point[2],
        wp.clamp(point[1] / point[2], -limy, limy) * point[2],
        point[2],
    )


@wp.func_grad(clamp_point)
def adj_clamp_point(point: wp.vec3, limx: float, limy: float, gradient: wp.vec3):
    x, y = point[0] / point[2], point[1] / point[2]
    wp.adjoint[point] += wp.vec3(
        gradient[0] * float(x >= -limx and x <= limx),
        gradient[1] * float(y >= -limy and y <= limy),
        gradient[2],
    )


@wp.func
def invert_covariance(cov: wp.vec3):
    a, b, c = cov[0], cov[1], cov[2]
    inverse = 1.0 / (a * c - b * b)
    return wp.vec3(c * inverse, -b * inverse, a * inverse)


@wp.func_grad(invert_covariance)
def adj_invert_covariance(cov: wp.vec3, gradient: wp.vec3):
    a, b, c = cov[0], cov[1], cov[2]
    det = a * c - b * b
    factor = 1.0 / (det * det + 0.0000001)
    x, y, z = gradient[0], gradient[1], gradient[2]
    wp.adjoint[cov] += factor * wp.vec3(
        -c * c * x + b * c * y + (det - a * c) * z,
        2.0 * b * c * x - (det + 2.0 * b * b) * y + 2.0 * a * b * z,
        -a * a * z + a * b * y + (det - a * c) * x,
    )


@wp.func
def load_matrix(matrix: wp.array2d(dtype=float)):
    return wp.mat44(
        matrix[0, 0],
        matrix[0, 1],
        matrix[0, 2],
        matrix[0, 3],
        matrix[1, 0],
        matrix[1, 1],
        matrix[1, 2],
        matrix[1, 3],
        matrix[2, 0],
        matrix[2, 1],
        matrix[2, 2],
        matrix[2, 3],
        matrix[3, 0],
        matrix[3, 1],
        matrix[3, 2],
        matrix[3, 3],
    )


@wp.func
def transform_point(point: wp.vec3, matrix: wp.mat44):
    return wp.vec4(
        point_row(matrix, 0, point),
        point_row(matrix, 1, point),
        point_row(matrix, 2, point),
        point_row(matrix, 3, point),
    )


@wp.func
def screen_covariance(
    position: wp.vec4,
    quat: wp.vec4,
    scale: wp.vec3,
    view: wp.mat44,
    width: int,
    height: int,
    fx: float,
    fy: float,
):
    t = clamp_point(
        wp.vec3(position[0], position[1], position[2]),
        1.3 * float(width) / (2.0 * fx),
        1.3 * float(height) / (2.0 * fy),
    )
    rotation = reference_rotation(quat)
    s = scale
    scaling = wp.mat33(s[0], 0.0, 0.0, 0.0, s[1], 0.0, 0.0, 0.0, s[2])
    m = reference_matmul(scaling, rotation)
    covariance = reference_matmul(wp.transpose(m), m)
    jacobian = wp.mat33(
        fx / t[2],
        0.0,
        0.0,
        0.0,
        fy / t[2],
        0.0,
        -(fx * t[0]) / (t[2] * t[2]),
        -(fy * t[1]) / (t[2] * t[2]),
        0.0,
    )
    camera = wp.mat33(
        view[0, 0],
        view[1, 0],
        view[2, 0],
        view[0, 1],
        view[1, 1],
        view[2, 1],
        view[0, 2],
        view[1, 2],
        view[2, 2],
    )
    transform = reference_matmul(camera, jacobian)
    projected = reference_matmul(
        reference_matmul(wp.transpose(transform), wp.transpose(covariance)), transform
    )
    # GLM cov[0][1] addresses row 1, column 0. Although the covariance is
    # mathematically symmetric, the opposite entry can round differently.
    return wp.vec3(projected[0, 0] + 0.3, projected[1, 0], projected[1, 1] + 0.3)


@wp.func
def pixel_center(mean: wp.vec3, projection: wp.mat44, width: int, height: int):
    clip = transform_point(mean, projection)
    inverse_w = 1.0 / (clip[3] + 0.0000001)
    # CUDA ndc2Pix uses double literals: promote after the float32 perspective
    # divide, then round only the final pixel coordinate back to float32.
    ndc_x = wp.float64(clip[0] * inverse_w)
    ndc_y = wp.float64(clip[1] * inverse_w)
    return wp.vec2(
        float(
            ((ndc_x + wp.float64(1.0)) * wp.float64(width) - wp.float64(1.0))
            * wp.float64(0.5)
        ),
        float(
            ((ndc_y + wp.float64(1.0)) * wp.float64(height) - wp.float64(1.0))
            * wp.float64(0.5)
        ),
    )


@wp.kernel(enable_backward=False)
def project(
    means: wp.array(dtype=wp.vec3),
    quats: wp.array(dtype=wp.vec4),
    scales: wp.array(dtype=wp.vec3),
    opacities: wp.array(dtype=float),
    view: wp.array2d(dtype=float),
    projection: wp.array2d(dtype=float),
    intrinsics: wp.array2d(dtype=float),
    width: int,
    height: int,
    near: float,
    far: float,
    radius_clip: float,
    centers: wp.array(dtype=wp.vec2),
    conics: wp.array(dtype=wp.vec4),
    radii: wp.array(dtype=int),
    tiles: wp.array(dtype=int),
    depths: wp.array(dtype=float),
):
    i = wp.tid()
    radii[i] = 0
    tiles[i] = 0
    camera = load_matrix(view)
    t4 = transform_point(means[i], camera)
    if t4[2] <= near or t4[2] >= far:
        return
    covariance = screen_covariance(
        t4,
        quats[i],
        scales[i],
        camera,
        width,
        height,
        intrinsics[0, 0],
        intrinsics[1, 1],
    )
    a, b, c = covariance[0], covariance[1], covariance[2]
    det = a * c - b * b
    if det == 0.0:
        return
    mid = 0.5 * (a + c)
    radius = int(wp.ceil(3.0 * wp.sqrt(mid + wp.sqrt(wp.max(0.1, mid * mid - det)))))
    if float(radius) <= radius_clip:
        return
    center = pixel_center(means[i], load_matrix(projection), width, height)
    bounds = tile_bounds(center, radius, (width + 15) // 16, (height + 15) // 16)
    count = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    if count == 0:
        return
    inverse = invert_covariance(wp.vec3(a, b, c))
    centers[i] = center
    conics[i] = wp.vec4(inverse[0], inverse[1], inverse[2], opacities[i])
    radii[i] = radius
    tiles[i] = count
    depths[i] = t4[2]


@wp.struct
class Gaussian:
    mean: wp.vec3
    quat: wp.vec4
    scale: wp.vec3
    opacity: float


@wp.struct
class Camera:
    view: wp.mat44
    projection: wp.mat44


@wp.struct
class ProjectionParameters:
    width: int
    height: int
    fx: float
    fy: float


@wp.struct
class ProjectionSeed:
    center: wp.vec2
    conic: wp.vec4


@wp.func
def projection_loss(
    g: Gaussian, camera: Camera, p: ProjectionParameters, seed: ProjectionSeed
):
    """A seeded local VJP; all differentiable inputs are thread-local values."""
    position = transform_point(g.mean, camera.view)
    covariance = screen_covariance(
        position, g.quat, g.scale, camera.view, p.width, p.height, p.fx, p.fy
    )
    inverse = invert_covariance(covariance)
    conic = wp.vec4(inverse[0], inverse[1], inverse[2], g.opacity)
    center = pixel_center(g.mean, camera.projection, p.width, p.height)
    return wp.dot(center, seed.center) + wp.dot(conic, seed.conic)


@cache
def _make_backward(needs: tuple[bool, ...]) -> wp.Kernel:
    """Specialize unused gradients away; aggregate camera updates per block."""

    @wp.kernel(module="unique", enable_backward=False)
    def project_backward(
        means: wp.array(dtype=wp.vec3),
        quats: wp.array(dtype=wp.vec4),
        scales: wp.array(dtype=wp.vec3),
        opacities: wp.array(dtype=float),
        view: wp.array2d(dtype=float),
        projection: wp.array2d(dtype=float),
        intrinsics: wp.array2d(dtype=float),
        radii: wp.array(dtype=int),
        center_seed: wp.array(dtype=wp.vec2),
        conic_seed: wp.array(dtype=wp.vec4),
        width: int,
        height: int,
        grad_means: wp.array(dtype=wp.vec3),
        grad_quats: wp.array(dtype=wp.vec4),
        grad_scales: wp.array(dtype=wp.vec3),
        grad_opacities: wp.array(dtype=float),
        camera_gradients: wp.array(dtype=wp.mat44),
    ):
        i = wp.tid()
        g_gradient, camera_gradient = Gaussian(), Camera()
        if i < means.shape[0] and radii[i] > 0:
            g = Gaussian()
            g.mean = means[i]
            g.quat = quats[i]
            g.scale = scales[i]
            g.opacity = opacities[i]
            camera = Camera()
            camera.view = load_matrix(view)
            camera.projection = load_matrix(projection)
            p = ProjectionParameters()
            p.width = width
            p.height = height
            p.fx = intrinsics[0, 0]
            p.fy = intrinsics[1, 1]
            seed = ProjectionSeed()
            seed.center = center_seed[i]
            seed.conic = conic_seed[i]
            g_gradient, camera_gradient, unused_parameters, unused_seed = wp.grad(
                projection_loss
            )(g, camera, p, seed)
        # Every real GS owns its gradient entries, including explicit culled zeros.
        if i < means.shape[0]:
            if wp.static(needs[0]):
                grad_means[i] = g_gradient.mean
            if wp.static(needs[1]):
                grad_quats[i] = g_gradient.quat
            if wp.static(needs[2]):
                grad_scales[i] = g_gradient.scale
            if wp.static(needs[3]):
                grad_opacities[i] = g_gradient.opacity
        # All lanes participate, with zeros for culled GSs and last-block padding.
        # Warp reduces within warps, then across warp leaders in shared memory.
        # Only the block leader atomically updates each final matrix entry.
        if wp.static(needs[4]):
            view_gradient = wp.mat44(camera_gradient.view)
            view_sum = wp.tile_sum(wp.tile(view_gradient, preserve_type=True))
            wp.tile_atomic_add(camera_gradients, view_sum, offset=0)
        if wp.static(needs[5]):
            projection_gradient = wp.mat44(camera_gradient.projection)
            projection_sum = wp.tile_sum(
                wp.tile(projection_gradient, preserve_type=True)
            )
            wp.tile_atomic_add(
                camera_gradients, projection_sum, offset=wp.static(int(needs[4]))
            )

    return project_backward


def _input_arguments(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    view: Tensor,
    projection: Tensor,
    intrinsics: Tensor,
    bindings: FrameBindings,
) -> list[Any]:
    return [
        kernel_arg(means, wp.vec3, bindings),
        kernel_arg(quats, wp.vec4, bindings),
        kernel_arg(scales, wp.vec3, bindings),
        kernel_arg(opacities, wp.float32, bindings),
        kernel_arg(view, bindings=bindings),
        kernel_arg(projection, bindings=bindings),
        kernel_arg(intrinsics, bindings=bindings),
    ]


@torch.no_grad()
def prepare(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    view: Tensor,
    projection: Tensor,
    intrinsics: Tensor,
    config: ViewConfig,
    bindings: FrameBindings,
) -> ProjectedGaussians:
    wp.init()
    n = len(means)
    state = ProjectedGaussians(
        means.new_empty((n, 2)),
        means.new_empty((n, 4)),
        torch.empty(n, dtype=torch.int32, device=means.device),
        torch.empty(n, dtype=torch.int32, device=means.device),
        means.new_empty(n),
    )
    if n:
        launch(
            project,
            dim=n,
            inputs=_input_arguments(
                means, quats, scales, opacities, view, projection, intrinsics, bindings
            )
            + [
                config.width,
                config.height,
                config.near,
                config.far,
                config.radius_clip,
            ],
            outputs=[
                kernel_arg(state.centers, wp.vec2, bindings),
                kernel_arg(state.conics, wp.vec4, bindings),
                kernel_arg(state.radii, bindings=bindings),
                kernel_arg(state.tile_counts, bindings=bindings),
                kernel_arg(state.depths, bindings=bindings),
            ],
            stream=current_stream(means.device, bindings=bindings),
            bindings=bindings,
        )
    return state


@torch.no_grad()
def backward(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    view: Tensor,
    projection: Tensor,
    intrinsics: Tensor,
    config: ViewConfig,
    state: ProjectedGaussians,
    grad_centers: Tensor,
    grad_conics: Tensor,
    needs: tuple[bool, ...],
    bindings: FrameBindings,
) -> list[Tensor | None]:
    if not any(needs):
        return [None] * 6
    tensors = [means, quats, scales, opacities, view, projection]
    gradients = [
        torch.empty_like(t) if needed else None
        for t, needed in zip(tensors[:4], needs[:4])
    ] + [None, None]
    if not len(means):
        for index in (4, 5):
            if needs[index]:
                gradients[index] = torch.zeros_like(tensors[index])
        return gradients
    camera_count = sum(needs[4:])
    # Only the final matrices need storage; local contributions stay in registers.
    camera_gradients = means.new_zeros((camera_count, 4, 4))
    count = len(means)
    launch_count = (
        ((count + _BLOCK_DIM - 1) // _BLOCK_DIM) * _BLOCK_DIM if camera_count else count
    )
    types = (wp.vec3, wp.vec4, wp.vec3, wp.float32)
    launch(
        _make_backward(needs),
        dim=launch_count,
        inputs=_input_arguments(
            means, quats, scales, opacities, view, projection, intrinsics, bindings
        )
        + [
            kernel_arg(state.radii, bindings=bindings),
            kernel_arg(grad_centers, wp.vec2, bindings),
            kernel_arg(grad_conics, wp.vec4, bindings),
            config.width,
            config.height,
        ],
        outputs=[
            kernel_arg(
                (
                    value
                    if value is not None
                    else tensor.new_empty((0,) + tuple(tensor.shape[1:]))
                ),
                dtype,
                bindings,
            )
            for tensor, value, dtype in zip(tensors[:4], gradients[:4], types)
        ]
        + [kernel_arg(camera_gradients, wp.mat44, bindings)],
        stream=current_stream(means.device, bindings=bindings),
        block_dim=_BLOCK_DIM,
        bindings=bindings,
    )
    if camera_count:
        if needs[4]:
            gradients[4] = camera_gradients[0]
        if needs[5]:
            gradients[5] = camera_gradients[-1]
    return gradients
