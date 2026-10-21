"""RaDe-GS ray-plane projection and local Warp-generated VJPs.

The inverse-covariance adjoint and method-local clamp preserve the CUDA
reference's regularization and saturated-Jacobian convention. Culling and tile membership
are discrete; only visible rows define projected geometry. Camera gradients
are reduced within each thread block before its leader updates the shared matrices.
"""

from functools import cache
from typing import Any

from .types import ViewConfig, ProjectedGaussians

import torch
from torch import Tensor
import warp as wp

from ..splatting.math import reference_matmul, reference_rotation
from ..runtime.dispatch import FrameBindings, kernel_arg, launch
from ..runtime.interop import current_stream

from ..splatting.geometry import (
    tile_bounds,
    invert_covariance,
    load_matrix,
    transform_point,
    pixel_center,
)

_BLOCK_DIM = 256


@wp.func_native("return norm3df(position[0], position[1], position[2]);")
def cuda_sort_distance(position: wp.vec3) -> float:
    """Keep CUDA's FP32 keys: a one-ULP change can reorder equal-depth splats."""
    pass


@wp.func
def clamp_point(point: wp.vec3, limx: float, limy: float):
    # RaDe differentiates limit*z through z when a coordinate saturates. This
    # differs from the original-3DGS custom adjoint in splatting.geometry.
    return wp.vec3(
        wp.clamp(point[0] / point[2], -limx, limx) * point[2],
        wp.clamp(point[1] / point[2], -limy, limy) * point[2],
        point[2],
    )


@wp.func
def opacity_coefficient(cov: wp.vec3, kernel: float):
    det0 = wp.max(1.0e-6, cov[0] * cov[2] - cov[1] * cov[1])
    det1 = wp.max(1.0e-6, (cov[0] + kernel) * (cov[2] + kernel) - cov[1] * cov[1])
    return wp.sqrt(det0 / det1)


@wp.func_grad(opacity_coefficient)
def adj_opacity_coefficient(cov: wp.vec3, kernel: float, gradient: float):
    # Preserve upstream's stabilized backward, including its sqrt-coef factor
    # in the second determinant term (rather than silently changing training).
    det0 = wp.max(1.0e-6, cov[0] * cov[2] - cov[1] * cov[1])
    det1 = wp.max(1.0e-6, (cov[0] + kernel) * (cov[2] + kernel) - cov[1] * cov[1])
    coef = wp.sqrt(det0 / det1)
    d0 = gradient * 0.5 / (coef + 1.0e-6) / det1
    d1 = -d0 * coef
    wp.adjoint[cov] += wp.vec3(
        d0 * cov[2] + d1 * (cov[2] + kernel),
        -2.0 * cov[1] * (d0 + d1),
        d0 * cov[0] + d1 * (cov[0] + kernel),
    )


@wp.func
def reference_normalize(value: wp.vec3):
    return wp.normalize(value)


@wp.func_grad(reference_normalize)
def adj_reference_normalize(value: wp.vec3, gradient: wp.vec3):
    # render_backward.cu uses the normalized vector's length here.
    normal = wp.normalize(value)
    wp.adjoint[value] += (gradient - normal * wp.dot(normal, gradient)) / wp.length(
        normal
    )


@wp.func
def raster_covariance(scale: wp.vec3, rotation: wp.mat33, transform: wp.mat33):
    scaling = wp.mat33(scale[0], 0.0, 0.0, 0.0, scale[1], 0.0, 0.0, 0.0, scale[2])
    m = reference_matmul(reference_matmul(scaling, rotation), transform)
    cov = reference_matmul(wp.transpose(m), m)
    return wp.vec3(cov[0, 0], cov[1, 0], cov[1, 1])


@wp.struct
class RayGeometry:
    covariance: wp.vec3
    plane: wp.vec3
    normal: wp.vec3


@wp.func
def geometry(
    position: wp.vec4,
    quat: wp.vec4,
    scale: wp.vec3,
    view: wp.mat44,
    width: int,
    height: int,
    fx: float,
    fy: float,
):
    original = wp.vec3(position[0], position[1], position[2])
    t = clamp_point(
        original, 1.3 * float(width) / (2.0 * fx), 1.3 * float(height) / (2.0 * fy)
    )
    u, v, z = t[0] / t[2], t[1] / t[2], t[2]
    rotation = reference_rotation(quat)
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
    jacobian = wp.mat33(
        fx / z,
        0.0,
        0.0,
        0.0,
        fy / z,
        0.0,
        -fx * t[0] / (z * z),
        -fy * t[1] / (z * z),
        0.0,
    )
    transform = reference_matmul(camera, jacobian)
    cov = raster_covariance(scale, rotation, transform)
    # Use the same tie-breaking order as find_min_from_triple in CUDA.
    bigger = int(scale[0] < scale[1])
    i0, i1 = (bigger + 1) % 3, (bigger + 2) % 3
    minimum = i0
    if scale[i0] > scale[i1]:
        minimum = i1
    inv_cov = wp.mat33()
    if scale[minimum] > 1.0e-7:
        inv_s = wp.mat33(
            1.0 / scale[0], 0.0, 0.0, 0.0, 1.0 / scale[1], 0.0, 0.0, 0.0, 1.0 / scale[2]
        )
        inv_m = reference_matmul(reference_matmul(inv_s, rotation), camera)
        inv_cov = reference_matmul(wp.transpose(inv_m), inv_m)
    else:
        axis = wp.vec3(rotation[minimum, 0], rotation[minimum, 1], rotation[minimum, 2])
        axis_camera = wp.transpose(camera) * axis
        inv_cov = wp.outer(axis_camera, axis_camera)
    uvh = wp.vec3(u, v, 1.0)
    direction = wp.normalize(inv_cov * uvh)
    result = RayGeometry()
    result.covariance = cov
    if wp.length_sq(direction) == 0.0 or not wp.isfinite(direction[0]):
        result.plane = wp.vec3(0.0)
        result.normal = wp.vec3(0.0, 0.0, -1.0)
    else:
        length = wp.length(t)
        factor = length / (u * u + v * v + 1.0)
        nj_inv = wp.mat33(
            v * v + 1.0, -u * v, -u, -u * v, u * u + 1.0, -v, 0.0, 0.0, 0.0
        )
        plane = nj_inv * (direction / wp.max(wp.dot(direction, uvh), 1.0e-7))
        result.plane = wp.vec3(
            plane[0] * factor / fx, plane[1] * factor / fy, wp.length(original)
        )
        ray_normal = wp.vec3(-plane[0] * factor, -plane[1] * factor, -1.0)
        nj = wp.mat33(
            1.0 / z,
            0.0,
            t[0] / length,
            0.0,
            1.0 / z,
            t[1] / length,
            -t[0] / (z * z),
            -t[1] / (z * z),
            z / length,
        )
        result.normal = reference_normalize(nj * ray_normal)
    return result


# Match the reference's --use_fast_math, including norm3df sorting-key rounding.
@wp.kernel(module="unique", enable_backward=False, module_options={"fast_math": True})
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
    kernel_size: float,
    centers: wp.array(dtype=wp.vec2),
    conics: wp.array(dtype=wp.vec4),
    radii: wp.array(dtype=int),
    tiles: wp.array(dtype=int),
    depths: wp.array(dtype=float),
    ray_planes: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    radii[i] = 0
    tiles[i] = 0
    camera = load_matrix(view)
    t4 = transform_point(means[i], camera)
    if t4[2] <= near or t4[2] >= far:
        return
    rays = geometry(
        t4,
        quats[i],
        scales[i],
        camera,
        width,
        height,
        intrinsics[0, 0],
        intrinsics[1, 1],
    )
    covariance = rays.covariance
    a, b, c = covariance[0] + kernel_size, covariance[1], covariance[2] + kernel_size
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
    conics[i] = wp.vec4(
        inverse[0],
        inverse[1],
        inverse[2],
        opacities[i] * opacity_coefficient(covariance, kernel_size),
    )
    radii[i] = radius
    tiles[i] = count
    depths[i] = cuda_sort_distance(wp.vec3(t4[0], t4[1], t4[2]))
    ray_planes[i] = rays.plane
    normals[i] = rays.normal


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
    kernel_size: float


@wp.struct
class ProjectionSeed:
    center: wp.vec2
    conic: wp.vec4
    plane: wp.vec3
    normal: wp.vec3


@wp.func
def projection_loss(
    g: Gaussian, camera: Camera, p: ProjectionParameters, seed: ProjectionSeed
):
    """A seeded local VJP; all differentiable inputs are thread-local values."""
    position = transform_point(g.mean, camera.view)
    rays = geometry(
        position, g.quat, g.scale, camera.view, p.width, p.height, p.fx, p.fy
    )
    covariance = rays.covariance + wp.vec3(p.kernel_size, 0.0, p.kernel_size)
    inverse = invert_covariance(covariance)
    conic = wp.vec4(
        inverse[0],
        inverse[1],
        inverse[2],
        g.opacity * opacity_coefficient(rays.covariance, p.kernel_size),
    )
    center = pixel_center(g.mean, camera.projection, p.width, p.height)
    return (
        wp.dot(center, seed.center)
        + wp.dot(conic, seed.conic)
        + wp.dot(rays.plane, seed.plane)
        + wp.dot(rays.normal, seed.normal)
    )


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
        plane_seed: wp.array(dtype=wp.vec3),
        normal_seed: wp.array(dtype=wp.vec3),
        width: int,
        height: int,
        kernel_size: float,
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
            p.kernel_size = kernel_size
            seed = ProjectionSeed()
            seed.center = center_seed[i]
            seed.conic = conic_seed[i]
            seed.plane = plane_seed[i]
            seed.normal = normal_seed[i]
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
        means.new_empty((n, 3)),
        means.new_empty((n, 3)),
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
                config.kernel_size,
            ],
            outputs=[
                kernel_arg(state.centers, wp.vec2, bindings),
                kernel_arg(state.conics, wp.vec4, bindings),
                kernel_arg(state.radii, bindings=bindings),
                kernel_arg(state.tile_counts, bindings=bindings),
                kernel_arg(state.depths, bindings=bindings),
                kernel_arg(state.ray_planes, wp.vec3, bindings),
                kernel_arg(state.normals, wp.vec3, bindings),
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
    grad_planes: Tensor,
    grad_normals: Tensor,
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
            kernel_arg(grad_planes, wp.vec3, bindings),
            kernel_arg(grad_normals, wp.vec3, bindings),
            config.width,
            config.height,
            config.kernel_size,
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
