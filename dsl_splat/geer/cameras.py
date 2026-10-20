"""OpenCV poses and GEER's pinhole, BEAP and Kannala--Brandt ray grids."""

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor
import warp as wp

from ..cameras import PinholeCamera, read_intrinsics, validate_camera_options
from ..inputs import check_tensor
from ..runtime.dispatch import FrameBindings, kernel_arg, launch
from ..runtime.interop import current_stream
from .geometry import launch_cache


@dataclass(frozen=True, slots=True)
class RayCamera:
    """One central camera; axes point right/down/forward and pose is world-to-camera.

    Pinhole/KB use K. BEAP uses increasing 1D tangent grids, with directions
    (tan_theta[x], tan_phi[y], 1). KB uses four radial coefficients and optionally
    a matching HWC raymap. tan_fov supplies GEER's symmetric tangent limits.
    These cameras follow the reference's center-Z near/far culling.
    """

    viewmat: Tensor
    width: int
    height: int
    K: Tensor | None = None
    model: Literal["pinhole", "beap", "fisheye"] = "pinhole"
    tan_theta: Tensor | None = None
    tan_phi: Tensor | None = None
    raymap: Tensor | None = None
    distortion_coeffs: Tensor | None = None
    tan_fov: tuple[float, float] | None = None
    near_plane: float = 0.2
    far_plane: float = math.inf


@wp.kernel(enable_backward=False)
def _rays(
    width: int,
    height: int,
    mode: int,
    K: wp.array2d(dtype=float),
    tx: wp.array(dtype=float),
    ty: wp.array(dtype=float),
    output: wp.array(dtype=wp.vec3),
    limits: wp.array(dtype=float),
):
    i = wp.tid()
    x, y = i % width, i // width
    ray = wp.vec3()
    if mode == 0:
        ray = wp.vec3(tx[x], ty[y], 1.0)
    else:
        ray = wp.vec3(
            (float(x) + 0.5) / K[0, 0] - K[0, 2] / K[0, 0],
            (float(y) + 0.5) / K[1, 1] - K[1, 2] / K[1, 1],
            1.0,
        )
    output[i] = ray
    if mode == 2 and i == 0:
        # Cover the actual pinhole image even when its principal point is
        # off center. Reuse this launch instead of submitting scalar Torch ops.
        limits[0] = wp.max(wp.abs(K[0, 2]), wp.abs(float(width) - K[0, 2])) / K[0, 0]
        limits[1] = wp.max(wp.abs(K[1, 2]), wp.abs(float(height) - K[1, 2])) / K[1, 1]


@wp.kernel(enable_backward=False)
def _fisheye_rays(
    width: int,
    K: wp.array2d(dtype=float),
    coeffs: wp.array(dtype=float),
    output: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    x = (float(i % width) + 0.5 - K[0, 2]) / K[0, 0]
    y = (float(i // width) + 0.5 - K[1, 2]) / K[1, 1]
    radius = wp.sqrt(x * x + y * y)
    theta = float(radius)
    for j in range(10):
        t2 = theta * theta
        value = (
            theta
            * (
                1.0
                + t2
                * (coeffs[0] + t2 * (coeffs[1] + t2 * (coeffs[2] + t2 * coeffs[3])))
            )
            - radius
        )
        derivative = 1.0 + t2 * (
            3.0 * coeffs[0]
            + t2 * (5.0 * coeffs[1] + t2 * (7.0 * coeffs[2] + t2 * 9.0 * coeffs[3]))
        )
        if wp.abs(derivative) > 1.0e-8:
            theta -= value / derivative
    factor = wp.sin(theta) / wp.max(radius, 1.0e-9)
    output[i] = wp.vec3(x * factor, y * factor, wp.cos(theta))


def prepare_camera(
    camera: RayCamera | PinholeCamera, bindings: FrameBindings | None = None
) -> tuple[RayCamera, int, Tensor, Tensor]:
    """Return the normalized camera, mode, flat raymap and tangent limits."""
    wp.init()
    if bindings is None:
        bindings = FrameBindings(launch_cache)
    if isinstance(camera, PinholeCamera):
        camera = RayCamera(
            camera.viewmat,
            camera.width,
            camera.height,
            camera.K,
            near_plane=camera.near_plane,
            far_plane=camera.far_plane,
        )
    if not isinstance(camera, RayCamera):
        raise TypeError("camera must be a RayCamera or PinholeCamera")
    validate_camera_options(
        camera.width, camera.height, camera.near_plane, camera.far_plane, 0.0
    )
    view = camera.viewmat
    if not isinstance(view, Tensor) or not view.is_cuda:
        raise ValueError("viewmat must be a CUDA float32 (4,4) tensor")
    check_tensor("viewmat", view, view.device, (4, 4))
    mode = {"beap": 0, "fisheye": 1, "pinhole": 2}.get(camera.model)
    if mode is None:
        raise ValueError("camera model must be pinhole, beap or fisheye")
    K, tx, ty, distortion = (
        camera.K,
        camera.tan_theta,
        camera.tan_phi,
        camera.distortion_coeffs,
    )
    if mode != 0:
        check_tensor("K", K, view.device, (3, 3))
        read_intrinsics(K)
        if torch.is_grad_enabled() and K.requires_grad:
            raise NotImplementedError(
                "K gradients are not implemented; pose and supplied raymap gradients are supported"
            )
    else:
        check_tensor("tan_theta", tx, view.device, (camera.width,))
        check_tensor("tan_phi", ty, view.device, (camera.height,))
        if torch.is_grad_enabled() and (tx.requires_grad or ty.requires_grad):
            raise NotImplementedError("BEAP grid gradients are not implemented")
        if not bool((torch.diff(tx) > 0).all() & (torch.diff(ty) > 0).all()):
            raise ValueError("BEAP tangent grids must be strictly increasing")
    if mode == 1:
        check_tensor("distortion_coeffs", distortion, view.device, (4,))
        if torch.is_grad_enabled() and distortion.requires_grad:
            raise NotImplementedError("distortion gradients are not implemented")
        if camera.raymap is not None:
            check_tensor(
                "raymap", camera.raymap, view.device, (camera.height, camera.width, 3)
            )
            rays = camera.raymap.contiguous().view(-1, 3)
        else:
            rays = view.new_empty((camera.width * camera.height, 3))
            launch(
                _fisheye_rays,
                dim=len(rays),
                inputs=[
                    camera.width,
                    kernel_arg(K, bindings=bindings),
                    kernel_arg(distortion, bindings=bindings),
                ],
                outputs=[kernel_arg(rays, wp.vec3, bindings)],
                stream=current_stream(view.device, bindings=bindings),
                bindings=bindings,
            )
    else:
        rays = view.new_empty((camera.height * camera.width, 3))
        limits = view.new_empty(2)
        launch(
            _rays,
            dim=len(rays),
            inputs=[
                camera.width,
                camera.height,
                mode,
                kernel_arg(K, bindings=bindings),
                kernel_arg(tx, bindings=bindings),
                kernel_arg(ty, bindings=bindings),
            ],
            outputs=[
                kernel_arg(rays, wp.vec3, bindings),
                kernel_arg(limits, bindings=bindings),
            ],
            stream=current_stream(view.device, bindings=bindings),
            bindings=bindings,
        )
    if camera.tan_fov is not None:
        if len(camera.tan_fov) != 2 or any(
            not math.isfinite(v) or v <= 0 for v in camera.tan_fov
        ):
            raise ValueError("tan_fov must contain two finite positive values")
        limits = view.new_tensor(camera.tan_fov)
    elif mode == 0:
        limits = torch.stack((tx.abs().max(), ty.abs().max()))
    elif mode == 1:
        tangents = torch.nan_to_num(
            rays[:, :2] / rays[:, 2:], nan=0.0, posinf=0.0, neginf=0.0
        )
        limits = tangents.abs().amax(dim=0)
    return camera, mode, rays, limits
