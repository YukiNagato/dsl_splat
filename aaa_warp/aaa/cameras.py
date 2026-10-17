"""OpenCV pinhole cameras and their internal AAA kernel parameters."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from ..cameras import PinholeCamera, validate_camera_options, read_intrinsics
from ..runtime.precision import camera_matmul
from .config import FrameConfig
from .settings import ExtendedSettings


def camera_config(
    camera: PinholeCamera, background: Tensor, settings: ExtendedSettings
) -> FrameConfig:
    from ..inputs import check_tensor

    validate_camera_options(
        camera.width, camera.height, camera.near_plane, camera.far_plane, 0.0
    )
    if not isinstance(camera.viewmat, Tensor):
        raise ValueError("viewmat must be a CUDA float32 tensor")
    device = camera.viewmat.device
    if device.type != "cuda":
        raise ValueError("camera tensors must be on CUDA")
    check_tensor("viewmat", camera.viewmat, device, (4, 4))
    check_tensor("K", camera.K, device, (3, 3))
    if torch.is_grad_enabled() and camera.K.requires_grad:
        raise NotImplementedError("intrinsics gradients are not supported")
    cameras = prepare_cameras(
        camera.viewmat[None, None],
        camera.K.detach()[None, None],
        camera.width,
        camera.height,
        camera.near_plane,
        camera.far_plane,
    )
    return cameras.config(0, 0, background, settings)


@dataclass(slots=True)
class CameraBatch:
    transposed_views: Tensor
    projection: Tensor
    inverse_projection: Tensor
    centers: Tensor
    intrinsics: list[list[tuple[float, float, float, float]]]
    width: int
    height: int

    def config(
        self, batch: int, camera: int, background: Tensor, settings: ExtendedSettings
    ) -> FrameConfig:
        fx, fy, _, _ = self.intrinsics[batch][camera]
        return FrameConfig(
            image_height=self.height,
            image_width=self.width,
            tanfovx=self.width / (2 * fx),
            tanfovy=self.height / (2 * fy),
            bg=background,
            scale_modifier=1.0,
            viewmatrix=self.transposed_views[batch, camera],
            projmatrix=self.projection[batch, camera],
            inv_viewprojmatrix=self.inverse_projection[batch, camera],
            sh_degree=0,
            campos=self.centers[batch, camera],
            prefiltered=False,
            settings=settings,
            render_depth=False,
            debug=False,
            principal_point=self.intrinsics[batch][camera][2:],
        )


def prepare_cameras(
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    near_plane: float,
    far_plane: float,
) -> CameraBatch:
    batch_count, cameras, device = viewmats.shape[0], viewmats.shape[1], viewmats.device
    # Read the small intrinsic array once: the core API takes scalar FOVs.
    intrinsics = read_intrinsics(Ks)
    views = viewmats
    inverse_views = torch.linalg.inv(views)
    projections = torch.zeros(batch_count, cameras, 4, 4, dtype=torch.float32)
    projections[..., 0, 0] = 2 * intrinsics[..., 0, 0] / width
    projections[..., 1, 1] = 2 * intrinsics[..., 1, 1] / height
    projections[..., 0, 2] = 2 * intrinsics[..., 0, 2] / width - 1
    projections[..., 1, 2] = 2 * intrinsics[..., 1, 2] / height - 1
    # Match AAA's frontend getProjectionMatrix: nominal near/far map to 0/1.
    # Keep its combination with the backend's (-1,1) frustum test unchanged.
    projections[..., 2, 2] = (
        1 if math.isinf(far_plane) else far_plane / (far_plane - near_plane)
    )
    projections[..., 2, 3] = (
        -near_plane
        if math.isinf(far_plane)
        else -far_plane * near_plane / (far_plane - near_plane)
    )
    projections[..., 3, 2] = 1
    full = camera_matmul(projections.to(device), views).transpose(-1, -2).contiguous()
    # Used by discrete depth sorting/culling, which remains fixed in backward.
    inverse_full = torch.linalg.inv(full.detach()).contiguous()
    values = [
        [(k[0][0], k[1][1], k[0][2], k[1][2]) for k in batch]
        for batch in intrinsics.tolist()
    ]
    return CameraBatch(
        views.transpose(-1, -2).contiguous(),
        full,
        inverse_full,
        inverse_views[..., :3, 3].contiguous(),
        values,
        width,
        height,
    )
