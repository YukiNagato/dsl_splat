"""OpenCV pinhole cameras and their internal AAA kernel parameters."""

from dataclasses import dataclass
import math
from numbers import Integral

import torch
from torch import Tensor

from .config import FrameConfig
from .settings import ExtendedSettings


def validate_camera_options(
    width: int, height: int, near_plane: float, far_plane: float, radius_clip: float
) -> None:
    if any(
        isinstance(v, bool) or not isinstance(v, Integral) or v <= 0
        for v in (width, height)
    ):
        raise ValueError("width and height must be positive integers")
    if (
        not math.isfinite(near_plane)
        or near_plane <= 0
        or math.isnan(far_plane)
        or far_plane <= near_plane
    ):
        raise ValueError("require 0 < near_plane < far_plane")
    if not math.isfinite(radius_clip) or radius_clip < 0:
        raise ValueError("radius_clip must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class PinholeCamera:
    """One OpenCV camera: rigid world-to-camera pose and zero-skew K.

    X points right, Y down and Z forward. K may have arbitrary cx/cy.
    near/far are AAA projection parameters, not strict center-depth cutoffs.
    """

    viewmat: Tensor
    K: Tensor
    width: int
    height: int
    near_plane: float = 0.01
    far_plane: float = 1e10

    def _config(self, background: Tensor, settings: ExtendedSettings) -> FrameConfig:
        from .inputs import check_tensor

        validate_camera_options(
            self.width, self.height, self.near_plane, self.far_plane, 0.0
        )
        if not isinstance(self.viewmat, Tensor):
            raise ValueError("viewmat must be a CUDA float32 tensor")
        device = self.viewmat.device
        if device.type != "cuda":
            raise ValueError("camera tensors must be on CUDA")
        check_tensor("viewmat", self.viewmat, device, (4, 4))
        check_tensor("K", self.K, device, (3, 3))
        if torch.is_grad_enabled() and self.K.requires_grad:
            raise NotImplementedError("intrinsics gradients are not supported")
        cameras = prepare_cameras(
            self.viewmat[None, None],
            self.K.detach()[None, None],
            self.width,
            self.height,
            self.near_plane,
            self.far_plane,
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
    intrinsics = Ks.cpu()
    if (
        not bool(torch.isfinite(intrinsics).all())
        or bool((intrinsics[..., 0, 0] <= 0).any())
        or bool((intrinsics[..., 1, 1] <= 0).any())
    ):
        raise ValueError("Ks must be finite with positive fx/fy")
    expected_bottom = torch.tensor([0.0, 0.0, 1.0])
    if (
        not torch.equal(
            intrinsics[..., 2, :], expected_bottom.expand(batch_count, cameras, 3)
        )
        or bool((intrinsics[..., 0, 1] != 0).any())
        or bool((intrinsics[..., 1, 0] != 0).any())
    ):
        raise NotImplementedError("Ks must be a zero-skew pinhole intrinsic matrix")
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
    full = (projections.to(device) @ views).transpose(-1, -2).contiguous()
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
