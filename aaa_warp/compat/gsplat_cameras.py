"""Conversion of zero-skew pinhole cameras to AAA's matrix conventions."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from ..settings import ExtendedSettings, GaussianRasterizationSettings
from .gsplat_inputs import Inputs


@dataclass(slots=True)
class Cameras:
    views: Tensor
    transposed_views: Tensor
    projection: Tensor
    inverse_projection: Tensor
    centers: Tensor
    intrinsics: list[list[tuple[float, float, float, float]]]
    width: int
    height: int

    def config(
        self, batch: int, camera: int, background: Tensor, settings: ExtendedSettings
    ) -> GaussianRasterizationSettings:
        fx, fy, _, _ = self.intrinsics[batch][camera]
        return GaussianRasterizationSettings(
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
        )


def prepare_cameras(
    inputs: Inputs, width: int, height: int, near_plane: float, far_plane: float
) -> Cameras:
    batch_count, cameras, device = inputs.batch_count, inputs.cameras, inputs.device
    # Read the small intrinsic array once: the core API takes scalar FOVs.
    intrinsics = inputs.Ks.cpu()
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
    views = inputs.viewmats
    inverse_views = torch.linalg.inv(views)
    projections = torch.zeros(batch_count, cameras, 4, 4, dtype=torch.float32)
    projections[..., 0, 0] = 2 * intrinsics[..., 0, 0] / width
    projections[..., 1, 1] = 2 * intrinsics[..., 1, 1] / height
    projections[..., 0, 2] = 2 * intrinsics[..., 0, 2] / width - 1
    projections[..., 1, 2] = 2 * intrinsics[..., 1, 2] / height - 1
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
    inverse_full = torch.linalg.inv(full).contiguous()
    values = [
        [(k[0][0], k[1][1], k[0][2], k[1][2]) for k in batch]
        for batch in intrinsics.tolist()
    ]
    return Cameras(
        views,
        views.transpose(-1, -2).contiguous(),
        full,
        inverse_full,
        inverse_views[..., :3, 3].contiguous(),
        values,
        width,
        height,
    )
