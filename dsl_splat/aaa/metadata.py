"""Dense/packed training statistics and frame-owned screen-gradient destinations."""

from dataclasses import dataclass
from typing import TypedDict

import torch
from torch import Tensor

from .frame import AbsGradientTarget, FrameOptions, PreparedFrame
from .config import FrameConfig
from .types import GaussianInputs
from ..inputs import Inputs
from ..training import prepare_training_metadata
from ..runtime.precision import camera_matmul


class Metadata(TypedDict):
    width: int
    height: int
    tile_size: int
    tile_width: int
    tile_height: int
    n_cameras: int
    packed: bool
    backend: str
    with_eval3d: bool
    conics: None
    supports_means2d_grad: bool
    screen_gradient_mode: str
    batch_ids: Tensor | None
    camera_ids: Tensor | None
    gaussian_ids: Tensor | None
    radii: Tensor
    means2d: Tensor
    opacities: Tensor
    tiles_per_gauss: Tensor


class ExtendedMetadata(Metadata, total=False):
    render_extra_signals: Tensor
    statistics: dict[str, Tensor]


@dataclass(slots=True)
class ViewFrame:
    batch: int
    camera: int
    config: FrameConfig
    inputs: GaussianInputs
    prepared: PreparedFrame
    intrinsics: tuple[float, float, float, float]


@torch.no_grad()
def _geometry(frames: list[ViewFrame], layout: Inputs) -> dict[str, Tensor]:
    values: dict[str, list[Tensor]] = {
        name: [] for name in ("radii", "means2d", "opacities", "tiles_per_gauss")
    }
    for frame in frames:
        state = frame.prepared.state
        # Auxiliary projected-center metadata for densification only. Rendering
        # and culling use the AAA stages; this no-grad projection changes neither.
        view, means = (
            layout.viewmats[frame.batch, frame.camera],
            layout.means[frame.batch],
        )
        points = camera_matmul(means, view[:3, :3].T) + view[:3, 3]
        fx, fy, cx, cy = frame.intrinsics
        z = points[:, 2]
        valid = state["radii"] > 0
        screen = torch.stack(
            (
                fx * points[:, 0] / z.clamp_min(1e-12) + cx,
                fy * points[:, 1] / z.clamp_min(1e-12) + cy,
            ),
            dim=-1,
        )
        values["radii"].append(
            torch.where(valid[:, None], state["rects2D"].ceil().to(torch.int32), 0)
        )
        values["means2d"].append(torch.where(valid[:, None], screen, 0))
        values["opacities"].append(torch.where(valid, state["opacity"], 0))
        values["tiles_per_gauss"].append(state["tiles_touched"])
    return {
        name: (rows[0] if len(rows) == 1 else torch.stack(rows)).reshape(
            (layout.batch_count, layout.cameras, layout.n) + tuple(rows[0].shape[1:])
        )
        for name, rows in values.items()
    }


def build_metadata(
    frames: list[ViewFrame],
    layout: Inputs,
    width: int,
    height: int,
    *,
    packed: bool,
    absgrad: bool,
) -> tuple[ExtendedMetadata, Tensor | None, list[FrameOptions]]:
    """Build auxiliary inputs before rendering; no scatter graph in inference."""
    dense = _geometry(frames, layout)
    training = prepare_training_metadata(dense, layout, packed=packed, absgrad=absgrad)
    geometry = training.values
    meta: ExtendedMetadata = dict(
        width=width,
        height=height,
        tile_size=16,
        tile_width=(width + 15) // 16,
        tile_height=(height + 15) // 16,
        n_cameras=layout.cameras,
        packed=packed,
        backend="aaa-warp",
        with_eval3d=True,
        conics=None,
        supports_means2d_grad=True,
        screen_gradient_mode="rigid_footprint_translation",
        batch_ids=geometry["batch_ids"],
        camera_ids=geometry["camera_ids"],
        gaussian_ids=geometry["gaussian_ids"],
        radii=geometry["radii"],
        means2d=geometry["means2d"],
        opacities=geometry["opacities"],
        tiles_per_gauss=geometry["tiles_per_gauss"],
    )
    options = [
        FrameOptions(
            prepared=frame.prepared,
            screen_grad=layout.requires_screen_grad,
            absgrad_target=target,
        )
        for frame, target in zip(frames, training.absolute_targets)
    ]
    return meta, training.screen, options
