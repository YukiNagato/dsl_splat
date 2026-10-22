"""Shared camera preparation and 3DGS-specific tile-selection configuration."""

from dataclasses import dataclass

from torch import Tensor

from ..cameras import PinholeCamera
from ..runtime.dispatch import FrameBindings
from ..splatting.intersections import TileMode

from ..splatting.cameras import (
    ViewConfig as _ViewConfig,
    _PrepareCameras,
    prepare_camera as _prepare_camera,
    prepare_cameras,
    prepare_cameras_with_centers,
)


@dataclass(frozen=True, slots=True)
class ViewConfig(_ViewConfig):
    tile_mode: TileMode = "radius"


def prepare_camera(
    camera: PinholeCamera, radius_clip: float, bindings: FrameBindings
) -> tuple[Tensor, Tensor, Tensor, ViewConfig]:
    view, projection, intrinsics, config = _prepare_camera(
        camera, radius_clip, bindings
    )
    return (
        view,
        projection,
        intrinsics,
        ViewConfig(
            config.width, config.height, config.near, config.far, config.radius_clip
        ),
    )
