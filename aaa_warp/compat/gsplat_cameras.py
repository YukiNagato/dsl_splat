"""Backward-compatible camera helper; OpenCV cameras now belong to the core."""

from ..cameras import CameraBatch as Cameras
from ..cameras import prepare_cameras as _prepare_cameras
from ..inputs import Inputs


def prepare_cameras(
    inputs: Inputs, width: int, height: int, near_plane: float, far_plane: float
) -> Cameras:
    return _prepare_cameras(
        inputs.viewmats, inputs.Ks, width, height, near_plane, far_plane
    )


__all__ = ["Cameras", "prepare_cameras"]
