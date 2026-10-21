"""Neighbor selection and debug images shared by geometry training demos."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from dsl_splat.pgsr import get_img_grad_weight

if __package__:
    from .colmap_dataset import Dataset, ImageFrame
else:
    from colmap_dataset import Dataset, ImageFrame


@torch.no_grad()
def render_debug_image(
    frame: ImageFrame,
    rgb: Tensor,
    renders: dict[str, Any],
    geometry_weights: Tensor | None = None,
    *,
    scalar_key: str = "render_distances",
    depth_key: str = "depths",
) -> np.ndarray:
    """Return a 2x4 geometry debug grid as an RGB uint8 HWC image.

    Top: ground truth, rendered RGB, rendered normal, plane distance.
    Bottom: geometry weight, plane depth, depth normal, image-gradient weight.
    Geometry weights are zero when multi-view supervision has not run.
    RaDe selects expected/median depth for the two scalar panels via their keys.
    """

    def to_rgb(value: Tensor) -> np.ndarray:
        array = value.detach().float().cpu().numpy()
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        return (array.clip(0, 1) * 255).astype(np.uint8)

    def colorize(value: Tensor, *, normalize: bool = False) -> np.ndarray:
        array = value.detach().float().cpu().numpy()
        valid = np.isfinite(array)
        if normalize:
            valid &= array > 0
        scaled = np.zeros_like(array)
        if valid.any():
            values = array[valid]
            if normalize:
                low, high = values.min(), values.max()
                scaled[valid] = (values - low) / max(float(high - low), 1e-20)
            else:
                scaled[valid] = values
        color = cv2.applyColorMap(
            (scaled.clip(0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        color[~valid] = 0
        return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    height, width = frame.image.shape[:2]
    depth = renders[depth_key].reshape(height, width)
    distance = renders[scalar_key].reshape(height, width)
    normals = renders["normals"].reshape(height, width, 3)
    depth_normals = renders["depth_normals"].reshape(height, width, 3)
    if geometry_weights is None:
        geometry_weights = depth.new_zeros((height, width))
    image_weights = (1 - get_img_grad_weight(frame.image)).clamp(0, 1).square()
    top = np.concatenate(
        (
            to_rgb(frame.image),
            to_rgb(rgb),
            to_rgb((normals + 1) / 2),
            colorize(distance, normalize=True),
        ),
        axis=1,
    )
    bottom = np.concatenate(
        (
            colorize(geometry_weights.reshape(height, width)),
            colorize(depth, normalize=True),
            to_rgb((depth_normals + 1) / 2),
            colorize(image_weights),
        ),
        axis=1,
    )
    return np.concatenate((top, bottom), axis=0)


def save_debug_image(
    directory: Path,
    step: int,
    frame: ImageFrame,
    rgb: Tensor,
    renders: dict[str, Any],
    geometry_weights: Tensor | None = None,
    *,
    scalar_key: str = "render_distances",
    depth_key: str = "depths",
) -> Path:
    """Save a debug grid, named by training step and reference frame."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = frame.im_path.stem if frame.im_path is not None else "frame"
    path = directory / f"{step:05d}_{name}.jpg"
    image = render_debug_image(
        frame,
        rgb,
        renders,
        geometry_weights,
        scalar_key=scalar_key,
        depth_key=depth_key,
    )
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not save debug image: {path}")
    return path


class MultiViewDataset(Dataset):
    def __init__(
        self,
        *args,
        multi_view_num: int = 1,
        multi_view_max_angle: float = 10,
        multi_view_min_dis: float = 0.1,
        multi_view_max_dis: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.multi_view_num = multi_view_num
        self.multi_view_max_angle = multi_view_max_angle
        self.multi_view_min_dis = multi_view_min_dis
        self.multi_view_max_dis = multi_view_max_dis
        self._prepare_views()

    def _prepare_views(self) -> None:
        views = torch.stack([frame.w2c for frame in self.frames])
        # Translation t is not the camera center; C_world = -R.T @ t.
        centers = -(views[:, :3, :3].transpose(-1, -2) @ views[:, :3, 3, None]).squeeze(
            -1
        )
        directions = F.normalize(views[:, 2, :3], dim=-1)
        distances = torch.cdist(centers, centers)
        angles = torch.acos((directions @ directions.T).clamp(-1, 1)).rad2deg()
        self.frame_nearest_ids: list[list[int]] = []
        for index, frame in enumerate(self.frames):
            # Distance is primary, angle breaks ties, as in PGSR's lexsort.
            order = torch.argsort(angles[index], stable=True)
            order = order[torch.argsort(distances[index, order], stable=True)]
            valid = (
                (order != index)
                & (angles[index, order] < self.multi_view_max_angle)
                & (distances[index, order] > self.multi_view_min_dis)
                & (distances[index, order] < self.multi_view_max_dis)
            )
            self.frame_nearest_ids.append(order[valid][: self.multi_view_num].tolist())
            frame.gray_image = (
                0.299 * frame.image[..., 0]
                + 0.587 * frame.image[..., 1]
                + 0.114 * frame.image[..., 2]
            )
