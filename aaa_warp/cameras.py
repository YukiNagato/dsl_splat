"""Public OpenCV pinhole camera definitions, independent of rendering method."""

from dataclasses import dataclass
import math
from numbers import Integral

from torch import Tensor
import torch


def read_intrinsics(Ks: Tensor) -> Tensor:
    """Read and validate the small (B,C,3,3) intrinsic array once per call."""
    intrinsics = Ks.cpu()
    if (
        not bool(torch.isfinite(intrinsics).all())
        or bool((intrinsics[..., 0, 0] <= 0).any())
        or bool((intrinsics[..., 1, 1] <= 0).any())
    ):
        raise ValueError("Ks must be finite with positive fx/fy")
    bottom = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    if (
        not torch.equal(intrinsics[..., 2, :], bottom.expand(Ks.shape[:-2] + (3,)))
        or bool((intrinsics[..., 0, 1] != 0).any())
        or bool((intrinsics[..., 1, 0] != 0).any())
    ):
        raise NotImplementedError("Ks must be a zero-skew pinhole intrinsic matrix")
    return intrinsics


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
    """Rigid world-to-camera pose and zero-skew K, with arbitrary cx/cy.

    Axes point right, down, and forward. Each rendering method documents how
    it uses near/far; the camera definition does not impose a clipping policy.
    """

    viewmat: Tensor
    K: Tensor
    width: int
    height: int
    near_plane: float = 0.01
    far_plane: float = 1e10
