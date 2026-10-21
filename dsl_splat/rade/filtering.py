"""Explicit Mip-style 3D filtering used by the official RaDe-GS frontend."""

import torch
from torch import Tensor


def apply_3d_filter(
    scales: Tensor, opacities: Tensor, filter_3d: Tensor
) -> tuple[Tensor, Tensor]:
    """Apply per-Gaussian filter widths, retaining ordinary PyTorch gradients.

    scales: (...,N,3), opacities: (...,N), filter_3d: (...,N,1).
    Widths are computed from the training cameras and refreshed after growth by
    the caller; they are intentionally not a hidden renderer-owned cache.
    """
    if (
        filter_3d.shape != scales.shape[:-1] + (1,)
        or opacities.shape != scales.shape[:-1]
    ):
        raise ValueError(
            "require scales (...,N,3), opacities (...,N), filter_3d (...,N,1)"
        )
    squares = scales.square()
    filtered = squares + filter_3d.square()
    coefficient = (squares.prod(-1) / filtered.prod(-1)).sqrt()
    return filtered.sqrt(), opacities * coefficient
