"""Keep camera arithmetic in the renderer's float32 contract under Torch AMP."""

import torch
from torch import Tensor


def camera_matmul(left: Tensor, right: Tensor) -> Tensor:
    """AMP must not turn a float32 camera buffer into a half-precision array."""
    if torch.is_autocast_enabled():
        with torch.autocast(device_type="cuda", enabled=False):
            return left @ right
    return left @ right
