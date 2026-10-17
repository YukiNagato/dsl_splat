"""Frame-owned destinations for dense or packed auxiliary gradients."""

from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass(slots=True)
class AbsGradientTarget:
    target: Tensor
    gaussian_ids: Tensor | None = None

    @torch.no_grad()
    def write(self, gradient: Tensor) -> None:
        value = gradient if self.gaussian_ids is None else gradient[self.gaussian_ids]
        self.target.copy_(value)
