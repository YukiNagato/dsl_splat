"""Named per-view inputs and owned execution state for RaDe-GS."""

from dataclasses import dataclass

from torch import Tensor

from .types import ViewConfig
from .types import ProjectedGaussians
from ..runtime.dispatch import FrameBindings


@dataclass(slots=True)
class ViewInputs:
    means: Tensor
    quats: Tensor
    scales: Tensor
    opacities: Tensor
    features: Tensor
    extra_features: Tensor | None
    view: Tensor
    projection: Tensor
    intrinsics: Tensor
    background: Tensor | None


@dataclass(slots=True)
class PreparedView:
    batch: int
    camera: int
    inputs: ViewInputs
    geometry: ProjectedGaussians
    config: ViewConfig
    bindings: FrameBindings
