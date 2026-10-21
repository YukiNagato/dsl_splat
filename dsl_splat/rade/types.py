"""RaDe-GS projection configuration and ray-plane attributes."""

from dataclasses import dataclass
from torch import Tensor
from ..splatting.cameras import ViewConfig as SplatConfig
from ..splatting.geometry import ProjectedGaussians as SplatGeometry


@dataclass(frozen=True, slots=True)
class ViewConfig(SplatConfig):
    kernel_size: float = 0.0


@dataclass(slots=True)
class ProjectedGaussians(SplatGeometry):
    ray_planes: Tensor
    normals: Tensor
