"""Original 3DGS square intersections and center-Z sorting."""

from ..splatting.intersections import TileMode
from .projection import ProjectedGaussians
from ..splatting.binning import bin_and_sort as bin_arrays
from ..primitives.tiling import TileBins
from ..runtime.dispatch import FrameBindings


def bin_and_sort(
    state: ProjectedGaussians,
    width: int,
    height: int,
    bindings: FrameBindings,
    *,
    tile_mode: TileMode = "radius",
) -> TileBins:
    return bin_arrays(
        state.centers,
        state.radii,
        state.tile_counts,
        state.depths,
        width,
        height,
        bindings,
        conics=state.conics,
        tile_mode=tile_mode,
    )
