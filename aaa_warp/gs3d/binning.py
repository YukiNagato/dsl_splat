"""Original 3DGS square intersections and center-Z sorting."""

import torch
import warp as wp

from .projection import ProjectedGaussians, tile_bounds
from ..primitives.tiling import TileBins, sort_key, scan_counts, sort_tile_pairs
from ..runtime.dispatch import FrameBindings, kernel_arg, launch
from ..runtime.interop import current_stream


@wp.kernel(enable_backward=False)
def duplicate(
    centers: wp.array(dtype=wp.vec2),
    radii: wp.array(dtype=int),
    depths: wp.array(dtype=float),
    offsets: wp.array(dtype=int),
    gx: int,
    gy: int,
    keys: wp.array(dtype=wp.uint64),
    ids: wp.array(dtype=int),
):
    i = wp.tid()
    if radii[i] > 0:
        bounds = tile_bounds(centers[i], radii[i], gx, gy)
        offset = int(0)
        if i > 0:
            offset = offsets[i - 1]
        for y in range(bounds[1], bounds[3]):
            for x in range(bounds[0], bounds[2]):
                keys[offset] = sort_key(y * gx + x, depths[i])
                ids[offset] = i
                offset += 1


@torch.no_grad()
def bin_and_sort(
    state: ProjectedGaussians, width: int, height: int, bindings: FrameBindings
) -> TileBins:
    stream = current_stream(state.radii.device, bindings=bindings)
    offsets, count = scan_counts(state.tile_counts, stream)
    keys = torch.empty(2 * count, dtype=torch.int64, device=state.radii.device)
    ids = torch.empty(2 * count, dtype=torch.int32, device=state.radii.device)
    gx, gy = (width + 15) // 16, (height + 15) // 16
    ranges = torch.zeros((gx * gy, 2), dtype=torch.int32, device=state.radii.device)
    if count:
        launch(
            duplicate,
            dim=len(state.radii),
            inputs=[
                kernel_arg(state.centers, wp.vec2, bindings),
                kernel_arg(state.radii, bindings=bindings),
                kernel_arg(state.depths, bindings=bindings),
                kernel_arg(offsets, bindings=bindings),
                gx,
                gy,
            ],
            outputs=[
                kernel_arg(keys[:count], wp.uint64, bindings),
                kernel_arg(ids[:count], bindings=bindings),
            ],
            stream=stream,
            bindings=bindings,
        )
        sort_tile_pairs(keys, ids, count, ranges, stream, bindings)
    return dict(point_list=ids[:count], ranges=ranges)
