"""Projected tile intersections and radix sorting with caller-supplied depths."""

from functools import cache

import torch
import warp as wp

from .geometry import tile_bounds
from .intersections import TileMode, TilePairs, make_intersector
from torch import Tensor
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


@cache
def _make_duplicate(tile_mode: TileMode):
    intersect = make_intersector(tile_mode)

    @wp.kernel(module="unique", enable_backward=False)
    def duplicate_precise(
        centers: wp.array(dtype=wp.vec2),
        conics: wp.array(dtype=wp.vec4),
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
            offset = int(0)
            if i > 0:
                offset = offsets[i - 1]
            output = TilePairs()
            output.keys = keys
            output.ids = ids
            intersect(
                centers[i], conics[i], radii[i], gx, gy, i, depths[i], offset, output
            )

    return duplicate_precise


@torch.no_grad()
def bin_and_sort(
    centers: Tensor,
    radii: Tensor,
    tile_counts: Tensor,
    depths: Tensor,
    width: int,
    height: int,
    bindings: FrameBindings,
    *,
    conics: Tensor | None = None,
    tile_mode: TileMode = "radius",
) -> TileBins:
    if tile_mode != "radius" and conics is None:
        raise ValueError("SnugBox/AccuTile require projected conics")
    stream = current_stream(radii.device, bindings=bindings)
    offsets, count = scan_counts(tile_counts, stream)
    keys = torch.empty(2 * count, dtype=torch.int64, device=radii.device)
    ids = torch.empty(2 * count, dtype=torch.int32, device=radii.device)
    gx, gy = (width + 15) // 16, (height + 15) // 16
    ranges = torch.zeros((gx * gy, 2), dtype=torch.int32, device=radii.device)
    if count:
        launch(
            duplicate if tile_mode == "radius" else _make_duplicate(tile_mode),
            dim=len(radii),
            inputs=[
                kernel_arg(centers, wp.vec2, bindings),
                *(
                    []
                    if tile_mode == "radius"
                    else [kernel_arg(conics, wp.vec4, bindings)]
                ),
                kernel_arg(radii, bindings=bindings),
                kernel_arg(depths, bindings=bindings),
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
