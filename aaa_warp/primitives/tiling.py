"""Tile-key encoding, prefix scans and radix sorting shared by renderers.

Methods generate their own intersections and depth keys. These utilities only
manage the resulting arrays; they impose no projection or culling policy.
"""

from typing import TypedDict

import torch
from torch import Tensor
import warp as wp

from ..runtime.dispatch import FrameBindings, kernel_arg, launch


class _TileRanges(TypedDict):
    point_list: Tensor
    ranges: Tensor


class TileBins(_TileRanges, total=False):
    point_offsets: Tensor
    point_list_keys_unsorted: Tensor
    point_list_unsorted: Tensor
    point_list_keys: Tensor


@wp.func_native("return __float_as_uint(value);")
def float_bits(value: wp.float32) -> wp.uint32: ...


@wp.func
def sort_key(tile_id: int, depth: float):
    return (wp.uint64(wp.uint32(tile_id)) << wp.uint64(32)) | wp.uint64(
        float_bits(depth)
    )


@wp.kernel(enable_backward=False)
def identify_ranges(keys: wp.array(dtype=wp.uint64), ranges: wp.array2d(dtype=int)):
    i = wp.tid()
    current = wp.uint32(keys[i] >> wp.uint64(32))
    if i == 0:
        if current != wp.uint32(0xFFFFFFFF):
            ranges[int(current), 0] = 0
    else:
        previous = wp.uint32(keys[i - 1] >> wp.uint64(32))
        if current != previous:
            if previous != wp.uint32(0xFFFFFFFF):
                ranges[int(previous), 1] = i
            if current != wp.uint32(0xFFFFFFFF):
                ranges[int(current), 0] = i
    if i == keys.shape[0] - 1 and current != wp.uint32(0xFFFFFFFF):
        ranges[int(current), 1] = keys.shape[0]


def scan_counts(counts: Tensor, stream: wp.Stream) -> tuple[Tensor, int]:
    """Inclusive int32 offsets and the host-visible allocation size."""
    offsets = torch.empty((len(counts),), dtype=torch.int32, device=counts.device)
    if not len(counts):
        return offsets, 0
    with wp.ScopedStream(stream, sync_enter=False):
        wp.utils.array_scan(wp.from_torch(counts), wp.from_torch(offsets))
    return offsets, int(offsets[-1].item())


def sort_tile_pairs(
    keys: Tensor,
    ids: Tensor,
    count: int,
    ranges: Tensor,
    stream: wp.Stream,
    bindings: FrameBindings | None = None,
    *,
    sorted_keys: Tensor | None = None,
) -> None:
    """Sort 2*count scratch arrays in place and fill zero-initialized ranges."""
    if not count:
        return
    with wp.ScopedStream(stream, sync_enter=False):
        wp.utils.radix_sort_pairs(
            wp.from_torch(keys, dtype=wp.uint64),
            wp.from_torch(ids),
            count,
            end_bit=32 + len(ranges).bit_length(),
        )
    launch(
        identify_ranges,
        dim=count,
        inputs=[
            kernel_arg(
                keys[:count] if sorted_keys is None else sorted_keys,
                wp.uint64,
                bindings,
            )
        ],
        outputs=[kernel_arg(ranges, bindings=bindings)],
        stream=stream,
        bindings=bindings,
    )
