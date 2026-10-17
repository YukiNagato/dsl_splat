"""Optional, non-differentiable reductions from accepted pixels to Gaussians.

Masks affect statistics only. They never discard pixels from the rendered
image. Each request returns (..., cameras, N) values owned by that render call.
"""

from dataclasses import dataclass
from functools import cache
from typing import Sequence

import torch
from torch import Tensor
import warp as wp


@dataclass(frozen=True, slots=True)
class MaskedContributionCount:
    name: str
    mask: Tensor


@dataclass(frozen=True, slots=True)
class WeightedContributionSum:
    name: str
    pixel_weights: Tensor


StatisticsRequest = MaskedContributionCount | WeightedContributionSum
StatisticsKey = tuple[int, ...]


@wp.struct
class DeviceStatistics:
    pixel_values: wp.array2d(dtype=float)  # type: ignore[valid-type]
    counts: wp.array2d(dtype=int)  # type: ignore[valid-type]
    sums: wp.array2d(dtype=float)  # type: ignore[valid-type]


@dataclass(slots=True)
class PreparedStatistics:
    key: StatisticsKey
    pixel_values: Tensor
    counts: Tensor
    sums: Tensor


@cache
def make_observer(key: StatisticsKey) -> wp.Function:
    """Compile the selected formulas into the renderer's accepted-blend path."""
    rows = tuple(sum(other == kind for other in key[:i]) for i, kind in enumerate(key))

    @wp.func
    def observe(context: DeviceStatistics, gaussian: int, pixel: int, weight: float):
        for index in range(wp.static(len(key))):
            i = wp.static(index)
            row = wp.static(rows[i])
            value = context.pixel_values[i, pixel]
            if wp.static(key[i] == 0):
                if value != 0.0:
                    wp.atomic_add(context.counts, row, gaussian, 1)
            else:
                if value != 0.0:
                    wp.atomic_add(context.sums, row, gaussian, value * weight)

    return observe


def prepare_statistics(
    requests: Sequence[StatisticsRequest],
    *,
    batch_dims: tuple[int, ...],
    batch_count: int,
    cameras: int,
    n: int,
    width: int,
    height: int,
    device: torch.device,
) -> tuple[list[PreparedStatistics], dict[str, Tensor]]:
    """Bind pixel data per call; cache keys contain formulas, never Tensor values.

    Pixel maps may have shape (H,W), shared by all views, or the explicit shape
    (...,C,H,W). Boolean masks and float32 weight maps are constants for autograd.
    """
    names: set[str] = set()
    maps: list[Tensor] = []
    kinds: list[int] = []
    shape = batch_dims + (cameras, height, width)
    for request in requests:
        if not isinstance(request, (MaskedContributionCount, WeightedContributionSum)):
            raise TypeError("unsupported contribution statistics request")
        if (
            not isinstance(request.name, str)
            or not request.name
            or request.name in names
        ):
            raise ValueError("statistics names must be nonempty and unique")
        names.add(request.name)
        count = isinstance(request, MaskedContributionCount)
        value = (
            request.mask
            if isinstance(request, MaskedContributionCount)
            else request.pixel_weights
        )
        dtype = torch.bool if count else torch.float32
        if (
            not isinstance(value, Tensor)
            or value.device != device
            or value.dtype != dtype
        ):
            raise ValueError(
                f"statistics {request.name!r} must use {dtype} on {device}"
            )
        if value.requires_grad:
            raise ValueError("statistics pixel maps are non-differentiable constants")
        if tuple(value.shape) == (height, width):
            value = value.expand(batch_count, cameras, height, width)
        elif tuple(value.shape) == shape:
            value = value.reshape(batch_count, cameras, height, width)
        else:
            raise ValueError(
                f"statistics maps must have shape {(height, width)} or {shape}"
            )
        maps.append(value.to(torch.float32))
        kinds.append(0 if count else 1)
    key = tuple(kinds)
    pixel_values = torch.stack(maps, dim=2).reshape(batch_count, cameras, len(maps), -1)
    counts = torch.zeros(
        (batch_count, cameras, kinds.count(0), n), dtype=torch.int32, device=device
    )
    sums = torch.zeros(
        (batch_count, cameras, kinds.count(1), n), dtype=torch.float32, device=device
    )
    destinations = {0: counts, 1: sums}
    results = {
        request.name: destinations[kinds[i]][
            :, :, sum(k == kinds[i] for k in kinds[:i])
        ].view(batch_dims + (cameras, n))
        for i, request in enumerate(requests)
    }
    prepared = [
        PreparedStatistics(key, pixel_values[b, c], counts[b, c], sums[b, c])
        for b in range(batch_count)
        for c in range(cameras)
    ]
    return prepared, results


__all__ = ["MaskedContributionCount", "WeightedContributionSum", "StatisticsRequest"]
