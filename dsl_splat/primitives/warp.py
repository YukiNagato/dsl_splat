"""Warp-wide collectives for scalar kernels.

All 32 lanes must participate. Warp's tile reductions synchronize a whole block;
these two intrinsics let independent warps skip work without block barriers.
"""

from typing import Any

import warp as wp


@wp.func_native("return __any_sync(0xffffffff, predicate);")
def any_active(predicate: bool) -> bool: ...


@wp.func_native("return __shfl_down_sync(0xffffffff, value, offset);")
def shuffle_down(value: float, offset: int) -> float: ...


@wp.func
def sum_lanes(value: float) -> float:
    """Sum 32 values; only lane zero holds the complete result."""
    for step in range(wp.static(5)):
        value += shuffle_down(value, wp.static(16 >> step))
    return value


@wp.func
def sum_vector_lanes(value: wp.types.vector(Any, wp.float32)):
    for component in range(wp.static(len(value))):
        value[wp.static(component)] = sum_lanes(value[wp.static(component)])
    return value
