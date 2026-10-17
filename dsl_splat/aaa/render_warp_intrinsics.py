"""Small CUDA primitives missing from Warp's scalar Python kernel API.

Only shared allocation, subgroup communication and aligned matrix loads live here.
Queue ownership, sorting and rendering belong to render_hierarchical_warp.py.
All snippets are compiled and cached by Warp; there is no extension build.

The renderer has independent 16-lane and four-lane queues, so it needs masked
warp synchronization inside divergent branches. Block-wide tile collectives
cannot directly replace these intrinsics. Returned shared arrays use ordinary
Warp indexing; queue reads, writes and sorting require no native snippets.
"""

import warp as wp


@wp.func_native("""
    __shared__ float values[1536];
    return wp::array_t<float>(values,1536);
""")
def shared_depths() -> wp.array(dtype=float):
    pass


@wp.func_native("""
    __shared__ int values[1536];
    return wp::array_t<int>(values,1536);
""")
def shared_ids() -> wp.array(dtype=int):
    pass


@wp.func_native("__syncwarp(mask);")
def sync(mask: wp.uint32):
    pass


@wp.func_native("return __any_sync(mask, predicate);")
def any_active(mask: wp.uint32, predicate: bool) -> bool:
    pass


@wp.func_native("return __popc(__ballot_sync(mask, predicate));")
def count_active(mask: wp.uint32, predicate: bool) -> int:
    pass


@wp.func_native("return __shfl_sync(mask, value, source, width);")
def shuffle_int(mask: wp.uint32, value: int, source: int, width: int) -> int:
    pass


@wp.func_native("return __shfl_sync(mask, value, source, width);")
def shuffle_float(mask: wp.uint32, value: float, source: int, width: int) -> float:
    pass


@wp.func_native("""
    wp::mat_t<4,4,float> result;
    const float4* rows = reinterpret_cast<const float4*>(matrices.data) + id*4;
    #pragma unroll
    for (int row=0; row<4; ++row) {
        const float4 v=rows[row];
        result.data[row][0]=v.x; result.data[row][1]=v.y;
        result.data[row][2]=v.z; result.data[row][3]=v.w;
    }
    return result;
""")
def load_matrix(matrices: wp.array(dtype=wp.mat44), id: int) -> wp.mat44:
    # Host orchestration guarantees 16-byte alignment. The native float4 load
    # avoids sixteen scalar loads from a plain Warp matrix-array access.
    pass
