"""Small shared-memory and synchronization primitives for 256-thread tiles.

Traversal, Gaussian evaluation and blending stay in Python/Warp. These helpers
only expose CUDA shared storage and block collectives to scalar Warp kernels.
"""

import warp as wp


@wp.func_native("__syncthreads();")
def synchronize(): ...


@wp.func_native("return __syncthreads_count(done) == 256;")
def all_done(done: bool) -> bool: ...


@wp.func_native("__shared__ int values[256]; return wp::array_t<int>(values, 256);")
def shared_ids() -> wp.array(dtype=int): ...


@wp.func_native(
    "__shared__ wp::vec_t<2,float> values[256]; return wp::array_t<wp::vec_t<2,float>>(values, 256);"
)
def shared_centers() -> wp.array(dtype=wp.vec2): ...


@wp.func_native(
    "__shared__ wp::vec_t<4,float> values[256]; return wp::array_t<wp::vec_t<4,float>>(values, 256);"
)
def shared_conics() -> wp.array(dtype=wp.vec4): ...
