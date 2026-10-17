"""One CUDA global-memory atomic for the analytic backward.

Warp's generic atomic also handles shared memory. These gradient arrays always
reside in global memory, so specify the address space as CUDA does. Keep byte
strides to support array views; queue logic and derivative math stay in Warp.
"""

import warp as wp


@wp.func_native("""
    float previous;
    asm volatile("atom.global.add.f32 %0, [%1], %2;"
        : "=f"(previous)
        : "l"(reinterpret_cast<float*>(reinterpret_cast<char*>(values.data)
                  + index * values.strides[0])), "f"(value)
        : "memory");
    return previous;
    """)
def global_atomic_add(values: wp.array(dtype=float), index: int, value: float) -> float:
    pass
