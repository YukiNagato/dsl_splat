"""Small arithmetic helpers preserving the CUDA reference's GLM conventions."""

import warp as wp

from ..primitives.math import glm_rotation


@wp.func_native(
    "return fmaf(a,b,c);",
    adj_snippet="adj_a += b * adj_ret; adj_b += a * adj_ret; adj_c += adj_ret;",
)
def fused_multiply_add(a: float, b: float, c: float) -> float:
    pass


@wp.func
def reference_rotation(q: wp.vec4):
    return glm_rotation(q)


@wp.func_grad(reference_rotation)
def adj_reference_rotation(q: wp.vec4, g: wp.mat33):
    r, x, y, z = q[0], q[1], q[2], q[3]
    wp.adjoint[q] += wp.vec4(
        2.0
        * (z * (g[0, 1] - g[1, 0]) + y * (g[2, 0] - g[0, 2]) + x * (g[1, 2] - g[2, 1])),
        2.0
        * (y * (g[0, 1] + g[1, 0]) + z * (g[0, 2] + g[2, 0]) + r * (g[1, 2] - g[2, 1]))
        - 4.0 * x * (g[1, 1] + g[2, 2]),
        2.0
        * (x * (g[0, 1] + g[1, 0]) + r * (g[2, 0] - g[0, 2]) + z * (g[1, 2] + g[2, 1]))
        - 4.0 * y * (g[0, 0] + g[2, 2]),
        2.0
        * (r * (g[0, 1] - g[1, 0]) + x * (g[0, 2] + g[2, 0]) + y * (g[1, 2] + g[2, 1]))
        - 4.0 * z * (g[0, 0] + g[1, 1]),
    )


@wp.func
def reference_matmul(a: wp.mat33, b: wp.mat33):
    result = wp.mat33()
    for row in range(3):
        for col in range(3):
            result[row, col] = fused_multiply_add(
                a[row, 2],
                b[2, col],
                fused_multiply_add(a[row, 0], b[0, col], a[row, 1] * b[1, col]),
            )
    return result


@wp.func
def point_row(matrix: wp.mat44, row: int, point: wp.vec3):
    xy = fused_multiply_add(matrix[row, 0], point[0], matrix[row, 1] * point[1])
    return fused_multiply_add(matrix[row, 2], point[2], xy) + matrix[row, 3]
