"""Degree 0--3 spherical harmonics, matching AAA forward_common.h."""

import warp as wp


@wp.func
def evaluate_sh(
    i: int, degree: int, mean: wp.vec3, camera: wp.vec3, sh: wp.array2d(dtype=wp.vec3)
):
    d = (mean - camera) / wp.length(mean - camera)
    x, y, z = d[0], d[1], d[2]
    result = 0.28209479177387814 * sh[i, 0]
    if degree > 0:
        result = (
            result
            - 0.4886025119029199 * y * sh[i, 1]
            + 0.4886025119029199 * z * sh[i, 2]
            - 0.4886025119029199 * x * sh[i, 3]
        )
        if degree > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + 1.0925484305920792 * xy * sh[i, 4]
                - 1.0925484305920792 * yz * sh[i, 5]
                + 0.31539156525252005 * (2.0 * zz - xx - yy) * sh[i, 6]
                - 1.0925484305920792 * xz * sh[i, 7]
                + 0.5462742152960396 * (xx - yy) * sh[i, 8]
            )
            if degree > 2:
                result = (
                    result
                    - 0.5900435899266435 * y * (3.0 * xx - yy) * sh[i, 9]
                    + 2.890611442640554 * xy * z * sh[i, 10]
                    - 0.4570457994644658 * y * (4.0 * zz - xx - yy) * sh[i, 11]
                    + 0.3731763325901154
                    * z
                    * (2.0 * zz - 3.0 * xx - 3.0 * yy)
                    * sh[i, 12]
                    - 0.4570457994644658 * x * (4.0 * zz - xx - yy) * sh[i, 13]
                    + 1.445305721320277 * z * (xx - yy) * sh[i, 14]
                    - 0.5900435899266435 * x * (xx - 3.0 * yy) * sh[i, 15]
                )
    return result + wp.vec3(0.5)
