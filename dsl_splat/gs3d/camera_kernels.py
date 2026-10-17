"""Per-camera projection preparation and its pose VJP in Warp.

These kernels keep float32 products and additions separate to match the previous
Torch small-matrix arithmetic. Gaussian projection keeps its own CUDA-matched
FMA conventions in projection.py/math.py.
"""

import warp as wp


@wp.kernel(module="unique", enable_backward=False, module_options={"fuse_fp": False})
def prepare_kernel(
    views: wp.array3d(dtype=float),
    Ks: wp.array3d(dtype=float),
    width: int,
    height: int,
    with_centers: int,
    full: wp.array3d(dtype=float),
    inverse: wp.array3d(dtype=float),
    centers: wp.array2d(dtype=float),
    errors: wp.array(dtype=int),
):
    i = wp.tid()
    finite = bool(True)
    for row in range(3):
        for col in range(3):
            finite = finite and wp.isfinite(Ks[i, row, col])
    error = int(0)
    if not finite or Ks[i, 0, 0] <= 0.0 or Ks[i, 1, 1] <= 0.0:
        error = 2
    elif (
        Ks[i, 0, 1] != 0.0
        or Ks[i, 1, 0] != 0.0
        or Ks[i, 2, 0] != 0.0
        or Ks[i, 2, 1] != 0.0
        or Ks[i, 2, 2] != 1.0
    ):
        error = 1
    errors[i] = error
    if error != 0:
        return
    px = 2.0 * Ks[i, 0, 0] / float(width)
    py = 2.0 * Ks[i, 1, 1] / float(height)
    cx = 2.0 * Ks[i, 0, 2] / float(width) - 1.0
    cy = 2.0 * Ks[i, 1, 2] / float(height) - 1.0
    # Preserve the separate products/additions of Torch's small batched P @ V.
    # Fusing them changes off-center projections at floating-point boundaries.
    for col in range(4):
        full[i, 0, col] = px * views[i, 0, col] + cx * views[i, 2, col]
        full[i, 1, col] = py * views[i, 1, col] + cy * views[i, 2, col]
        full[i, 2, col] = views[i, 2, col]
        full[i, 3, col] = views[i, 2, col]
    if with_centers != 0:
        view = wp.mat44()
        for row in range(4):
            for col in range(4):
                view[row, col] = views[i, row, col]
        if wp.determinant(view) == 0.0:
            errors[i] = 3
            return
        inv = wp.inverse(view)
        for row in range(4):
            for col in range(4):
                inverse[i, row, col] = inv[row, col]
        for row in range(3):
            centers[i, row] = inv[row, 3]


@wp.kernel(module="unique", enable_backward=False, module_options={"fuse_fp": False})
def backward_kernel(
    Ks: wp.array3d(dtype=float),
    gradient: wp.array3d(dtype=float),
    width: int,
    height: int,
    inverse: wp.array3d(dtype=float),
    grad_centers: wp.array2d(dtype=float),
    has_projection: int,
    has_centers: int,
    grad_views: wp.array3d(dtype=float),
):
    i = wp.tid()
    px = 2.0 * Ks[i, 0, 0] / float(width)
    py = 2.0 * Ks[i, 1, 1] / float(height)
    cx = 2.0 * Ks[i, 0, 2] / float(width) - 1.0
    cy = 2.0 * Ks[i, 1, 2] / float(height) - 1.0
    for row in range(4):
        for col in range(4):
            value = float(0.0)
            if has_projection != 0:
                if row == 0:
                    value = px * gradient[i, 0, col]
                elif row == 1:
                    value = py * gradient[i, 1, col]
                elif row == 2:
                    value = (
                        (cx * gradient[i, 0, col] + cy * gradient[i, 1, col])
                        + gradient[i, 2, col]
                        + gradient[i, 3, col]
                    )
            if has_centers != 0:
                # d(V^-1) = -V^-1 dV V^-1, valid for general 4x4 poses.
                center_vjp = float(0.0)
                for k in range(3):
                    center_vjp += inverse[i, k, row] * grad_centers[i, k]
                value -= center_vjp * inverse[i, col, 3]
            grad_views[i, row, col] = value
