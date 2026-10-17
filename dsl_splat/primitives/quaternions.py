"""Covariance rotation convention used by the reference Gaussian rasterizers."""

import warp as wp


@wp.func
def covariance_rotation(q: wp.vec4):
    """Transpose of the usual wxyz local-to-world rotation; q is normalized."""
    r, x, y, z = q[0], q[1], q[2], q[3]
    return wp.mat33(
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + r * z),
        2.0 * (x * z - r * y),
        2.0 * (x * y - r * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + r * x),
        2.0 * (x * z + r * y),
        2.0 * (y * z - r * x),
        1.0 - 2.0 * (x * x + y * y),
    )
