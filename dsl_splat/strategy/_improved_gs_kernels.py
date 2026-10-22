# Copyright (c) 2026 ImprovedGS authors and Zhejiang University.
# Adapted for DSL Splat from Improved-GS long_axis_split: Python/Warp port,
# strided arrays, and stable log-scale/logit arithmetic. See IMPROVED_GS_LICENSE.txt.
"""Long-axis split geometry; parameter and optimizer ownership stays in Python."""

import warp as wp

from ..primitives.quaternions import covariance_rotation

wp.set_module_options({"enable_backward": False})


@wp.kernel
def split_geometry(
    means: wp.array2d(dtype=float),
    quats: wp.array2d(dtype=float),
    scales: wp.array2d(dtype=float),
    logits: wp.array(dtype=float),
    selected: wp.array(dtype=wp.int64),
    distance: float,
    opacity_reduction: float,
    child_means: wp.array2d(dtype=float),
    child_scales: wp.array2d(dtype=float),
    child_logits: wp.array(dtype=float),
):
    i = wp.tid()
    gid = selected[i]
    n = selected.shape[0]
    longest = int(0)
    for axis in range(1, 3):
        if scales[gid, axis] > scales[gid, longest]:
            longest = axis
    q = wp.vec4(quats[gid, 0], quats[gid, 1], quats[gid, 2], quats[gid, 3])
    q /= wp.max(wp.length(q), 1.0e-12)
    rotation_t = covariance_rotation(q)
    # The support semi-axis is 3 sigma in the reference implementation.
    offset = 3.0 * distance * wp.exp(scales[gid, longest])
    minor_log_ratio = 0.5 * wp.log(1.0 - distance * distance)
    major_log_ratio = wp.log(1.0 - distance)
    for axis in range(3):
        delta = offset * rotation_t[longest, axis]
        child_means[i, axis] = means[gid, axis] + delta
        child_means[i + n, axis] = means[gid, axis] - delta
        ratio = minor_log_ratio
        if axis == longest:
            ratio = major_log_ratio
        value = scales[gid, axis] + ratio
        child_scales[i, axis] = value
        child_scales[i + n, axis] = value
    # logit(r*sigmoid(x)) = x+log(r)-softplus(x+log(1-r)).
    # This also keeps finite, saturated parent logits finite in the children.
    opacity = logits[gid]
    if opacity_reduction < 1.0:
        x = opacity + wp.log(1.0 - opacity_reduction)
        softplus = wp.max(x, 0.0) + wp.log(1.0 + wp.exp(-wp.abs(x)))
        opacity = opacity + wp.log(opacity_reduction) - softplus
    child_logits[i] = opacity
    child_logits[i + n] = opacity
