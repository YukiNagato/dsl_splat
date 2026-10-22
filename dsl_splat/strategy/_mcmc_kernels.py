# SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# MCMC equations adapted from gsplat (see README.md for revisions and changes).
# Modified for DSL Splat: Python/Warp kernels, FP64 relocation accumulation,
# saturated-opacity handling, strided input support and fused covariance action.
"""Non-differentiable MCMC relocation and position perturbation in Warp."""

import warp as wp

from ..primitives.quaternions import covariance_rotation

wp.set_module_options({"enable_backward": False})


@wp.kernel
def relocation(
    opacities: wp.array(dtype=float),
    scales: wp.array2d(dtype=float),
    ratios: wp.array(dtype=wp.int64),
    binoms: wp.array2d(dtype=wp.float64),
    min_opacity: float,
    new_opacities: wp.array(dtype=float),
    new_scales: wp.array2d(dtype=float),
):
    idx = wp.tid()
    # gsplat limits multiplicity to the size of its binomial table (51).
    n = wp.clamp(int(ratios[idx]), 1, binoms.shape[0])
    opacity = wp.float64(wp.min(opacities[idx], 1.0 - 1.1920928955078125e-7))
    new_opacity = wp.float64(1.0) - wp.pow(
        wp.float64(1.0) - opacity, wp.float64(1.0) / wp.float64(n)
    )
    new_opacity = wp.clamp(
        new_opacity,
        wp.float64(min_opacity),
        wp.float64(1.0 - 1.1920928955078125e-7),
    )
    # Eq. 9: FP64 avoids cancellation in the alternating binomial sum.
    denominator = wp.float64(0.0)
    for i in range(1, n + 1):
        power = new_opacity
        for k in range(i):
            denominator += binoms[i - 1, k] * power / wp.sqrt(wp.float64(k + 1))
            power *= -new_opacity
    coefficient = wp.float64(opacities[idx]) / denominator
    new_opacities[idx] = float(new_opacity)
    for axis in range(3):
        new_scales[idx, axis] = float(coefficient * wp.float64(scales[idx, axis]))


@wp.kernel
def perturb_positions(
    means: wp.array2d(dtype=float),
    quats: wp.array2d(dtype=float),
    log_scales: wp.array2d(dtype=float),
    logits: wp.array(dtype=float),
    noise: wp.array2d(dtype=float),
    noise_scale: float,
    opacity_t: float,
    opacity_k: float,
):
    idx = wp.tid()
    q = wp.vec4(quats[idx, 0], quats[idx, 1], quats[idx, 2], quats[idx, 3])
    q /= wp.max(wp.length(q), 1.0e-12)
    world_to_local = covariance_rotation(q)
    opacity = 1.0 / (1.0 + wp.exp(-logits[idx]))
    weight = noise_scale / (1.0 + wp.exp(opacity_k * (opacity - opacity_t)))
    local_noise = world_to_local * (
        wp.vec3(noise[idx, 0], noise[idx, 1], noise[idx, 2]) * weight
    )
    # MCMC uses covariance * noise, not its square root * noise.
    for axis in range(3):
        local_noise[axis] *= wp.exp(2.0 * log_scales[idx, axis])
    delta = wp.transpose(world_to_local) * local_noise
    for axis in range(3):
        means[idx, axis] += delta[axis]
