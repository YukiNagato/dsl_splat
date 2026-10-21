"""Projected footprint storage and CUDA-compatible small geometry operations."""

from dataclasses import dataclass
from torch import Tensor
import warp as wp
from .math import point_row


@dataclass(slots=True)
class ProjectedFootprint:
    centers: Tensor
    conics: Tensor


@dataclass(slots=True)
class ProjectedGaussians(ProjectedFootprint):
    radii: Tensor
    tile_counts: Tensor
    depths: Tensor


@wp.func
def tile_bounds(center: wp.vec2, radius: int, gx: int, gy: int):
    # Keep CUDA getRect's integer truncation and +BLOCK-1 arithmetic.
    return wp.vec4i(
        wp.clamp(int((center[0] - float(radius)) / 16.0), 0, gx),
        wp.clamp(int((center[1] - float(radius)) / 16.0), 0, gy),
        wp.clamp(int(((center[0] + float(radius)) + 16.0 - 1.0) / 16.0), 0, gx),
        wp.clamp(int(((center[1] + float(radius)) + 16.0 - 1.0) / 16.0), 0, gy),
    )


@wp.func
def clamp_point(point: wp.vec3, limx: float, limy: float):
    return wp.vec3(
        wp.clamp(point[0] / point[2], -limx, limx) * point[2],
        wp.clamp(point[1] / point[2], -limy, limy) * point[2],
        point[2],
    )


@wp.func_grad(clamp_point)
def adj_clamp_point(point: wp.vec3, limx: float, limy: float, gradient: wp.vec3):
    # Original 3DGS treats saturated x/y as constants in this VJP. RaDe-GS
    # uses a different z derivative and supplies its own local clamp function.
    x, y = point[0] / point[2], point[1] / point[2]
    wp.adjoint[point] += wp.vec3(
        gradient[0] * float(x >= -limx and x <= limx),
        gradient[1] * float(y >= -limy and y <= limy),
        gradient[2],
    )


@wp.func
def invert_covariance(cov: wp.vec3):
    a, b, c = cov[0], cov[1], cov[2]
    inverse = 1.0 / (a * c - b * b)
    return wp.vec3(c * inverse, -b * inverse, a * inverse)


@wp.func_grad(invert_covariance)
def adj_invert_covariance(cov: wp.vec3, gradient: wp.vec3):
    a, b, c = cov[0], cov[1], cov[2]
    det = a * c - b * b
    factor = 1.0 / (det * det + 0.0000001)
    x, y, z = gradient[0], gradient[1], gradient[2]
    wp.adjoint[cov] += factor * wp.vec3(
        -c * c * x + b * c * y + (det - a * c) * z,
        2.0 * b * c * x - (det + 2.0 * b * b) * y + 2.0 * a * b * z,
        -a * a * z + a * b * y + (det - a * c) * x,
    )


@wp.func
def load_matrix(matrix: wp.array2d(dtype=float)):
    return wp.mat44(
        matrix[0, 0],
        matrix[0, 1],
        matrix[0, 2],
        matrix[0, 3],
        matrix[1, 0],
        matrix[1, 1],
        matrix[1, 2],
        matrix[1, 3],
        matrix[2, 0],
        matrix[2, 1],
        matrix[2, 2],
        matrix[2, 3],
        matrix[3, 0],
        matrix[3, 1],
        matrix[3, 2],
        matrix[3, 3],
    )


@wp.func
def transform_point(point: wp.vec3, matrix: wp.mat44):
    return wp.vec4(
        point_row(matrix, 0, point),
        point_row(matrix, 1, point),
        point_row(matrix, 2, point),
        point_row(matrix, 3, point),
    )


@wp.func
def pixel_center(mean: wp.vec3, projection: wp.mat44, width: int, height: int):
    clip = transform_point(mean, projection)
    inverse_w = 1.0 / (clip[3] + 0.0000001)
    # CUDA ndc2Pix uses double literals: promote after the float32 perspective
    # divide, then round only the final pixel coordinate back to float32.
    ndc_x = wp.float64(clip[0] * inverse_w)
    ndc_y = wp.float64(clip[1] * inverse_w)
    return wp.vec2(
        float(
            ((ndc_x + wp.float64(1.0)) * wp.float64(width) - wp.float64(1.0))
            * wp.float64(0.5)
        ),
        float(
            ((ndc_y + wp.float64(1.0)) * wp.float64(height) - wp.float64(1.0))
            * wp.float64(0.5)
        ),
    )
