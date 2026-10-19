"""PGSR plane geometry with RGB/HWC tensors and OpenCV camera matrices.

K follows the renderers' half-integer pixel centers. Integer array indices use
principal points shifted by -0.5. Depths are camera Z; normals face the camera.
"""

# PGSR adaptations: Copyright 2023-2024, 3D Vision Group, Zhejiang University.
# See PGSR_LICENSE.txt and README.md for upstream terms and local changes.

from typing import Literal, overload

import torch
from torch import Tensor
import torch.nn.functional as F

from .types import PlaneMaps, Reprojection

MIN_DEPTH = 1e-3
PLANE_EPS = 1e-6


def quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    """Convert wxyz quaternions to rotation matrices, including non-unit inputs."""
    r, i, j, k = F.normalize(quaternions, dim=-1, eps=1e-12).unbind(-1)
    return torch.stack(
        (
            1 - 2 * (j * j + k * k),
            2 * (i * j - k * r),
            2 * (i * k + j * r),
            2 * (i * j + k * r),
            1 - 2 * (i * i + k * k),
            2 * (j * k - i * r),
            2 * (i * k - j * r),
            2 * (j * k + i * r),
            1 - 2 * (i * i + j * j),
        ),
        dim=-1,
    ).reshape(quaternions.shape[:-1] + (3, 3))


@overload
def get_smallest_axis(
    quats: Tensor, scales: Tensor, return_idx: Literal[False] = False
) -> Tensor: ...


@overload
def get_smallest_axis(
    quats: Tensor, scales: Tensor, return_idx: Literal[True]
) -> tuple[Tensor, Tensor]: ...


def get_smallest_axis(
    quats: Tensor, scales: Tensor, return_idx: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    rotations = quaternion_to_matrix(quats)
    indices = scales.argmin(dim=-1)
    axes = rotations.gather(2, indices[:, None, None].expand(-1, 3, 1)).squeeze(-1)
    return (axes, indices) if return_idx else axes


def get_normal(
    camera_center: Tensor, means: Tensor, scales: Tensor, quats: Tensor
) -> Tensor:
    normals = get_smallest_axis(quats, scales)
    facing = (normals * (camera_center - means)).sum(dim=-1, keepdim=True)
    return torch.where(facing < 0, -normals, normals)


def plane_features(
    viewmat: Tensor, means: Tensor, scales: Tensor, quats: Tensor
) -> Tensor:
    """Camera-space normal and positive plane distance, before alpha blending."""
    rotation, translation = viewmat[:3, :3], viewmat[:3, 3]
    center = -(rotation.T @ translation)
    normals = get_normal(center, means, scales, quats) @ rotation.T
    camera_points = means @ rotation.T + translation
    distances = (normals * camera_points).sum(dim=-1, keepdim=True).abs()
    return torch.cat((normals, distances), dim=-1)


def pixel_rays(K: Tensor, height: int, width: int) -> tuple[Tensor, Tensor]:
    """Integer array coordinates and rays; K must use integer-center coordinates."""
    y, x = torch.meshgrid(
        torch.arange(height, device=K.device),
        torch.arange(width, device=K.device),
        indexing="ij",
    )
    pixels = torch.stack((x, y), dim=-1).to(torch.int32)
    rays = torch.stack(
        ((x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1], torch.ones_like(x)),
        dim=-1,
    ).to(K.dtype)
    return pixels, rays


def integer_center_intrinsics(K: Tensor) -> Tensor:
    """Convert the rasterizer's half-integer centers to NCC's integer centers."""
    return K - K.new_tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])


def valid_depth(depth: Tensor, mask: Tensor | None = None) -> Tensor:
    valid = torch.isfinite(depth) & (depth > MIN_DEPTH)
    if mask is not None:
        valid = valid & mask.reshape(depth.shape)
    return valid


def project_points(
    points: Tensor, K: Tensor, height: int, width: int
) -> tuple[Tensor, Tensor]:
    """Mask invalid points before division, so their backward path stays finite."""
    valid = torch.isfinite(points).all(dim=-1) & (points[..., 2] > MIN_DEPTH)
    # Reject the frustum in homogeneous coordinates before perspective division.
    # This also excludes huge finite X/Y that would overflow during projection.
    focal = torch.stack((K[0, 0], K[1, 1]))
    lower = -K[:2, 2] / focal
    upper = (K.new_tensor([width - 1, height - 1]) - K[:2, 2]) / focal
    valid = valid & (
        (points[..., :2] >= lower * points[..., 2:3])
        & (points[..., :2] <= upper * points[..., 2:3])
    ).all(dim=-1)
    safe_points = torch.where(valid[..., None], points, 0.0)
    safe_z = torch.where(valid, safe_points[..., 2], 1.0)
    xy = safe_points[..., :2] / safe_z[..., None]
    pixels = xy * torch.stack((K[0, 0], K[1, 1])) + K[:2, 2]
    valid = valid & torch.isfinite(pixels).all(dim=-1)
    return torch.where(valid[..., None], pixels, 0.0), valid


def sample_depth(
    depth: Tensor,
    points_in_camera_space: Tensor,
    K: Tensor,
    depth_valid: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Bilinearly sample positive camera-Z depth; both results have shape (N,)."""
    height, width = depth.shape
    K = integer_center_intrinsics(K.to(depth))
    pixels, inside = project_points(points_in_camera_space, K, height, width)
    inside = (
        inside
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] <= width - 1)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] <= height - 1)
    )
    # Never pass NaN/Inf or out-of-bounds coordinates into grid_sample.
    pixels = torch.where(inside[:, None], pixels, 0.0)
    grid = torch.stack(
        (
            2 * pixels[:, 0] / max(width - 1, 1) - (1 if width > 1 else 0),
            2 * pixels[:, 1] / max(height - 1, 1) - (1 if height > 1 else 0),
        ),
        dim=-1,
    ).reshape(1, -1, 1, 2)
    source_valid = valid_depth(depth, depth_valid)
    safe_depth = torch.where(source_valid, depth, 0.0)
    # The validity channel rejects a footprint that interpolates through a hole.
    source = torch.stack((safe_depth, source_valid.to(depth.dtype)))[None]
    sampled = F.grid_sample(
        source, grid, mode="bilinear", padding_mode="border", align_corners=True
    )[0, :, :, 0]
    mask = inside & valid_depth(sampled[0]) & (sampled[1] >= 1 - 1e-6)
    return torch.where(mask, sampled[0], 0.0), mask


def depth_to_normal(
    depth: Tensor, rays: Tensor, valid: Tensor
) -> tuple[Tensor, Tensor]:
    """Camera-facing HWC normals from central differences and a valid stencil."""
    if min(depth.shape) < 3:
        return depth.new_zeros((*depth.shape, 3)), torch.zeros_like(valid)
    safe_depth = torch.where(valid, depth, 0.0)
    points = safe_depth[..., None] * rays
    vertical = points[2:, 1:-1] - points[:-2, 1:-1]
    horizontal = points[1:-1, 2:] - points[1:-1, :-2]
    cross = torch.cross(vertical, horizontal, dim=-1)
    stencil = (
        valid[1:-1, 1:-1]
        & valid[2:, 1:-1]
        & valid[:-2, 1:-1]
        & valid[1:-1, 2:]
        & valid[1:-1, :-2]
        & torch.isfinite(cross).all(dim=-1)
        & (torch.linalg.vector_norm(cross, dim=-1) > 1e-8)
    )
    safe_cross = torch.where(stencil[..., None], cross, 0.0)
    normals = F.normalize(safe_cross, dim=-1, eps=1e-8)
    return (
        F.pad(normals, (0, 0, 1, 1, 1, 1)),
        F.pad(stencil, (1, 1, 1, 1), value=False),
    )


def plane_maps(
    normals: Tensor, distances: Tensor, alpha: Tensor, K: Tensor
) -> PlaneMaps:
    """Recover camera-Z depth from equally alpha-weighted normal/distance maps."""
    _, rays = pixel_rays(
        integer_center_intrinsics(K), normals.shape[0], normals.shape[1]
    )
    finite = (
        torch.isfinite(normals).all(dim=-1)
        & torch.isfinite(distances)
        & torch.isfinite(alpha)
    )
    safe_normals = torch.where(finite[..., None], normals, 0.0)
    safe_distances = torch.where(finite, distances, 0.0)
    denominator = -(safe_normals * rays).sum(dim=-1)
    mask = finite & (alpha > 1e-6) & (denominator > PLANE_EPS) & (safe_distances > 0)
    # Adding epsilon can change the sign near zero. Replace invalid denominators.
    safe_denominator = torch.where(mask, denominator, 1.0)
    depths = torch.where(mask, safe_distances / safe_denominator, 0.0)
    mask = mask & valid_depth(depths)
    depths = torch.where(mask, depths, 0.0)
    depth_normals, normal_valid = depth_to_normal(depths, rays, mask)
    # PGSR compares the blended normal against an alpha-weighted depth normal.
    safe_alpha = torch.where(torch.isfinite(alpha), alpha, 0.0).detach()
    depth_normals = depth_normals * safe_alpha[..., None]
    return {
        "depths": depths[None, ..., None],
        "normals": normals[None],
        "render_distances": distances[None, ..., None],
        "depth_normals": depth_normals[None],
        "depth_valid": mask[None, ..., None],
        "depth_normal_valid": normal_valid[None, ..., None],
    }


def multiview_geometry(
    depth: Tensor,
    normal: Tensor,
    neighbor_depth: Tensor,
    view: Tensor,
    K: Tensor,
    near: Tensor,
    near_K: Tensor,
    depth_valid: Tensor | None = None,
    neighbor_valid: Tensor | None = None,
    pixel_noise_th: float = 1.0,
) -> Reprojection:
    """Dense reprojection, masks and loss, suitable for one compiled graph.

    Depths are (H,W); the normal map is (H,W,3). Camera tensors are already on
    the depth device. Dynamic selection and Warp NCC are separate from this
    function, so compilation does not capture data-dependent output shapes.
    """
    height, width = depth.shape
    K_integer = integer_center_intrinsics(K)
    pixels, rays = pixel_rays(K_integer, height, width)
    ref_valid = valid_depth(depth, depth_valid)
    safe_depth = torch.where(ref_valid, depth, 0.0)
    points = (safe_depth[..., None] * rays).reshape(-1, 3)
    world = (points - view[:3, 3]) @ view[:3, :3]
    neighbor_points = world @ near[:3, :3].T + near[:3, 3]
    sampled_depth, mask = sample_depth(
        neighbor_depth, neighbor_points, near_K, neighbor_valid
    )
    mask = mask & ref_valid.reshape(-1)
    safe_points = torch.where(mask[:, None], neighbor_points, 0.0)
    safe_z = torch.where(mask, safe_points[:, 2], 1.0)
    reconstructed = safe_points / safe_z[:, None] * sampled_depth[:, None]
    world_back = (reconstructed - near[:3, 3]) @ near[:3, :3]
    reference_back = world_back @ view[:3, :3].T + view[:3, 3]
    projected, back_valid = project_points(reference_back, K_integer, height, width)
    mask = (
        mask
        & back_valid
        & (projected[:, 0] >= 0)
        & (projected[:, 0] <= width - 1)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] <= height - 1)
    )
    reference_pixels = pixels.reshape(-1, 2).to(depth.dtype)
    projected = torch.where(mask[:, None], projected, reference_pixels)
    noise = torch.linalg.vector_norm(projected - reference_pixels, dim=-1)
    mask = mask & torch.isfinite(noise) & (noise < pixel_noise_th)
    weights = torch.where(mask, torch.exp(-noise.detach()), 0.0)
    # A masked reduction also gives a differentiable zero for an empty mask.
    loss = torch.where(mask, weights * noise, 0.0).sum() / mask.sum().clamp_min(1)
    flat_normal = normal.reshape(-1, 3)
    cosine = (flat_normal * rays.reshape(-1, 3)).sum(dim=-1)
    normal_valid = (
        torch.isfinite(flat_normal).all(dim=-1)
        & torch.isfinite(cosine)
        & (cosine.abs() > PLANE_EPS)
    )
    return {
        "pixels": pixels,
        "weights": weights,
        "valid": mask,
        "ncc_valid": mask & normal_valid,
        "loss": loss,
    }
