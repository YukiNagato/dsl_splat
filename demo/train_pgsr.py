"""PGSR-style plane supervision with RaDe-GS's Warp patch NCC.

Images use HWC; w2c is an ordinary OpenCV matrix, so row-vector points transform
as p_camera = p_world @ w2c[:3, :3].T + w2c[:3, 3]. Rasterizer K coordinates
have half-integer pixel centers. Geometry/NCC use integer array coordinates,
with their principal points shifted by -0.5 to describe the same rays.
This is a minimal training example, not a full reproduction of PGSR's trainer.
"""

from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any, Callable, Literal

import cv2
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm
import tyro

from dsl_splat.strategy.default import DefaultStrategy

if __package__:
    from .colmap_dataset import Dataset, ImageFrame
    from .model import GaussianContainer
    from .utils import IndexRandomChooser, MovingAverager, psnr
    from .warp_patch_ncc import warp_patch_ncc
else:
    from colmap_dataset import Dataset, ImageFrame
    from model import GaussianContainer
    from utils import IndexRandomChooser, MovingAverager, psnr
    from warp_patch_ncc import warp_patch_ncc


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


def get_pgsr_extra_feature(
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


def get_points_depth_in_depth_map(
    fov_frame: ImageFrame,
    depth: Tensor,
    points_in_camera_space: Tensor,
    depth_valid: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Bilinearly sample positive camera-Z depth; both results have shape (N,)."""
    height, width = fov_frame.height, fov_frame.width
    depth = depth.reshape(height, width)
    K = integer_center_intrinsics(fov_frame.K.to(depth))
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
) -> dict[str, Tensor]:
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


def get_img_grad_weight(image: Tensor) -> Tensor:
    """PGSR image-gradient weights for RGB HWC, including constant/tiny images."""
    height, width = image.shape[:2]
    if min(height, width) < 3:
        return image.new_ones((height, width))
    horizontal = (image[1:-1, 2:] - image[1:-1, :-2]).abs().mean(dim=-1)
    vertical = (image[:-2, 1:-1] - image[2:, 1:-1]).abs().mean(dim=-1)
    gradient = torch.maximum(horizontal, vertical)
    gradient = (gradient - gradient.min()) / (
        gradient.max() - gradient.min()
    ).clamp_min(1e-8)
    return F.pad(gradient, (1, 1, 1, 1), value=1.0)


def single_view_loss(image: Tensor, renders: dict[str, Any]) -> Tensor:
    weights = (1 - get_img_grad_weight(image)).clamp(0, 1).detach().square()
    normal = renders["normals"][0]
    depth_normal = renders["depth_normals"][0]
    mask = renders["depth_normal_valid"][0, ..., 0]
    normal = torch.where(mask[..., None], normal, 0.0)
    depth_normal = torch.where(mask[..., None], depth_normal, 0.0)
    error = (depth_normal - normal).abs().sum(dim=-1)
    return torch.where(mask, weights * error, 0.0).mean()


@torch.no_grad()
def render_debug_image(
    frame: ImageFrame,
    rgb: Tensor,
    renders: dict[str, Any],
    geometry_weights: Tensor | None = None,
) -> np.ndarray:
    """Return PGSR's 2x4 debug grid as an RGB uint8 HWC image.

    Top: ground truth, rendered RGB, rendered normal, plane distance.
    Bottom: geometry weight, plane depth, depth normal, image-gradient weight.
    Geometry weights are zero when multi-view supervision has not run.
    """

    def to_rgb(value: Tensor) -> np.ndarray:
        array = value.detach().float().cpu().numpy()
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        return (array.clip(0, 1) * 255).astype(np.uint8)

    def colorize(value: Tensor, *, normalize: bool = False) -> np.ndarray:
        array = value.detach().float().cpu().numpy()
        valid = np.isfinite(array)
        if normalize:
            valid &= array > 0
        scaled = np.zeros_like(array)
        if valid.any():
            values = array[valid]
            if normalize:
                low, high = values.min(), values.max()
                scaled[valid] = (values - low) / max(float(high - low), 1e-20)
            else:
                scaled[valid] = values
        color = cv2.applyColorMap(
            (scaled.clip(0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        color[~valid] = 0
        return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    height, width = frame.image.shape[:2]
    depth = renders["depths"].reshape(height, width)
    distance = renders["render_distances"].reshape(height, width)
    normals = renders["normals"].reshape(height, width, 3)
    depth_normals = renders["depth_normals"].reshape(height, width, 3)
    if geometry_weights is None:
        geometry_weights = depth.new_zeros((height, width))
    image_weights = (1 - get_img_grad_weight(frame.image)).clamp(0, 1).square()
    top = np.concatenate(
        (
            to_rgb(frame.image),
            to_rgb(rgb),
            to_rgb((normals + 1) / 2),
            colorize(distance, normalize=True),
        ),
        axis=1,
    )
    bottom = np.concatenate(
        (
            colorize(geometry_weights.reshape(height, width)),
            colorize(depth, normalize=True),
            to_rgb((depth_normals + 1) / 2),
            colorize(image_weights),
        ),
        axis=1,
    )
    return np.concatenate((top, bottom), axis=0)


def save_debug_image(
    directory: Path,
    step: int,
    frame: ImageFrame,
    rgb: Tensor,
    renders: dict[str, Any],
    geometry_weights: Tensor | None = None,
) -> Path:
    """Save one PGSR debug grid, named by training step and reference frame."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = frame.im_path.stem if frame.im_path is not None else "frame"
    path = directory / f"{step:05d}_{name}.jpg"
    image = render_debug_image(frame, rgb, renders, geometry_weights)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not save debug image: {path}")
    return path


class MultiViewLoss:
    def __init__(
        self, pixel_noise_th: float, sample_num: int, *, half_step: bool = False
    ):
        if pixel_noise_th <= 0 or not math.isfinite(pixel_noise_th) or sample_num <= 0:
            raise ValueError("pixel_noise_th and sample_num must be positive")
        self.pixel_noise_th = pixel_noise_th
        self.sample_num = sample_num
        self.half_step = half_step

    def __call__(
        self,
        view_frame: ImageFrame,
        view_renders: dict[str, Any],
        near_frame: ImageFrame,
        near_renders: dict[str, Any],
        *,
        debug_info: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        height, width = view_frame.height, view_frame.width
        depth = view_renders["depths"].reshape(height, width)
        neighbor_depth = near_renders["depths"]
        K = integer_center_intrinsics(view_frame.K.to(depth))
        view = view_frame.w2c.to(depth)
        near = near_frame.w2c.to(depth)
        pixels, rays = pixel_rays(K, height, width)
        ref_valid = valid_depth(depth, view_renders.get("depth_valid"))
        safe_depth = torch.where(ref_valid, depth, 0.0)
        points = (safe_depth[..., None] * rays).reshape(-1, 3)
        # camera -> world -> neighbor camera, using ordinary OpenCV matrices.
        world = (points - view[:3, 3]) @ view[:3, :3]
        neighbor_points = world @ near[:3, :3].T + near[:3, 3]
        sampled_depth, mask = get_points_depth_in_depth_map(
            near_frame, neighbor_depth, neighbor_points, near_renders.get("depth_valid")
        )
        mask = mask & ref_valid.reshape(-1)
        safe_points = torch.where(mask[:, None], neighbor_points, 0.0)
        safe_z = torch.where(mask, safe_points[:, 2], 1.0)
        reconstructed = safe_points / safe_z[:, None] * sampled_depth[:, None]
        world_back = (reconstructed - near[:3, 3]) @ near[:3, :3]
        reference_back = world_back @ view[:3, :3].T + view[:3, 3]
        projected, back_valid = project_points(reference_back, K, height, width)
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
        mask = mask & torch.isfinite(noise) & (noise < self.pixel_noise_th)
        weights = torch.exp(-noise.detach())
        if debug_info is not None:
            debug_info["geometry_weights"] = (
                torch.where(mask, weights, 0.0).reshape(height, width).detach()
            )
        valid_indices = mask.nonzero(as_tuple=True)[0]

        # Empty differentiable sums avoid both NaN * 0 and a CPU zero tensor.
        zero = safe_depth.reshape(-1)[:0].sum() + neighbor_depth.reshape(-1)[:0].sum()
        if valid_indices.numel() == 0:
            return zero, zero
        geo_loss = (weights[valid_indices] * noise[valid_indices]).mean()

        normal = view_renders["normals"].reshape(-1, 3)
        plane_cosine = (normal * rays.reshape(-1, 3)).sum(dim=-1)
        normal_valid = (
            torch.isfinite(normal).all(dim=-1)
            & torch.isfinite(plane_cosine)
            & (plane_cosine.abs() > PLANE_EPS)
        )
        valid_indices = valid_indices[normal_valid[valid_indices]]
        if valid_indices.numel() > self.sample_num:
            selection = torch.randperm(valid_indices.numel(), device=depth.device)[
                : self.sample_num
            ]
            valid_indices = valid_indices[selection]
        if valid_indices.numel() == 0:
            return zero, geo_loss

        # p_neighbor = R_relative @ p_reference + t_relative.
        rotation = near[:3, :3] @ view[:3, :3].T
        translation = near[:3, 3] - rotation @ view[:3, 3]
        gray_r = view_frame.gray_image
        gray_n = near_frame.gray_image
        if gray_r is None or gray_n is None:
            raise ValueError("PGSRDataset must prepare grayscale images before NCC")
        score, ncc_valid = warp_patch_ncc(
            depth.reshape(-1)[valid_indices],
            normal[valid_indices],
            pixels.reshape(-1, 2)[valid_indices],
            rotation.T.contiguous(),
            translation,
            gray_r.to(depth),
            gray_n.to(depth),
            view_frame.fx,
            view_frame.fy,
            view_frame.cx - 0.5,
            view_frame.cy - 0.5,
            near_frame.fx,
            near_frame.fy,
            near_frame.cx - 0.5,
            near_frame.cy - 0.5,
            half_step=self.half_step,
        )
        error = (1 - score).clamp(0, 2)
        ncc_mask = ncc_valid & torch.isfinite(score) & (error < 0.9)
        if not ncc_mask.any():
            return zero, geo_loss
        ncc_loss = (error[ncc_mask] * weights[valid_indices][ncc_mask]).mean()
        return ncc_loss, geo_loss


class PGSRDataset(Dataset):
    def __init__(
        self,
        *args,
        multi_view_num: int = 1,
        multi_view_max_angle: float = 10,
        multi_view_min_dis: float = 0.1,
        multi_view_max_dis: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.multi_view_num = multi_view_num
        self.multi_view_max_angle = multi_view_max_angle
        self.multi_view_min_dis = multi_view_min_dis
        self.multi_view_max_dis = multi_view_max_dis
        self._prepare_views()

    def _prepare_views(self) -> None:
        views = torch.stack([frame.w2c for frame in self.frames])
        # Translation t is not the camera center; C_world = -R.T @ t.
        centers = -(views[:, :3, :3].transpose(-1, -2) @ views[:, :3, 3, None]).squeeze(
            -1
        )
        directions = F.normalize(views[:, 2, :3], dim=-1)
        distances = torch.cdist(centers, centers)
        angles = torch.acos((directions @ directions.T).clamp(-1, 1)).rad2deg()
        self.frame_nearest_ids: list[list[int]] = []
        for index, frame in enumerate(self.frames):
            # Distance is primary, angle breaks ties, as in PGSR's lexsort.
            order = torch.argsort(angles[index], stable=True)
            order = order[torch.argsort(distances[index, order], stable=True)]
            valid = (
                (order != index)
                & (angles[index, order] < self.multi_view_max_angle)
                & (distances[index, order] > self.multi_view_min_dis)
                & (distances[index, order] < self.multi_view_max_dis)
            )
            self.frame_nearest_ids.append(order[valid][: self.multi_view_num].tolist())
            frame.gray_image = (
                0.299 * frame.image[..., 0]
                + 0.587 * frame.image[..., 1]
                + 0.114 * frame.image[..., 2]
            )


@dataclass
class Config:
    # Data/model parameters.
    data_root: Path = Path("data")
    debug_save_path: Path | None = None
    im_stride: int = 8
    ply_save_path: Path = Path("splat.ply")
    sh_degree: int = 3
    sh_degree_interval: int = 2000
    random_gs_num: int = 50000
    rasterizer_type: Literal["3dgs", "aaa"] = "3dgs"

    # Densification parameters.
    prune_opa: float = 0.05
    grow_grad2d: float = 0.0004
    grow_scale3d: float = 0.01
    refine_start_iter: int = 500
    refine_stop_iter: int = 15000
    reset_every: int = 5000
    refine_every: int = 100
    absgrad: bool = True

    # Training parameters.
    train_iters: int = 30000
    means_lr: float = 1e-4
    scales_lr: float = 0.01
    quats_lr: float = 0.001
    opacities_lr: float = 0.05
    features_dc_lr: float = 0.0025
    features_rest_lr: float = 0.0025 / 20

    # PGSR parameters; the multi-view factor weights NCC and geometry separately.
    pgsr_flatten_loss_factor: float = 100.0
    pgsr_view_supervise_start_iter: int = 7000
    pgsr_single_view_loss_factor: float = 0.015
    pgsr_multi_view_ncc_loss_factor: float = 0.15
    pgsr_multi_view_geo_loss_factor: float = 0.03
    pgsr_multi_view_pixel_noise_th: float = 1.0
    pgsr_multi_view_sample_num: int = 102400
    pgsr_multi_view_half_step: bool = False  # True: 0.5-pixel; False: 1-pixel.
    pgsr_multi_view_max_angle: float = 30
    pgsr_multi_view_min_dis: float = 0.01
    pgsr_multi_view_max_dis: float = 1.5
    pgsr_multi_view_num: int = 8


def render_with_rasterizer(
    rasterizer: Callable,
    gaussian: GaussianContainer,
    K: Tensor,
    viewmat: Tensor,
    H: int,
    W: int,
    sh_degree_to_use: int | None = None,
    *,
    absgrad: bool = True,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    params = gaussian.gauss_params
    means, quats = params["means"], params["quats"]
    scales = params["scales"].exp()
    colors = torch.cat((params["features_dc"][:, None], params["features_rest"]), dim=1)
    if gaussian.sh_degree == 0:
        colors = torch.sigmoid(colors[:, 0])
        degree = None
    else:
        degree = gaussian.sh_degree if sh_degree_to_use is None else sh_degree_to_use
    extra = get_pgsr_extra_feature(viewmat, means, scales, quats)
    render, alpha, info = rasterizer(
        means=means,
        quats=quats,
        scales=scales,
        opacities=params["opacities"].sigmoid().squeeze(-1),
        colors=colors,
        viewmats=viewmat[None],
        Ks=K[None],
        width=W,
        height=H,
        packed=False,
        near_plane=MIN_DEPTH,
        far_plane=1e10,
        render_mode="RGB",
        sh_degree=degree,
        absgrad=absgrad,
        extra_signals=extra,
    )
    # gsplat adapters return RGB separately from their rendered extra signals.
    planes = info["render_extra_signals"][0]
    info.update(plane_maps(planes[..., :3], planes[..., 3], alpha[0, ..., 0], K))
    return render, alpha, info


def main(cfg: Config) -> None:
    device = torch.device("cuda")
    dataset = PGSRDataset(
        data_root=cfg.data_root,
        im_stride=cfg.im_stride,
        multi_view_num=cfg.pgsr_multi_view_num,
        multi_view_max_angle=cfg.pgsr_multi_view_max_angle,
        multi_view_min_dis=cfg.pgsr_multi_view_min_dis,
        multi_view_max_dis=cfg.pgsr_multi_view_max_dis,
    )
    if cfg.rasterizer_type == "3dgs":
        from dsl_splat.gs3d.gsplat import rasterization
    else:
        from dsl_splat.aaa.gsplat import rasterization

    gaussian = GaussianContainer(
        xyz=dataset.points_xyz,
        rgb=dataset.points_rgb,
        num_gs=cfg.random_gs_num,
        sh_degree=cfg.sh_degree,
    ).to(device)
    chooser = IndexRandomChooser(len(dataset))
    loss_average, psnr_average = MovingAverager(10), MovingAverager(10)
    rates = {
        "means": cfg.means_lr,
        "scales": cfg.scales_lr,
        "quats": cfg.quats_lr,
        "opacities": cfg.opacities_lr,
        "features_dc": cfg.features_dc_lr,
        "features_rest": cfg.features_rest_lr,
    }
    optimizers = {
        key: torch.optim.Adam([gaussian.gauss_params[key]], lr=rate)
        for key, rate in rates.items()
    }
    strategy = DefaultStrategy(
        prune_opa=cfg.prune_opa,
        grow_grad2d=cfg.grow_grad2d,
        grow_scale3d=cfg.grow_scale3d,
        refine_start_iter=cfg.refine_start_iter,
        refine_stop_iter=cfg.refine_stop_iter,
        reset_every=cfg.reset_every,
        refine_every=cfg.refine_every,
        absgrad=cfg.absgrad,
        revised_opacity=False,
        verbose=True,
    )
    strategy.check_sanity(gaussian.gauss_params, optimizers)
    # Tune scene_scale and the learning rates for your reconstruction's units.
    state = strategy.initialize_state(scene_scale=1.0)
    multi_view = MultiViewLoss(
        cfg.pgsr_multi_view_pixel_noise_th,
        cfg.pgsr_multi_view_sample_num,
        half_step=cfg.pgsr_multi_view_half_step,
    )
    for step in tqdm(range(cfg.train_iters), desc="train"):
        debug_info = {} if cfg.debug_save_path is not None and step % 200 == 0 else None
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        data_idx = chooser()
        frame = dataset[data_idx]
        degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)

        def render_frame(camera: ImageFrame):
            return render_with_rasterizer(
                rasterization,
                gaussian,
                camera.K.to(device),
                camera.w2c.to(device),
                camera.height,
                camera.width,
                degree,
                absgrad=cfg.absgrad,
            )

        render, alpha, info = render_frame(frame)
        strategy.step_pre_backward(gaussian.gauss_params, optimizers, state, step, info)
        rgb, gt = render[0, ..., :3], frame.image.to(device)
        losses = {"main_loss": (gt - rgb).abs().mean()}
        visible = (info["radii"][0] > 0).all(dim=-1)
        if visible.any():
            min_scale = gaussian.gauss_params["scales"][visible].exp().amin(dim=-1)
            losses["flatten_loss"] = cfg.pgsr_flatten_loss_factor * min_scale.mean()
        if step > cfg.pgsr_view_supervise_start_iter:
            losses["single_view_loss"] = (
                cfg.pgsr_single_view_loss_factor * single_view_loss(gt, info)
            )
            neighbors = dataset.frame_nearest_ids[data_idx]
            if neighbors:
                neighbor = dataset[random.choice(neighbors)]
                _, _, near_info = render_frame(neighbor)
                ncc, geometry = multi_view(
                    frame, info, neighbor, near_info, debug_info=debug_info
                )
                losses["multi_view_ncc"] = cfg.pgsr_multi_view_ncc_loss_factor * ncc
                losses["multi_view_geometry"] = (
                    cfg.pgsr_multi_view_geo_loss_factor * geometry
                )

        if debug_info is not None:
            save_debug_image(
                cfg.debug_save_path,
                step,
                frame,
                rgb,
                info,
                geometry_weights=debug_info.get("geometry_weights"),
            )
        loss = sum(losses.values())
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
        strategy.step_post_backward(
            params=gaussian.gauss_params,
            optimizers=optimizers,
            state=state,
            step=step,
            info=info,
            packed=False,
        )
        average_loss = loss_average(loss.detach().item())
        average_psnr = psnr_average(psnr(rgb, gt).item())
        if step % 100 == 0:
            details = ", ".join(
                f"{key}={value.detach().item():.6f}" for key, value in losses.items()
            )
            tqdm.write(
                f"step {step}, psnr {average_psnr:.4f}, {details}, ave_total_loss={average_loss:.6f}"
            )
    gaussian.save_ply(cfg.ply_save_path)


if __name__ == "__main__":
    main(tyro.cli(Config))
