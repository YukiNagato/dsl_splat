"""PGSR-style plane supervision with RaDe-GS's Warp patch NCC.

Images use HWC; w2c is an ordinary OpenCV matrix, so row-vector points transform
as p_camera = p_world @ w2c[:3, :3].T + w2c[:3, 3]. Rasterizer K coordinates
have half-integer pixel centers. Geometry/NCC use integer array coordinates,
with their principal points shifted by -0.5 to describe the same rays.
This is a minimal training example, not a full reproduction of PGSR's trainer.
"""

from dataclasses import dataclass
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
from dsl_splat.pgsr import (
    MultiViewLoss,
    plane_features,
    plane_maps,
    multiview_geometry,
    flatten_loss,
    get_img_grad_weight,
    single_view_loss,
)
from dsl_splat.pgsr.geometry import MIN_DEPTH
from dsl_splat.metrics import fused_ssim

if __package__:
    from .colmap_dataset import Dataset, ImageFrame
    from .model import GaussianContainer
    from .utils import IndexRandomChooser, MovingAverager, psnr
else:
    from colmap_dataset import Dataset, ImageFrame
    from model import GaussianContainer
    from utils import IndexRandomChooser, MovingAverager, psnr


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
    ssim_loss_factor: float = 0.2

    # PGSR parameters; the multi-view factor weights NCC and geometry separately.
    pgsr_compile: bool = True
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
    plane_features_fn: Callable = plane_features,
    plane_maps_fn: Callable = plane_maps,
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
    # Dynamo specializes Parameter shapes even with dynamic=True. Tensor views
    # keep densification dynamic without copying storage or detaching gradients.
    extra = plane_features_fn(
        viewmat, means.view_as(means), scales, quats.view_as(quats)
    )
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
    info.update(plane_maps_fn(planes[..., :3], planes[..., 3], alpha[0, ..., 0], K))
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

    # Compile only dense tensor blocks; Warp dispatch and dynamic selection stay eager.
    def compile_geometry(function: Callable) -> Callable:
        if not cfg.pgsr_compile:
            return function
        return torch.compile(function, fullgraph=True, dynamic=True)

    feature_fn = compile_geometry(plane_features)
    maps_fn = compile_geometry(plane_maps)
    multi_view = MultiViewLoss(
        cfg.pgsr_multi_view_pixel_noise_th,
        cfg.pgsr_multi_view_sample_num,
        half_step=cfg.pgsr_multi_view_half_step,
        geometry_fn=compile_geometry(multiview_geometry),
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
                plane_features_fn=feature_fn,
                plane_maps_fn=maps_fn,
            )

        render, alpha, info = render_frame(frame)
        strategy.step_pre_backward(gaussian.gauss_params, optimizers, state, step, info)
        rgb, gt = render[0, ..., :3], frame.image.to(device)
        l1 = torch.abs(gt - rgb).mean()
        ssim_loss = 1 - fused_ssim(rgb, gt)
        main_loss = cfg.ssim_loss_factor * ssim_loss + (1 - cfg.ssim_loss_factor) * l1
        losses = {"main_loss": main_loss}
        visible = (info["radii"][0] > 0).all(dim=-1)
        if visible.any():
            losses["flatten_loss"] = cfg.pgsr_flatten_loss_factor * flatten_loss(
                gaussian.gauss_params["scales"][visible].exp()
            )
        if step > cfg.pgsr_view_supervise_start_iter:
            losses["single_view_loss"] = (
                cfg.pgsr_single_view_loss_factor * single_view_loss(gt, info)
            )
            neighbors = dataset.frame_nearest_ids[data_idx]
            if neighbors:
                neighbor = dataset[random.choice(neighbors)]
                _, _, near_info = render_frame(neighbor)
                ncc, geometry = multi_view(
                    info,
                    near_info,
                    viewmat=frame.w2c,
                    K=frame.K,
                    near_viewmat=neighbor.w2c,
                    near_K=neighbor.K,
                    image=frame.gray_image,
                    near_image=neighbor.gray_image,
                    debug_info=debug_info,
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
