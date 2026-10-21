"""Minimal RaDe-GS training.

Uses the native RaDe renderer's depth/normals, explicit camera-dependent 3D
filtering, cosine normal consistency and the shared Warp patch-NCC supervision.
The demo uses DefaultStrategy and rendered neighbor depth sampling; it does
not reproduce upstream's GOF strategy, appearance network or sample_depth API.
"""

from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any, Callable, Literal, Sequence

import torch
from torch import Tensor
from tqdm import tqdm
import tyro

from dsl_splat import rade
from dsl_splat.metrics import fused_ssim
from dsl_splat.pgsr import MultiViewLoss, multiview_geometry
from dsl_splat.pgsr.geometry import (
    depth_to_normal,
    integer_center_intrinsics,
    pixel_rays,
    valid_depth,
)
from dsl_splat.strategy.default import DefaultStrategy
from dsl_splat.strategy.ops import reset_opa

if __package__:
    from .colmap_dataset import ImageFrame
    from .model import GaussianContainer
    from .multiview import MultiViewDataset, save_debug_image
    from .utils import IndexRandomChooser, MovingAverager, psnr, set_random_seed
else:
    from colmap_dataset import ImageFrame
    from model import GaussianContainer
    from multiview import MultiViewDataset, save_debug_image
    from utils import IndexRandomChooser, MovingAverager, psnr, set_random_seed


@dataclass
class Config:
    data_root: Path = Path("data")
    im_stride: int = 8
    ply_save_path: Path = Path("splat.ply")
    debug_save_path: Path | None = None
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    random_gs_num: int = 50000
    seed: int = 42
    train_iters: int = 30000

    # Frontend 3D filter and rasterizer 2D dilation are separate controls.
    filter_3d: bool = True
    filter_update_every: int = 100
    kernel_size: float = 0.0
    near_plane: float = 0.2
    depth_ratio: float = 0.0  # Normal loss: 0=expected depth, 1=median depth.

    # Example optimization rates, without upstream's appearance network.
    means_lr: float = 1.6e-4
    means_lr_final: float = 1.6e-6
    scales_lr: float = 0.005
    quats_lr: float = 0.001
    opacities_lr: float = 0.05
    features_dc_lr: float = 0.0025
    features_rest_lr: float = 0.0025 / 20
    ssim_loss_factor: float = 0.2

    # DefaultStrategy, not the original GOF/quantile densification policy.
    max_gs_num: int | None = None
    scene_scale: float | None = None
    prune_opa: float = 0.05
    prune_scale3d: float = math.inf
    grow_grad2d: float = 0.0004
    grow_scale3d: float = 0.01
    refine_start_iter: int = 500
    refine_stop_iter: int = 15000
    refine_every: int = 100
    reset_every: int = 3000
    reset_opacity: float = 0.01  # Cap filtered opacity independently of prune_opa.
    pause_refine_after_reset: int | Literal["auto"] | None = None
    absgrad: bool = True

    compile_geometry: bool = True
    regularization_from_iter: int = 7000
    normal_loss_factor: float = 0.05
    multi_view_ncc_loss_factor: float = 0.3
    multi_view_geo_loss_factor: float = 0.02
    multi_view_pixel_noise_th: float = 1.0
    multi_view_sample_num: int = 102400
    multi_view_half_step: bool = True
    multi_view_max_angle: float = 30
    multi_view_min_dis: float = 0.01
    multi_view_max_dis: float = 1.5
    multi_view_num: int = 8

    def __post_init__(self) -> None:
        if not 0 <= self.depth_ratio <= 1:
            raise ValueError("depth_ratio must be in [0,1]")
        if not 0 < self.reset_opacity < 1:
            raise ValueError("reset_opacity must be in (0,1)")
        if not math.isfinite(self.near_plane) or self.near_plane <= 0:
            raise ValueError("near_plane must be positive and finite")
        if (
            min(
                self.sh_degree_interval,
                self.filter_update_every,
                self.refine_every,
                self.reset_every,
                self.train_iters,
            )
            <= 0
        ):
            raise ValueError("iteration counts and intervals must be positive")
        if min(self.means_lr, self.means_lr_final) <= 0:
            raise ValueError("position learning rates must be positive")


@torch.no_grad()
def compute_3d_filter(
    means: Tensor, frames: Sequence[ImageFrame], near_plane: float = 0.2
) -> Tensor:
    """RaDe's sqrt(0.2)*min_visible_z/max_fx filter width, per Gaussian.

    Bounds extend the image by 7.5% on each side (the centered-camera 0.575
    rule), retaining arbitrary principal points. Unseen points use the largest
    visible depth. If all points are unseen, use the largest positive camera
    depth, or near_plane if all are behind the cameras. Empty inputs are valid.
    """
    if not frames:
        raise ValueError("3D filtering requires at least one training camera")
    if not len(means):
        return means.new_empty((0, 1))
    distance = means.new_full((len(means),), math.inf)
    fallback = means.new_tensor(near_plane)
    max_focal = max(frame.fx for frame in frames)
    for frame in frames:
        view = frame.w2c.to(means)
        points = means @ view[:3, :3].T + view[:3, 3]
        z = points[:, 2]
        valid = torch.isfinite(points).all(-1) & (z > near_plane)
        safe = torch.where(valid[:, None], points, 0.0)
        uv = safe[:, :2] / torch.where(valid, z, 1.0)[:, None]
        u, v = uv[:, 0] * frame.fx + frame.cx, uv[:, 1] * frame.fy + frame.cy
        inside = (
            valid
            & (u >= -0.075 * frame.width)
            & (u <= 1.075 * frame.width)
            & (v >= -0.075 * frame.height)
            & (v <= 1.075 * frame.height)
        )
        distance = torch.where(inside, torch.minimum(distance, z), distance)
        fallback = torch.maximum(fallback, torch.where(valid, z, 0.0).amax())
    visible = torch.isfinite(distance)
    farthest = torch.where(visible, distance, 0.0).amax()
    fallback = torch.where(visible.any(), farthest, fallback)
    distance = torch.where(visible, distance, fallback)
    return (distance * (math.sqrt(0.2) / max_focal))[:, None]


class RaDeStrategy(DefaultStrategy):
    """Keep filter rows aligned through growth/pruning and reset filtered opacity."""

    def __init__(
        self,
        frames: Sequence[ImageFrame],
        *,
        filter_3d: bool = True,
        near_plane: float = 0.2,
        reset_opacity: float = 0.01,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.frames, self.filter_enabled = frames, filter_3d
        self.near_plane, self.reset_opacity = near_plane, reset_opacity

    @torch.no_grad()
    def refresh_filter(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        state: dict[str, Any],
    ) -> None:
        state["filter_3d"] = (
            compute_3d_filter(params["means"], self.frames, self.near_plane)
            if self.filter_enabled
            else params["means"].new_zeros((len(params["means"]), 1))
        )

    def _prune_gs(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Any],
        step: int,
    ) -> int:
        removed = super()._prune_gs(params, optimizers, state, step)
        # Called after growth; recompute even when a split leaves N unchanged.
        self.refresh_filter(params, state)
        return removed

    def _reset_opacities(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Any],
        info: dict[str, Any],
    ) -> None:
        reset_opa(
            params,
            optimizers,
            state,
            self.reset_opacity,
            opacity_filter_variance=(
                state["filter_3d"].square() if self.filter_enabled else None
            ),
        )


def geometry_maps(
    expected: Tensor,
    median: Tensor,
    normals: Tensor,
    alpha: Tensor,
    K: Tensor,
    depth_ratio: float = 0.0,
) -> dict[str, Tensor]:
    """Direct RaDe depth/normal supervision; no PGSR plane or alpha weighting."""
    expected, median, alpha = expected[0, ..., 0], median[0, ..., 0], alpha[0, ..., 0]
    normal = normals[0]
    finite = torch.isfinite(normal).all(-1) & torch.isfinite(alpha) & (alpha > 1e-6)
    safe_normal = torch.where(finite[..., None], normal, 0.0)
    expected_valid = valid_depth(expected) & finite
    _, rays = pixel_rays(integer_center_intrinsics(K), *expected.shape)
    depth_normal = torch.zeros_like(safe_normal)
    normal_valid = torch.zeros_like(expected_valid)
    loss = safe_normal.reshape(-1)[:0].sum()
    for weight, depth in ((1 - depth_ratio, expected), (depth_ratio, median)):
        if weight > 0:
            valid = valid_depth(depth) & finite
            dn, mask = depth_to_normal(depth, rays, valid)
            error = 1 - (safe_normal * dn).sum(-1)
            loss = loss + weight * torch.where(mask, error, 0.0).mean()
            depth_normal = depth_normal + weight * dn
            normal_valid = normal_valid | mask
    return dict(
        depths=torch.where(expected_valid, expected, 0.0)[None, ..., None],
        normals=safe_normal[None],
        depth_valid=expected_valid[None, ..., None],
        depth_normals=depth_normal[None],
        depth_normal_valid=normal_valid[None, ..., None],
        normal_loss=loss,
    )


def render_frame(
    gaussian: GaussianContainer,
    frame: ImageFrame,
    filter_3d: Tensor | None,
    sh_degree: int,
    *,
    absgrad: bool = True,
    kernel_size: float = 0.0,
    near_plane: float = 0.2,
) -> rade.RasterizationResult:
    params = gaussian.gauss_params
    scales = params["scales"].exp()
    opacity = params["opacities"].sigmoid().squeeze(-1)
    if filter_3d is not None:
        scales, opacity = rade.apply_3d_filter(scales, opacity, filter_3d)
    if gaussian.sh_degree == 0:
        attributes = dict(features=params["features_dc"].sigmoid())
    else:
        attributes = dict(
            sh_coeffs=torch.cat(
                (params["features_dc"][:, None], params["features_rest"]), dim=1
            ),
            sh_degree=sh_degree,
        )
    return rade.rasterize(
        params["means"],
        params["quats"],
        scales,
        opacity,
        frame.w2c.to(scales)[None],
        frame.K.to(scales)[None],
        frame.width,
        frame.height,
        kernel_size=kernel_size,
        near_plane=near_plane,
        packed=False,
        absgrad=absgrad,
        **attributes,
    )


def main(cfg: Config) -> None:
    set_random_seed(cfg.seed)
    device = torch.device("cuda")
    dataset = MultiViewDataset(
        data_root=cfg.data_root,
        im_stride=cfg.im_stride,
        multi_view_num=cfg.multi_view_num,
        multi_view_max_angle=cfg.multi_view_max_angle,
        multi_view_min_dis=cfg.multi_view_min_dis,
        multi_view_max_dis=cfg.multi_view_max_dis,
    )
    gaussian = GaussianContainer(
        dataset.points_xyz, dataset.points_rgb, cfg.random_gs_num, cfg.sh_degree
    ).to(device)
    rates = dict(
        means=cfg.means_lr,
        scales=cfg.scales_lr,
        quats=cfg.quats_lr,
        opacities=cfg.opacities_lr,
        features_dc=cfg.features_dc_lr,
        features_rest=cfg.features_rest_lr,
    )
    optimizers = {
        key: torch.optim.Adam([gaussian.gauss_params[key]], lr=rate)
        for key, rate in rates.items()
    }
    strategy = RaDeStrategy(
        dataset.frames,
        filter_3d=cfg.filter_3d,
        near_plane=cfg.near_plane,
        reset_opacity=cfg.reset_opacity,
        max_gs_num=cfg.max_gs_num,
        prune_opa=cfg.prune_opa,
        prune_scale3d=cfg.prune_scale3d,
        grow_grad2d=cfg.grow_grad2d,
        grow_scale3d=cfg.grow_scale3d,
        refine_start_iter=cfg.refine_start_iter,
        refine_stop_iter=cfg.refine_stop_iter,
        reset_every=cfg.reset_every,
        refine_every=cfg.refine_every,
        pause_refine_after_reset=(
            len(dataset) + 100
            if cfg.pause_refine_after_reset == "auto"
            else cfg.pause_refine_after_reset
        ),
        absgrad=cfg.absgrad,
        revised_opacity=False,
        verbose=True,
    )
    strategy.check_sanity(gaussian.gauss_params, optimizers)
    scene_scale = dataset.scene_scale if cfg.scene_scale is None else cfg.scene_scale
    state = strategy.initialize_state(scene_scale=scene_scale)
    strategy.refresh_filter(gaussian.gauss_params, state)

    def compile_function(fn: Callable) -> Callable:
        return (
            torch.compile(fn, fullgraph=True, dynamic=True)
            if cfg.compile_geometry
            else fn
        )

    maps_fn = compile_function(geometry_maps)
    multi_view = MultiViewLoss(
        cfg.multi_view_pixel_noise_th,
        cfg.multi_view_sample_num,
        half_step=cfg.multi_view_half_step,
        geometry_fn=compile_function(multiview_geometry),
    )
    chooser = IndexRandomChooser(len(dataset))
    loss_average, psnr_average = MovingAverager(10), MovingAverager(10)
    for step in tqdm(range(1, cfg.train_iters + 1), desc="train"):
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        progress = (step - 1) / max(cfg.train_iters - 1, 1)
        optimizers["means"].param_groups[0]["lr"] = math.exp(
            math.log(cfg.means_lr) * (1 - progress)
            + math.log(cfg.means_lr_final) * progress
        )
        index = chooser()
        frame = dataset[index]
        degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)

        def render(camera: ImageFrame):
            return render_frame(
                gaussian,
                camera,
                state["filter_3d"] if cfg.filter_3d else None,
                degree,
                absgrad=cfg.absgrad,
                kernel_size=cfg.kernel_size,
                near_plane=cfg.near_plane,
            )

        result = render(frame)
        info = result.meta
        strategy.step_pre_backward(gaussian.gauss_params, optimizers, state, step, info)
        rgb, gt = result.features[0], frame.image.to(device)
        losses = dict(
            rgb=(1 - cfg.ssim_loss_factor) * (rgb - gt).abs().mean()
            + cfg.ssim_loss_factor * (1 - fused_ssim(rgb, gt))
        )
        regularize = step >= cfg.regularization_from_iter
        debug = cfg.debug_save_path is not None and (step == 1 or step % 200 == 0)
        debug_info: dict[str, Tensor] | None = {} if debug else None
        if regularize or debug:
            info.update(
                maps_fn(
                    result.depth,
                    result.median_depth,
                    result.normals,
                    result.alpha,
                    frame.K.to(device),
                    cfg.depth_ratio,
                )
            )
        if regularize:
            losses["normal"] = cfg.normal_loss_factor * info["normal_loss"]
            neighbors = dataset.frame_nearest_ids[index]
            if neighbors and (
                cfg.multi_view_ncc_loss_factor > 0 or cfg.multi_view_geo_loss_factor > 0
            ):
                neighbor = dataset[random.choice(neighbors)]
                near = render(neighbor)
                near_info = dict(
                    depths=near.depth,
                    normals=near.normals,
                    depth_valid=(
                        torch.isfinite(near.depth)
                        & (near.depth > 0)
                        & (near.alpha > 1e-6)
                    ),
                )
                ncc, geo = multi_view(
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
                losses["ncc"] = cfg.multi_view_ncc_loss_factor * ncc
                losses["geometry"] = cfg.multi_view_geo_loss_factor * geo
        if debug_info is not None:
            save_debug_image(
                cfg.debug_save_path,
                step,
                frame,
                rgb,
                info,
                geometry_weights=debug_info.get("geometry_weights"),
                scalar_key="depths",
                depth_key="median_depth",
            )
        loss = sum(losses.values())
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
        strategy.step_post_backward(
            gaussian.gauss_params, optimizers, state, step, info, packed=False
        )
        if step >= cfg.refine_stop_iter and step % cfg.filter_update_every == 0:
            strategy.refresh_filter(gaussian.gauss_params, state)
        average_loss = loss_average(loss.detach().item())
        average_psnr = psnr_average(psnr(rgb, gt).item())
        if step == 1 or step % 100 == 0:
            details = ", ".join(
                f"{key}={value.detach().item():.6f}" for key, value in losses.items()
            )
            tqdm.write(
                f"step {step}, psnr {average_psnr:.4f}, {details}, ave_total_loss={average_loss:.6f}"
            )
    # Include the current filter widths so a RaDe loader can reproduce the render.
    strategy.refresh_filter(gaussian.gauss_params, state)
    gaussian.save_ply(
        cfg.ply_save_path, extra_attributes={"filter_3D": state["filter_3d"]}
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
