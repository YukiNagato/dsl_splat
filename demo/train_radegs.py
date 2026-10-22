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

import torch
from torch import Tensor
from tqdm import tqdm
import tyro

from dsl_splat import rade
from dsl_splat.pgsr import MultiViewLoss, multiview_geometry

if __package__:
    from .colmap_dataset import ImageFrame
    from .model import GaussianContainer
    from .multiview import MultiViewDataset, save_debug_image
    from .rade import RaDeStrategy, compute_3d_filter, geometry_maps
    from .training import (
        Trainer,
        TrainConfig,
        compile_geometry,
        default_strategy_options,
        photometric_loss,
    )
    from .utils import set_random_seed
else:
    from colmap_dataset import ImageFrame
    from model import GaussianContainer
    from multiview import MultiViewDataset, save_debug_image
    from rade import RaDeStrategy, compute_3d_filter, geometry_maps
    from training import (
        Trainer,
        TrainConfig,
        compile_geometry,
        default_strategy_options,
        photometric_loss,
    )
    from utils import set_random_seed


@dataclass
class Config(TrainConfig):
    debug_save_path: Path | None = None
    sh_degree_interval: int = 1000

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

    # DefaultStrategy, not the original GOF/quantile densification policy.
    prune_scale3d: float = math.inf
    grow_grad2d: float = 0.0004
    reset_every: int = 3000
    reset_opacity: float = 0.01  # Cap filtered opacity independently of prune_opa.

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
    colors, degree = gaussian.appearance(sh_degree)
    attributes = (
        dict(features=colors)
        if degree is None
        else dict(sh_coeffs=colors, sh_degree=degree)
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
    dataset = MultiViewDataset(
        data_root=cfg.data_root,
        im_stride=cfg.im_stride,
        multi_view_num=cfg.multi_view_num,
        multi_view_max_angle=cfg.multi_view_max_angle,
        multi_view_min_dis=cfg.multi_view_min_dis,
        multi_view_max_dis=cfg.multi_view_max_dis,
    )
    strategy = RaDeStrategy(
        dataset.frames,
        filter_3d=cfg.filter_3d,
        near_plane=cfg.near_plane,
        reset_opacity=cfg.reset_opacity,
        **default_strategy_options(cfg, dataset),
    )
    trainer = Trainer(cfg, dataset, strategy)
    gaussian, device, state = trainer.gaussian, trainer.device, trainer.state
    strategy.refresh_filter(gaussian.gauss_params, state)

    maps_fn = compile_geometry(geometry_maps, cfg.compile_geometry)
    multi_view = MultiViewLoss(
        cfg.multi_view_pixel_noise_th,
        cfg.multi_view_sample_num,
        half_step=cfg.multi_view_half_step,
        geometry_fn=compile_geometry(multiview_geometry, cfg.compile_geometry),
    )
    for step in tqdm(range(1, cfg.train_iters + 1), desc="train"):
        trainer.zero_grad()
        progress = (step - 1) / max(cfg.train_iters - 1, 1)
        trainer.optimizers["means"].param_groups[0]["lr"] = math.exp(
            math.log(cfg.means_lr) * (1 - progress)
            + math.log(cfg.means_lr_final) * progress
        )
        index, frame = trainer.sample_frame()
        degree = trainer.sh_degree(step)

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
        trainer.pre_backward(step, info)
        rgb, gt = result.features[0], frame.image.to(device)
        losses = dict(rgb=photometric_loss(rgb, gt, cfg.ssim_loss_factor))
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
        trainer.backward_and_step(step, loss, info)
        if step >= cfg.refine_stop_iter and step % cfg.filter_update_every == 0:
            strategy.refresh_filter(gaussian.gauss_params, state)
        trainer.log(step, loss, rgb, gt, losses=losses, first=step == 1)
    # Include the current filter widths so a RaDe loader can reproduce the render.
    strategy.refresh_filter(gaussian.gauss_params, state)
    gaussian.save_ply(
        cfg.ply_save_path, extra_attributes={"filter_3D": state["filter_3d"]}
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
