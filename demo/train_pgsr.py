"""PGSR-style plane supervision with RaDe-GS's Warp patch NCC.

Images use HWC; w2c is an ordinary OpenCV matrix, so row-vector points transform
as p_camera = p_world @ w2c[:3, :3].T + w2c[:3, 3]. Rasterizer K coordinates
have half-integer pixel centers. Geometry/NCC use integer array coordinates,
with their principal points shifted by -0.5 to describe the same rays.
This is a minimal training example, not a full reproduction of PGSR's trainer.
"""

from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any, Callable, Literal

from torch import Tensor
from tqdm import tqdm
import tyro

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
from dsl_splat.strategy import ImprovedGSStrategy

if __package__:
    from .colmap_dataset import ImageFrame
    from .multiview import (
        MultiViewDataset as PGSRDataset,
        render_debug_image,
        save_debug_image,
    )
    from .model import GaussianContainer
    from .improvedgs import (
        ImprovedGSOptions,
        ImprovedGSScorer,
        improved_strategy_options,
    )
    from .training import (
        Trainer,
        TrainConfig,
        compile_geometry,
        load_rasterizer,
        photometric_loss,
    )
    from .utils import set_random_seed
else:
    from colmap_dataset import ImageFrame
    from multiview import (
        MultiViewDataset as PGSRDataset,
        render_debug_image,
        save_debug_image,
    )
    from model import GaussianContainer
    from improvedgs import (
        ImprovedGSOptions,
        ImprovedGSScorer,
        improved_strategy_options,
    )
    from training import (
        Trainer,
        TrainConfig,
        compile_geometry,
        load_rasterizer,
        photometric_loss,
    )
    from utils import set_random_seed


@dataclass
class Config(TrainConfig):
    # Data/model parameters.
    debug_save_path: Path | None = None
    rasterizer_type: Literal["3dgs", "aaa", "geer"] = "3dgs"
    grow_grad2d: float = 0.0004
    strategy_type: Literal["default", "improvedgs"] = "default"
    improvedgs: ImprovedGSOptions = field(default_factory=ImprovedGSOptions)

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
    colors, degree = gaussian.appearance(sh_degree_to_use)
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
    if cfg.strategy_type == "improvedgs" and cfg.rasterizer_type == "geer":
        raise ValueError(
            "Improved-GS EAS requires 3dgs or aaa; GEER has no contribution statistics"
        )
    set_random_seed(cfg.seed)
    dataset = PGSRDataset(
        data_root=cfg.data_root,
        im_stride=cfg.im_stride,
        multi_view_num=cfg.pgsr_multi_view_num,
        multi_view_max_angle=cfg.pgsr_multi_view_max_angle,
        multi_view_min_dis=cfg.pgsr_multi_view_min_dis,
        multi_view_max_dis=cfg.pgsr_multi_view_max_dis,
    )
    rasterization = load_rasterizer(cfg.rasterizer_type)
    strategy = (
        ImprovedGSStrategy(**improved_strategy_options(cfg.improvedgs, cfg, dataset))
        if cfg.strategy_type == "improvedgs"
        else None
    )
    trainer = Trainer(cfg, dataset, strategy)
    scorer = (
        ImprovedGSScorer.from_options(dataset, cfg.improvedgs)
        if strategy is not None
        else None
    )
    gaussian, device = trainer.gaussian, trainer.device
    feature_fn = compile_geometry(plane_features, cfg.pgsr_compile)
    maps_fn = compile_geometry(plane_maps, cfg.pgsr_compile)
    multi_view = MultiViewLoss(
        cfg.pgsr_multi_view_pixel_noise_th,
        cfg.pgsr_multi_view_sample_num,
        half_step=cfg.pgsr_multi_view_half_step,
        geometry_fn=compile_geometry(multiview_geometry, cfg.pgsr_compile),
    )
    first_step = 1 if scorer is not None else 0
    for step in tqdm(range(first_step, cfg.train_iters + first_step), desc="train"):
        debug_info: dict[str, Tensor] | None = (
            {} if cfg.debug_save_path is not None and step % 200 == 0 else None
        )
        trainer.begin_step()
        data_idx, frame = trainer.sample_frame()
        degree = trainer.sh_degree(step)

        def render_frame(camera: ImageFrame):
            return render_with_rasterizer(
                rasterization,
                gaussian,
                camera.K.to(device),
                camera.w2c.to(device),
                camera.height,
                camera.width,
                degree,
                absgrad=cfg.absgrad and (scorer is None or step < cfg.refine_stop_iter),
                plane_features_fn=feature_fn,
                plane_maps_fn=maps_fn,
            )

        render, alpha, info = render_frame(frame)
        trainer.pre_backward(step, info)
        rgb, gt = render[0, ..., :3], frame.image.to(device)
        losses = {"main_loss": photometric_loss(rgb, gt, cfg.ssim_loss_factor)}
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
        trainer.backward_and_step(
            step,
            loss,
            info,
            score_fn=(
                None
                if scorer is None
                else lambda: scorer(step, gaussian, rasterization, degree)
            ),
            final=step == cfg.train_iters,
        )
        trainer.log(step, loss, rgb, gt, losses=losses)
    gaussian.save_ply(cfg.ply_save_path)


if __name__ == "__main__":
    main(tyro.cli(Config))
