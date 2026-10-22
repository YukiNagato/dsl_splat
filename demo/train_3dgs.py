"""3DGS/AAA/GEER training with Default, MCMC or Improved-GS strategies."""

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Callable

import torch
import tyro
from tqdm import tqdm

from dsl_splat.strategy import ImprovedGSStrategy, MCMCStrategy

if __package__:
    from .colmap_dataset import Dataset
    from .model import GaussianContainer
    from .improvedgs import (
        ImprovedGSOptions,
        ImprovedGSScorer,
        improved_strategy_options,
    )
    from .training import Trainer, TrainConfig, load_rasterizer, photometric_loss
    from .utils import set_random_seed
else:
    from colmap_dataset import Dataset
    from model import GaussianContainer
    from improvedgs import (
        ImprovedGSOptions,
        ImprovedGSScorer,
        improved_strategy_options,
    )
    from training import Trainer, TrainConfig, load_rasterizer, photometric_loss
    from utils import set_random_seed


@dataclass
class Config(TrainConfig):
    rasterizer_type: Literal["3dgs", "aaa", "geer"] = "3dgs"
    strategy_type: Literal["default", "mcmc", "improvedgs"] = "default"
    improvedgs: ImprovedGSOptions = field(default_factory=ImprovedGSOptions)
    # None keeps a constant position learning rate, as in the original demo.
    means_lr_final: float | None = None

    # MCMC-only settings; max_gs_num defaults to 1M for this strategy.
    mcmc_noise_lr: float = 5e5
    mcmc_min_opacity: float = 0.005
    mcmc_noise_stop_iter: int | None = None
    mcmc_opacity_reg: float = 0.01
    mcmc_scale_reg: float = 0.01


def render_with_rasterizer(
    rasterizer: Callable,
    gaussian: GaussianContainer,
    K: torch.Tensor,
    viewmat: torch.Tensor,
    H: int,
    W: int,
    sh_degree_to_use: int | None = None,
    *,
    absgrad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    means = gaussian.gauss_params["means"]
    scales = gaussian.gauss_params["scales"]
    opacities = gaussian.gauss_params["opacities"]
    quats = gaussian.gauss_params["quats"]
    colors, degree = gaussian.appearance(sh_degree_to_use)
    render, alpha, info = rasterizer(
        means=means,
        quats=quats,
        scales=torch.exp(scales),
        opacities=torch.sigmoid(opacities).squeeze(-1),
        colors=colors,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        width=W,
        height=H,
        packed=False,
        near_plane=0.001,
        far_plane=1e10,
        render_mode="RGB",
        sh_degree=degree,
        sparse_grad=False,
        absgrad=absgrad,
        rasterize_mode="classic",
        camera_model="pinhole",
    )
    return render, alpha, info


def main(cfg: Config) -> None:
    if cfg.strategy_type == "improvedgs" and cfg.rasterizer_type == "geer":
        raise ValueError(
            "Improved-GS EAS requires 3dgs or aaa; GEER has no contribution statistics"
        )
    if cfg.strategy_type == "improvedgs" and cfg.train_iters < 1:
        raise ValueError("train_iters must be positive")
    if cfg.means_lr_final is not None:
        if any(
            not math.isfinite(rate) or rate <= 0
            for rate in (cfg.means_lr, cfg.means_lr_final)
        ):
            raise ValueError(
                "decaying position learning rates must be positive and finite"
            )
    set_random_seed(cfg.seed)
    dataset = Dataset(data_root=cfg.data_root, im_stride=cfg.im_stride)
    rasterization = load_rasterizer(cfg.rasterizer_type)
    strategy: MCMCStrategy | ImprovedGSStrategy | None = None
    if cfg.strategy_type == "mcmc":
        strategy = MCMCStrategy(
            cap_max=1_000_000 if cfg.max_gs_num is None else cfg.max_gs_num,
            noise_lr=cfg.mcmc_noise_lr,
            min_opacity=cfg.mcmc_min_opacity,
            noise_stop_iter=cfg.mcmc_noise_stop_iter,
            refine_start_iter=cfg.refine_start_iter,
            refine_stop_iter=cfg.refine_stop_iter,
            refine_every=cfg.refine_every,
            verbose=True,
        )
    elif cfg.strategy_type == "improvedgs":
        strategy = ImprovedGSStrategy(
            **improved_strategy_options(cfg.improvedgs, cfg, dataset)
        )
    trainer = Trainer(cfg, dataset, strategy)
    gaussian, device = trainer.gaussian, trainer.device
    scorer = (
        ImprovedGSScorer.from_options(dataset, cfg.improvedgs)
        if cfg.strategy_type == "improvedgs"
        else None
    )

    first_step = 1 if scorer is not None else 0
    for step in tqdm(range(first_step, cfg.train_iters + first_step), desc="train"):
        trainer.begin_step()
        if cfg.means_lr_final is not None:
            progress = (step - first_step) / max(cfg.train_iters - 1, 1)
            trainer.optimizers["means"].param_groups[0]["lr"] = math.exp(
                math.log(cfg.means_lr) * (1 - progress)
                + math.log(cfg.means_lr_final) * progress
            )
        _, frame = trainer.sample_frame()
        degree = trainer.sh_degree(step)

        render, alpha, info = render_with_rasterizer(
            rasterization,
            gaussian,
            frame.K.to(device),
            frame.w2c.to(device),
            frame.image.shape[0],
            frame.image.shape[1],
            degree,
            absgrad=cfg.absgrad
            and cfg.strategy_type != "mcmc"
            and (scorer is None or step < cfg.refine_stop_iter),
        )
        trainer.pre_backward(step, info)

        rgb = render[:, ..., :3]
        rgb = rgb.squeeze(0)
        gt = frame.image.to(device)
        loss = photometric_loss(rgb, gt, cfg.ssim_loss_factor)
        if cfg.strategy_type == "mcmc":
            # MCMC's sparsity regularizers are losses, not population updates.
            loss = (
                loss
                + cfg.mcmc_opacity_reg
                * gaussian.gauss_params["opacities"].sigmoid().mean()
                + cfg.mcmc_scale_reg * gaussian.gauss_params["scales"].exp().mean()
            )
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
        trainer.log(step, loss, rgb, gt)

    gaussian.save_ply(cfg.ply_save_path)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
