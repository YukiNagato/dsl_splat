"""Small shared training utilities; method-specific loops stay in the demos."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from tqdm import tqdm

from dsl_splat.metrics import fused_ssim
from dsl_splat.strategy import DefaultStrategy, MCMCStrategy

if __package__:
    from .colmap_dataset import Dataset, ImageFrame
    from .model import GaussianContainer
    from .utils import IndexRandomChooser, MovingAverager, psnr
else:
    from colmap_dataset import Dataset, ImageFrame
    from model import GaussianContainer
    from utils import IndexRandomChooser, MovingAverager, psnr


@dataclass
class TrainConfig:
    """Shared defaults, inherited as flat CLI flags by each demo's Config."""

    # Data/model.
    data_root: Path = Path("data")
    im_stride: int = 8
    ply_save_path: Path = Path("splat.ply")
    sh_degree: int = 3
    sh_degree_interval: int = 2000
    random_gs_num: int = 50000

    # Densification and opacity reset.
    max_gs_num: int | None = None
    # None estimates the scene extent from camera centers.
    scene_scale: float | None = None
    prune_opa: float = 0.05
    prune_scale3d: float = 0.1
    grow_grad2d: float = 0.0005
    grow_scale3d: float = 0.01
    refine_start_iter: int = 500
    refine_stop_iter: int = 15000
    reset_every: int = 5000
    refine_every: int = 100
    # "auto" uses len(dataset) + 100; None disables the post-reset pause.
    pause_refine_after_reset: int | Literal["auto"] | None = None
    absgrad: bool = True

    # Optimization.
    seed: int = 42
    train_iters: int = 30000
    means_lr: float = 1e-4
    scales_lr: float = 0.01
    quats_lr: float = 0.001
    opacities_lr: float = 0.05
    features_dc_lr: float = 0.0025
    features_rest_lr: float = 0.0025 / 20
    ssim_loss_factor: float = 0.2


def default_strategy_options(cfg: TrainConfig, dataset: Dataset) -> dict[str, Any]:
    """Options shared by DefaultStrategy and RaDe's filter-aware subclass."""
    return dict(
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


def load_rasterizer(name: Literal["3dgs", "aaa", "geer"]) -> Callable:
    """Select a demo's gsplat-style adapter once, outside the training loop."""
    if name == "3dgs":
        from dsl_splat.gs3d.gsplat import rasterization
    elif name == "aaa":
        from dsl_splat.aaa.gsplat import rasterization
    elif name == "geer":
        from dsl_splat.geer.gsplat import rasterization
    else:
        raise ValueError(f"Invalid rasterizer type: {name}")
    return rasterization


def photometric_loss(rgb: Tensor, target: Tensor, ssim_weight: float) -> Tensor:
    """The demos' shared L1/SSIM objective, on HWC RGB tensors."""
    l1 = (target - rgb).abs().mean()
    return ssim_weight * (1 - fused_ssim(rgb, target)) + (1 - ssim_weight) * l1


def compile_geometry(function: Callable, enabled: bool) -> Callable:
    """Compile dense tensor geometry; dynamic selection and Warp stay eager."""
    return (
        torch.compile(function, fullgraph=True, dynamic=True) if enabled else function
    )


class Trainer:
    """Own shared state and optimizer/strategy updates, without owning the loop.

    Callers seed before constructing the dataset. Each demo keeps its render,
    loss, debug output and method-specific scheduling explicit. ``pre_backward``
    belongs after the main render, and ``backward_and_step`` after all losses
    have been built; optimizer updates always precede strategy refinement.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        dataset: Dataset,
        strategy: DefaultStrategy | MCMCStrategy | None = None,
    ) -> None:
        self.cfg, self.dataset = cfg, dataset
        self.device = torch.device("cuda")
        self.gaussian = GaussianContainer(
            xyz=dataset.points_xyz,
            rgb=dataset.points_rgb,
            num_gs=cfg.random_gs_num,
            sh_degree=cfg.sh_degree,
        ).to(self.device)
        rates = dict(
            means=cfg.means_lr,
            scales=cfg.scales_lr,
            quats=cfg.quats_lr,
            opacities=cfg.opacities_lr,
            features_dc=cfg.features_dc_lr,
            features_rest=cfg.features_rest_lr,
        )
        self.optimizers = {
            name: torch.optim.Adam([self.gaussian.gauss_params[name]], lr=rate)
            for name, rate in rates.items()
        }
        self.strategy = (
            strategy
            if strategy is not None
            else DefaultStrategy(**default_strategy_options(cfg, dataset))
        )
        self.strategy.check_sanity(self.gaussian.gauss_params, self.optimizers)
        scene_scale = (
            dataset.scene_scale if cfg.scene_scale is None else cfg.scene_scale
        )
        self.state = (
            self.strategy.initialize_state()
            if isinstance(self.strategy, MCMCStrategy)
            else self.strategy.initialize_state(scene_scale=scene_scale)
        )
        self.chooser = IndexRandomChooser(len(dataset))
        self.loss_average, self.psnr_average = MovingAverager(10), MovingAverager(10)

    def sample_frame(self) -> tuple[int, ImageFrame]:
        index = self.chooser()
        return index, self.dataset[index]

    def sh_degree(self, step: int) -> int:
        return min(step // self.cfg.sh_degree_interval, self.cfg.sh_degree)

    def zero_grad(self) -> None:
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)

    def pre_backward(self, step: int, info: dict[str, Any]) -> None:
        self.strategy.step_pre_backward(
            self.gaussian.gauss_params, self.optimizers, self.state, step, info
        )

    def backward_and_step(self, step: int, loss: Tensor, info: dict[str, Any]) -> None:
        loss.backward()
        for optimizer in self.optimizers.values():
            optimizer.step()
        args = (self.gaussian.gauss_params, self.optimizers, self.state, step, info)
        if isinstance(self.strategy, MCMCStrategy):
            self.strategy.step_post_backward(
                *args, lr=self.optimizers["means"].param_groups[0]["lr"]
            )
        else:
            self.strategy.step_post_backward(*args, packed=False)

    @torch.no_grad()
    def log(
        self,
        step: int,
        loss: Tensor,
        rgb: Tensor,
        target: Tensor,
        *,
        losses: dict[str, Tensor] | None = None,
        first: bool = False,
    ) -> None:
        average_loss = self.loss_average(loss.item())
        average_psnr = self.psnr_average(psnr(rgb, target).item())
        if first or step % 100 == 0:
            details = ", ".join(
                f"{key}={value.detach().item():.6f}"
                for key, value in (losses or {}).items()
            )
            if details:
                details += ", "
            tqdm.write(
                f"step {step}, psnr {average_psnr:.4f}, "
                f"{details}ave_total_loss={average_loss:.6f}"
            )
