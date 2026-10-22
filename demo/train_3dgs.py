import torch
from dsl_splat.strategy.mcmc import MCMCStrategy
from dataclasses import dataclass
from typing import Any, Literal, Callable
import tyro
from tqdm import tqdm

if __package__:
    from .colmap_dataset import Dataset
    from .model import GaussianContainer
    from .training import Trainer, TrainConfig, load_rasterizer, photometric_loss
    from .utils import set_random_seed
else:
    from colmap_dataset import Dataset
    from model import GaussianContainer
    from training import Trainer, TrainConfig, load_rasterizer, photometric_loss
    from utils import set_random_seed


@dataclass
class Config(TrainConfig):
    rasterizer_type: Literal["3dgs", "aaa", "geer"] = "3dgs"
    strategy_type: Literal["default", "mcmc"] = "default"

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
    set_random_seed(cfg.seed)
    dataset = Dataset(data_root=cfg.data_root, im_stride=cfg.im_stride)
    rasterization = load_rasterizer(cfg.rasterizer_type)
    strategy = None
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
    trainer = Trainer(cfg, dataset, strategy)
    gaussian, device = trainer.gaussian, trainer.device

    for step in tqdm(range(cfg.train_iters), desc="train"):
        trainer.zero_grad()
        _, frame = trainer.sample_frame()

        render, alpha, info = render_with_rasterizer(
            rasterization,
            gaussian,
            frame.K.to(device),
            frame.w2c.to(device),
            frame.image.shape[0],
            frame.image.shape[1],
            trainer.sh_degree(step),
            absgrad=cfg.absgrad and cfg.strategy_type == "default",
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
        trainer.backward_and_step(step, loss, info)
        trainer.log(step, loss, rgb, gt)

    gaussian.save_ply(cfg.ply_save_path)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
