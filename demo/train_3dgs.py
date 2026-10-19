import torch
from dsl_splat.strategy.default import DefaultStrategy
from dsl_splat.metrics import fused_ssim
from dataclasses import dataclass
from colmap_dataset import Dataset, ImageFrame
from model import GaussianContainer
from typing import List, Dict, Any, Tuple, Literal, Callable
import tyro
from pathlib import Path
from utils import IndexRandomChooser, MovingAverager, psnr
from tqdm import tqdm


@dataclass
class Config:
    # data params
    data_root: Path = Path("data")
    im_stride: int = 8
    ply_save_path: Path = Path("splat.ply")

    # model params
    sh_degree: int = 3
    sh_degree_interval: int = 2000
    random_gs_num: int = 50000
    rasterizer_type: Literal["3dgs", "aaa"] = "3dgs"

    # strategy params
    prune_opa: float = 0.05
    grow_grad2d: float = 0.0005
    grow_scale3d: float = 0.01
    refine_start_iter: int = 500
    refine_stop_iter: int = 15000
    reset_every: int = 5000
    refine_every: int = 100
    absgrad: bool = True

    # training params
    train_iters: int = 30000
    means_lr: float = 1e-4
    scales_lr: float = 0.01
    quats_lr: float = 0.001
    opacities_lr: float = 0.05
    features_dc_lr: float = 0.0025
    features_rest_lr: float = 0.0025 / 20
    ssim_loss_factor: float = 0.2


def render_with_rasterizer(
        rasterizer: Callable, 
        gaussian: GaussianContainer, 
        K: torch.Tensor, 
        viewmat: torch.Tensor, 
        H: int, W: int, 
        sh_degree_to_use: int = None) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    means = gaussian.gauss_params["means"]
    scales = gaussian.gauss_params["scales"]
    opacities = gaussian.gauss_params["opacities"]
    quats = gaussian.gauss_params["quats"]
    features_dc = gaussian.gauss_params["features_dc"]
    features_rest = gaussian.gauss_params["features_rest"]
    colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
    if gaussian.sh_degree == 0:
        colors = torch.sigmoid(colors).squeeze(1)
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
        sh_degree=sh_degree_to_use,
        sparse_grad=False,
        absgrad=True,
        rasterize_mode="classic",
        camera_model="pinhole",
    )
    return render, alpha, info


def main(cfg: Config):
    device = 'cuda'
    dataset = Dataset(data_root=cfg.data_root, im_stride=cfg.im_stride)

    if cfg.rasterizer_type == "3dgs":
        from dsl_splat.gs3d.gsplat import rasterization
    elif cfg.rasterizer_type == "aaa":
        from dsl_splat.aaa.gsplat import rasterization
    else:
        raise ValueError(f"Invalid rasterizer type: {cfg.rasterizer_type}")

    gaussian = GaussianContainer(
        xyz=dataset.points_xyz,
        rgb=dataset.points_rgb,
        num_gs=cfg.random_gs_num,
        sh_degree=cfg.sh_degree,
    )

    gaussian = gaussian.to(device)
    index_random_chooser = IndexRandomChooser(len(dataset))
    loss_moving_averager = MovingAverager(10)
    psnr_moving_averager = MovingAverager(10)
    optimizers_dict = {
        "means": torch.optim.Adam([gaussian.gauss_params["means"]], lr=cfg.means_lr),
        "scales": torch.optim.Adam([gaussian.gauss_params["scales"]], lr=cfg.scales_lr),
        "quats": torch.optim.Adam([gaussian.gauss_params["quats"]], lr=cfg.quats_lr),
        "opacities": torch.optim.Adam([gaussian.gauss_params["opacities"]], lr=cfg.opacities_lr),
        "features_dc": torch.optim.Adam(
            [gaussian.gauss_params["features_dc"]], lr=cfg.features_dc_lr
        ),
        "features_rest": torch.optim.Adam(
            [gaussian.gauss_params["features_rest"]], lr=cfg.features_rest_lr
        ),
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

    # you may need to change scene_scale according to your dataset
    strategy_state = strategy.initialize_state(scene_scale=1.0)

    for step in tqdm(range(cfg.train_iters), desc="train"):
        for key, optimizer in optimizers_dict.items():
            optimizer.zero_grad()
        data_idx = index_random_chooser()
        sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        frame = dataset[data_idx]

        render, alpha, info = render_with_rasterizer(
            rasterization,
            gaussian,
            frame.K.to(device),
            frame.w2c.to(device),
            frame.image.shape[0],
            frame.image.shape[1],
            sh_degree_to_use
        )
        strategy.step_pre_backward(gaussian.gauss_params, optimizers, state, step, info)

        rgb = render[:, ..., :3]
        rgb = rgb.squeeze(0)
        gt = frame.image.to(device)
        l1 = torch.abs(gt - rgb).mean()
        ssim_loss = 1 - fused_ssim(rgb, gt)
        loss = cfg.ssim_loss_factor * ssim_loss + (1 - cfg.ssim_loss_factor) * l1
        loss.backward()
        for optimizer in optimizers_dict.values():
            optimizer.step()

        strategy.step_post_backward(
            params=gaussian.gauss_params,
            optimizers=optimizers_dict,
            state=strategy_state,
            step=step,
            info=info,
            packed=False,
        )

        ave_loss = loss_moving_averager(loss.item())
        ave_psnr = psnr_moving_averager(psnr(rgb, gt))
        if step % 100 == 0:
            print(f"step {step}, loss {ave_loss:.4f}, psnr {ave_psnr:.4f}")

    gaussian.save_ply(cfg.ply_save_path)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
