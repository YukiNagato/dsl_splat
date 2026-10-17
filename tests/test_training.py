from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from dsl_splat.colmap_utils import Reconstruction, qvec2rotmat, Image
from sklearn.neighbors import NearestNeighbors
import math
from dsl_splat.aaa.gsplat import rasterization
import random
from collections import deque
from gsplat.strategy.default import DefaultStrategy


class MovingAverager:
    def __init__(self, n):
        if n <= 0:
            raise ValueError("n must be positive")
        self.data = deque(maxlen=n)
        self.total = 0

    def __call__(self, x):
        if len(self.data) == self.data.maxlen:
            self.total -= self.data[0]
        self.data.append(x)
        self.total += x
        return self.total / len(self.data)


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def k_nearest(x, k):
    x_np = x.cpu().numpy()
    nn_model = NearestNeighbors(
        n_neighbors=k + 1, algorithm="auto", metric="euclidean"
    ).fit(x_np)
    distances, indices = nn_model.kneighbors(x_np)
    return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)


def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


@torch.no_grad()
def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return -10 * torch.log10(mse)


class IndexRandomChooser:
    def __init__(self, n):
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = n
        self.indices = []

    def __call__(self):
        if not self.indices:
            self.indices = list(range(self.n))
            random.shuffle(self.indices)
        return self.indices.pop()


@dataclass
class Config:
    train_iters: int = 30000
    densify_grad_thresh: float = 0.0008
    densify_size_thresh: float = 0.01
    split_screen_size: float = 0.04
    cull_screen_size: float = 0.15
    cull_scale_thresh: float = 0.5
    stop_screen_size_at: int = 4000
    warmup_length: int = 500
    stop_split_at: int = 15000
    reset_every: int = 5000
    cull_alpha_thresh: float = 0.05
    refine_every: int = 100

    densify_warm_up_iters: int = 2000
    densify_update_iters: int = 200
    max_gs_num: int = 1000000
    random_gs_num: int = 50000
    sh_degree: int = 3
    sh_degree_interval: int = 2000


@dataclass
class ImageFrame:
    K: torch.Tensor = None
    im_path: Path = None
    image: torch.Tensor = None
    w2c: torch.Tensor = None

    @classmethod
    def from_colmap_image(cls, root: Path, im: Image):
        qvec = im.qvec
        tvec = im.tvec
        w2c = np.identity(4)
        w2c[:3, :3] = qvec2rotmat(qvec)
        w2c[:3, 3] = tvec
        im_path = root / im.name
        image = cv2.imread(im_path)
        image = torch.from_numpy(image).float() / 255
        w2c = torch.from_numpy(w2c).float()
        return cls(im_path=im_path, image=image, w2c=w2c)


class Gaussian(nn.Module):
    def __init__(self, xyz, rgb, num_gs, sh_degree, random_scale=1.0):
        super().__init__()
        if xyz is not None:
            num_gs = len(xyz)
        else:
            xyz = (torch.rand((num_gs, 3)) - 0.5) * random_scale
        self.sh_degree = sh_degree
        means = torch.nn.Parameter(xyz.float().contiguous())
        distances, _ = k_nearest(means.data, 3)
        distances = torch.from_numpy(distances)
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        quats = torch.nn.Parameter(random_quat_tensor(num_gs).contiguous())
        opacities = torch.nn.Parameter(
            torch.logit(0.1 * torch.ones(num_gs, 1)).contiguous()
        )
        dim_sh = (sh_degree + 1) ** 2
        if rgb is not None:
            if sh_degree > 0:
                shs = torch.zeros((rgb.shape[0], dim_sh, 3)).float()
                shs[:, 0, :3] = RGB2SH(rgb / 260)
                shs[:, 1:, 3:] = 0.0
            else:
                shs[:, 0, :3] = torch.logit((rgb.float() + 1) / 260)
            features_dc = torch.nn.Parameter(shs[:, 0, :]).contiguous()
            features_rest = torch.nn.Parameter(shs[:, 1:, :]).contiguous()
        else:
            features_dc = torch.nn.Parameter(torch.rand(num_gs, 3)).contiguous()
            features_rest = torch.nn.Parameter(
                torch.zeros((num_gs, dim_sh - 1, 3))
            ).contiguous()

        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "opacities": opacities,
                "features_dc": features_dc,
                "features_rest": features_rest,
            }
        )

    def rasterize(self, K, viewmat, H, W, sh_degree_to_use=None):
        if sh_degree_to_use is None:
            sh_degree_to_use = self.sh_degree
        means = self.gauss_params["means"]
        scales = self.gauss_params["scales"]
        opacities = self.gauss_params["opacities"]
        quats = self.gauss_params["quats"]
        features_dc = self.gauss_params["features_dc"]
        features_rest = self.gauss_params["features_rest"]
        colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
        if self.sh_degree == 0:
            colors = torch.sigmoid(colors).squeeze(1)
        render, alpha, info = rasterization(
            means=means,
            quats=quats,  # rasterization does normalization internally
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


class Dataset:
    def __init__(self, data_root: Path, im_stride=8):
        image_folder_name = "images"
        if im_stride > 1:
            image_folder_name += f"_{im_stride}"
        self.image_folder = data_root / image_folder_name
        colmap_root = data_root / "sparse" / "0"
        reconstruction = Reconstruction.from_binary(colmap_root)
        camera = reconstruction.cameras[1]
        if im_stride > 1:
            camera.width = int(camera.width / im_stride)
            camera.height = int(camera.height / im_stride)
            camera.params = camera.params / im_stride
        K = np.array(
            [
                [camera.params[0], 0, camera.params[2]],
                [0, camera.params[1], camera.params[3]],
                [0, 0, 1],
            ]
        )
        K = torch.from_numpy(K).float()
        points_xyz = []
        points_rgb = []
        for point in reconstruction.points3D.values():
            points_xyz.append(point.xyz)
            points_rgb.append(point.rgb)
        self.points_xyz = torch.from_numpy(np.array(points_xyz)).float()
        self.points_rgb = torch.from_numpy(np.array(points_rgb)).float()
        colmap_images = list(reconstruction.images.values())
        self.frames = []
        for im in tqdm(colmap_images, desc="cache images"):
            frame = ImageFrame.from_colmap_image(self.image_folder, im)
            frame.K = K
            self.frames.append(frame)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx) -> ImageFrame:
        return self.frames[idx]


def train_loop(dataset: Dataset, cfg: Config, device="cuda"):
    gaussian = Gaussian(
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
        "means": torch.optim.Adam([gaussian.gauss_params["means"]], lr=1e-4),
        "scales": torch.optim.Adam([gaussian.gauss_params["scales"]], lr=0.01),
        "quats": torch.optim.Adam([gaussian.gauss_params["quats"]], lr=0.001),
        "opacities": torch.optim.Adam([gaussian.gauss_params["opacities"]], lr=0.05),
        "features_dc": torch.optim.Adam(
            [gaussian.gauss_params["features_dc"]], lr=0.0025
        ),
        "features_rest": torch.optim.Adam(
            [gaussian.gauss_params["features_rest"]], lr=0.0025 / 20
        ),
    }

    strategy_params = {
        "prune_opa": cfg.cull_alpha_thresh,
        "grow_grad2d": cfg.densify_grad_thresh,
        "grow_scale3d": cfg.densify_size_thresh,
        "grow_scale2d": cfg.split_screen_size,
        "prune_scale3d": cfg.cull_scale_thresh,
        "prune_scale2d": cfg.cull_screen_size,
        "refine_scale2d_stop_iter": cfg.stop_screen_size_at,
        "refine_start_iter": cfg.warmup_length,
        "refine_stop_iter": cfg.stop_split_at,
        "reset_every": cfg.reset_every,
        "refine_every": cfg.refine_every,
        "pause_refine_after_reset": 0,
        "absgrad": True,
        "revised_opacity": False,
        "verbose": True,
    }
    strategy = DefaultStrategy(**strategy_params)
    strategy_state = strategy.initialize_state(scene_scale=1.0)

    for step in tqdm(range(cfg.train_iters), desc="train"):
        for key, optimizer in optimizers_dict.items():
            optimizer.zero_grad()
        data_idx = index_random_chooser()
        sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        frame = dataset[data_idx]
        render, alpha, info = gaussian.rasterize(
            frame.K.to(device),
            frame.w2c.to(device),
            frame.image.shape[0],
            frame.image.shape[1],
            sh_degree_to_use,
        )
        info["means2d"].retain_grad()

        rgb = render[:, ..., :3]
        rgb = rgb.squeeze(0)
        gt = frame.image.to(device)
        loss = torch.abs(gt - rgb).mean()

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


if __name__ == "__main__":
    dataset = Dataset(Path("/root/autodl-tmp/data/garden"))
    cfg = Config()
    train_loop(dataset, cfg)
