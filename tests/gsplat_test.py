# import warp as wp
# import torch

# # Warp Debug
# wp.config.mode = "debug"
# wp.config.verify_cuda = True
# # wp.config.print_launches = True
# # wp.config.log_level = wp.LOG_DEBUG
# # wp.config.launch_array_access_mode = (
# #     wp.config.LaunchArrayAccessMode.CHECKED
# # )

# wp.init()

import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
from dsl_splat.aaa.gsplat import rasterization as dsl_splat_rasterization

# from fused_ssim import fused_ssim
from typing import Union, Dict, List
import math


@torch.no_grad()
def _update_param_with_optimizer(
    param_fn,
    optimizer_fn,
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    names: Union[List[str], None] = None,
):
    """Update the parameters and the state in the optimizers with defined functions.

    Args:
        param_fn: A function that takes the name of the parameter and the parameter itself,
            and returns the new parameter.
        optimizer_fn: A function that takes the key of the optimizer state and the state value,
            and returns the new state value.
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        names: A list of key names to update. If None, update all. Default: None.
    """
    if names is None:
        # If names is not provided, update all parameters
        names = list(params.keys())

    for name in names:
        param = params[name]
        new_param = param_fn(name, param)
        params[name] = new_param
        if name not in optimizers:
            assert not param.requires_grad, (
                f"Optimizer for {name} is not found, but the parameter is trainable."
                f"Got requires_grad={param.requires_grad}"
            )
            continue
        optimizer = optimizers[name]
        for i in range(len(optimizer.param_groups)):
            param_state = optimizer.state[param]
            del optimizer.state[param]
            for key in param_state.keys():
                if key != "step":
                    v = param_state[key]
                    param_state[key] = optimizer_fn(key, v)
            optimizer.param_groups[i]["params"] = [new_param]
            optimizer.state[new_param] = param_state


class SimpleStrategy:
    def __init__(self):
        self.key_for_gradient = "means2d"
        self.state = {"grad2d": None, "count": None, "scene_scale": 1.0}

    @torch.no_grad
    def update(self, params, info):
        grads = info[self.key_for_gradient].absgrad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        n_gaussian = len(list(params.values())[0])

        state = self.state
        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)

        sel = (info["radii"] > 0.0).all(dim=-1)  # [C, N]
        gs_ids = torch.where(sel)[1]  # [nnz]
        grads = grads[sel]  # [nnz, 2]
        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )

    def densify_and_prune(self, params, optimizers, step):
        num_gaussian = len(list(params.values())[0])
        mask_num = int(num_gaussian * 0.1)

        idx = torch.randperm(num_gaussian)[:mask_num]
        mask = torch.zeros(num_gaussian, dtype=torch.bool)
        mask[idx] = True

        sel = torch.where(~mask)[0]

        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            return torch.nn.Parameter(p[sel], requires_grad=p.requires_grad)

        def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return v[sel]

        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        # update the extra running state
        state = self.state
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v[sel]
        state["grad2d"].zero_()
        state["count"].zero_()
        torch.cuda.empty_cache()


class Gaussian:
    def __init__(self, device):
        sh_degree = 1
        dim_sh = (sh_degree + 1) ** 2
        device = "cuda"
        N_points = 1000001
        means = nn.Parameter(
            (torch.rand(N_points, 3) * 100 - 50).contiguous().to(device)
        )  # [-50, 50]
        scales = nn.Parameter(
            torch.log(torch.rand(N_points, 3) * 0.1 + 0.01).contiguous().to(device)
        )
        opacities = torch.nn.Parameter(
            torch.logit(0.1 * torch.ones(N_points, 1)).contiguous().to(device)
        )
        quats = torch.nn.Parameter(random_quat_tensor(N_points).contiguous().to(device))
        shs = torch.zeros((N_points, dim_sh, 3)).float().cuda()
        features_dc = torch.nn.Parameter(shs[:, 0, :])
        features_rest = torch.nn.Parameter(shs[:, 1:, :])
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "opacities": opacities,
                "shs": shs,
                "features_dc": features_dc,
                "features_rest": features_rest,
            }
        )

    def __len__(self):
        return len(self.gauss_params["means"])


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


random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)


H, W = 1000, 1000
im_num = 10
sh_degree = 1
dim_sh = (sh_degree + 1) ** 2
device = "cuda"

gaussian = Gaussian(device)

K = (
    torch.tensor([[W / 2, 0, W / 2], [0, H / 2, H / 2], [0, 0, 1]])
    .float()
    .to(device)[None]
    .repeat(len(gaussian), 1, 1)
)
poses = []
for degree in range(0, 360, im_num):
    theta = np.deg2rad(degree)
    c = np.cos(theta)
    s = np.sin(theta)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    poses.append(pose)
poses = torch.tensor(poses).float().to(device)

gt_images = [torch.rand(H, W, 3).float().to(device) for _ in range(im_num)]

densify_step = 10
total_step = 1000
# params = [means, scales, opacities, quats, shs, features_dc]
# optimizer = torch.optim.Adam(params, lr=0.00001) # make small lr

optimizers_dict = {}
# params_dict = get_param_dict()
for name, param in gaussian.gauss_params.items():
    optimizers_dict[name] = torch.optim.Adam([param], lr=0.00001)

sh_degree_to_use = 0
strategy = SimpleStrategy()

for step in tqdm(range(total_step)):
    im_idx = random.randint(0, im_num - 1)
    for optimizer in optimizers_dict.values():
        optimizer.zero_grad()

    means = gaussian.gauss_params["means"]
    scales = gaussian.gauss_params["scales"]
    opacities = gaussian.gauss_params["opacities"]
    quats = gaussian.gauss_params["quats"]
    shs = gaussian.gauss_params["shs"]
    features_dc = gaussian.gauss_params["features_dc"]
    features_rest = gaussian.gauss_params["features_rest"]

    colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
    viewmat = torch.linalg.inv(poses[im_idx])

    k = K[im_idx]
    render, alpha, info = dsl_splat_rasterization(
        means=means,
        quats=quats,  # rasterization does normalization internally
        scales=torch.exp(scales),
        opacities=torch.sigmoid(opacities).squeeze(-1),
        colors=colors,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=k[None],  # [1, 3, 3]
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

    info["means2d"].retain_grad()
    alpha = alpha[:, ...]

    background = torch.rand(3, device=device)
    rgb = render[:, ..., :3] + (1 - alpha) * background
    rgb = rgb.squeeze(0)
    gt = gt_images[im_idx]
    # ssim_loss = 1 - fused_ssim(rgb.permute(2, 0, 1)[None, ...], gt.permute(2, 0, 1)[None, ...])
    Ll1 = torch.abs(gt - rgb).mean()
    # loss = 0.8 * Ll1 + 0.2 * ssim_loss
    loss = Ll1
    loss.backward()

    for optimizer in optimizers_dict.values():
        optimizer.step()

    # strategy.update(gaussian.gauss_params, info)
    # if step > 0 and step % densify_step == 0:
    #     strategy.densify_and_prune(gaussian.gauss_params, optimizers_dict, step)
