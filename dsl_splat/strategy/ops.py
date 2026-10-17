# SPDX-License-Identifier: Apache-2.0
# Derived from gsplat v1.4.0, gsplat/strategy/ops.py (gsplat contributors).
# Upstream commit: 4d3a3b69db4de0326f983ccf7b7b255271a17b01.
# Modified for DSL Splat: selected helpers/imports removed and the optimizer
# helper renamed; this local adaptation is in progress.
# See README.md and LICENSE in this directory for source and license details.

import numpy as np
from typing import Callable, Dict, List, Union

import torch
import torch.nn.functional as F
from torch import Tensor


def normalized_quat_to_rotmat(quat: Tensor) -> Tensor:
    """Convert normalized quaternion to rotation matrix.

    Args:
        quat: Normalized quaternion in wxyz convension. (..., 4)

    Returns:
        Rotation matrix (..., 3, 3)
    """
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))


@torch.no_grad()
def update_param_with_optimizer(
    param_fn: Callable[[str, Tensor], Tensor],
    optimizer_fn: Callable[[str, Tensor], Tensor],
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
        optimizer = optimizers[name]
        for i, param_group in enumerate(optimizer.param_groups):
            p = param_group["params"][0]
            p_state = optimizer.state[p]
            del optimizer.state[p]
            for key in p_state.keys():
                if key != "step":
                    v = p_state[key]
                    p_state[key] = optimizer_fn(key, v)
            p_new = param_fn(name, p)
            optimizer.param_groups[i]["params"] = [p_new]
            optimizer.state[p_new] = p_state
            params[name] = p_new


@torch.no_grad()
def duplicate(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
):
    """Inplace duplicate the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to duplicate the Gaussians.
    """
    device = mask.device
    sel = torch.where(mask)[0]

    def param_fn(name: str, p: Tensor) -> Tensor:
        return torch.nn.Parameter(torch.cat([p, p[sel]]))

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.cat([v, torch.zeros((len(sel), *v.shape[1:]), device=device)])

    # update the parameters and the state in the optimizers
    update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat((v, v[sel]))


@torch.no_grad()
def split(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
    revised_opacity: bool = False,
):
    """Inplace split the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to split the Gaussians.
        revised_opacity: Whether to use revised opacity formulation
          from arXiv:2404.06109. Default: False.
    """
    device = mask.device
    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]

    scales = torch.exp(params["scales"][sel])
    quats = F.normalize(params["quats"][sel], dim=-1)
    rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
    samples = torch.einsum(
        "nij,nj,bnj->bni",
        rotmats,
        scales,
        torch.randn(2, len(scales), 3, device=device),
    )  # [2, N, 3]

    def param_fn(name: str, p: Tensor) -> Tensor:
        repeats = [2] + [1] * (p.dim() - 1)
        if name == "means":
            p_split = (p[sel] + samples).reshape(-1, 3)  # [2N, 3]
        elif name == "scales":
            p_split = torch.log(scales / 1.6).repeat(2, 1)  # [2N, 3]
        elif name == "opacities" and revised_opacity:
            new_opacities = 1.0 - torch.sqrt(1.0 - torch.sigmoid(p[sel]))
            p_split = torch.logit(new_opacities).repeat(repeats)  # [2N]
        else:
            p_split = p[sel].repeat(repeats)
        p_new = torch.cat([p[rest], p_split])
        p_new = torch.nn.Parameter(p_new)
        return p_new

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
        return torch.cat([v[rest], v_split])

    # update the parameters and the state in the optimizers
    update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            repeats = [2] + [1] * (v.dim() - 1)
            v_new = v[sel].repeat(repeats)
            state[k] = torch.cat((v[rest], v_new))


@torch.no_grad()
def remove(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
):
    """Inplace remove the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to remove the Gaussians.
    """
    sel = torch.where(~mask)[0]

    def param_fn(name: str, p: Tensor) -> Tensor:
        return torch.nn.Parameter(p[sel])

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return v[sel]

    # update the parameters and the state in the optimizers
    update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v[sel]


@torch.no_grad()
def reset_opa(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    value: float,
):
    """Inplace reset the opacities to the given post-sigmoid value.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        value: The value to reset the opacities
    """

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            opacities = torch.clamp(p, max=torch.logit(torch.tensor(value)).item())
            return torch.nn.Parameter(opacities)
        else:
            raise ValueError(f"Unexpected parameter name: {name}")

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.zeros_like(v)

    # update the parameters and the state in the optimizers
    update_param_with_optimizer(
        param_fn, optimizer_fn, params, optimizers, names=["opacities"]
    )


@torch.no_grad()
def relocate(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
    binoms: Tensor,
    min_opacity: float = 0.005,
):
    """Inplace relocate some dead Gaussians to the lives ones.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to indicates which Gaussians are dead.
    """
    # support "opacities" with shape [N,] or [N, 1]
    opacities = torch.sigmoid(params["opacities"])

    dead_indices = mask.nonzero(as_tuple=True)[0]
    alive_indices = (~mask).nonzero(as_tuple=True)[0]
    n = len(dead_indices)

    # Sample for new GSs
    eps = torch.finfo(torch.float32).eps
    probs = opacities[alive_indices].flatten()  # ensure its shape is [N,]
    sampled_idxs = _multinomial_sample(probs, n, replacement=True)
    sampled_idxs = alive_indices[sampled_idxs]
    new_opacities, new_scales = compute_relocation(
        opacities=opacities[sampled_idxs],
        scales=torch.exp(params["scales"])[sampled_idxs],
        ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        binoms=binoms,
    )
    new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            p[sampled_idxs] = torch.logit(new_opacities)
        elif name == "scales":
            p[sampled_idxs] = torch.log(new_scales)
        p[dead_indices] = p[sampled_idxs]
        return torch.nn.Parameter(p)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v[sampled_idxs] = 0
        return v

    # update the parameters and the state in the optimizers
    update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            v[sampled_idxs] = 0
