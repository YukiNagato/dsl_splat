# SPDX-License-Identifier: Apache-2.0
# Adapted from gsplat's relocation and strategy operations; see README.md.
"""MCMC numerical operations and Gaussian/optimizer population updates."""

import math

import torch
from torch import Tensor, nn
import warp as wp

from ..runtime.interop import current_stream
from . import _mcmc_kernels as kernels
from .ops import update_param_with_optimizer
from .base import Params, Optimizers


def binomial_table() -> Tensor:
    """Equation 9 lookup table, built once on CPU and moved on refinement."""
    return torch.tensor(
        [[math.comb(n, k) if k <= n else 0 for k in range(51)] for n in range(51)],
        dtype=torch.float64,
    )


def _launch(
    kernel: wp.Kernel,
    n: int,
    tensors: list[Tensor],
    scalars: list[float],
    outputs: tuple[Tensor, ...] = (),
) -> None:
    if n == 0:
        return
    wp.init()
    device = tensors[0].device
    wp.launch(
        kernel,
        dim=n,
        inputs=[wp.from_torch(t, requires_grad=False) for t in tensors] + scalars,
        outputs=[wp.from_torch(t, requires_grad=False) for t in outputs],
        device=wp.device_from_torch(device),
        stream=current_stream(device) if device.type == "cuda" else None,
    )


@torch.no_grad()
def compute_relocation(
    opacities: Tensor,
    scales: Tensor,
    ratios: Tensor,
    binoms: Tensor,
    min_opacity: float = 0.005,
) -> tuple[Tensor, Tensor]:
    """Equation 9 on activated float32 opacity [N] and scales [N,3].

    Multiplicities are clamped to [1,51], following gsplat. The opacity floor
    is applied before scale compensation. The inputs are never mutated.
    """
    n = opacities.numel()
    if opacities.shape != (n,) or scales.shape != (n, 3) or ratios.shape != (n,):
        raise ValueError("Expected opacities [N], scales [N,3], ratios [N].")
    if opacities.dtype != torch.float32 or scales.dtype != torch.float32:
        raise TypeError("MCMC relocation requires float32 opacities and scales.")
    if not 0.0 < min_opacity < 1.0:
        raise ValueError("min_opacity must be between 0 and 1.")
    if any(t.device != opacities.device for t in (scales, ratios, binoms)):
        raise ValueError("Relocation tensors must be on the same device.")
    if binoms.shape != (51, 51) or binoms.dtype != torch.float64:
        raise ValueError("Use binomial_table() on the input device.")
    new_opacities, new_scales = torch.empty_like(opacities), torch.empty_like(scales)
    _launch(
        kernels.relocation,
        n,
        [opacities, scales, ratios.to(torch.int64), binoms],
        [min_opacity],
        (new_opacities, new_scales),
    )
    return new_opacities, new_scales


@torch.no_grad()
def inject_noise_to_position(params: Params, noise_scale: float) -> None:
    """Add sigmoid-gated covariance noise in-place, using Torch's RNG/stream.

    ``scales`` are log scales; ``opacities`` are logits ([N] or [N,1]);
    ``quats`` are wxyz and normalized in the kernel. No covariance buffer is
    materialized. Strided parameters, including strided means, are supported.
    """
    if noise_scale == 0 or len(params["means"]) == 0:
        return
    means = params["means"]
    noise = torch.randn_like(means)
    _launch(
        kernels.perturb_positions,
        len(means),
        [means, params["quats"], params["scales"], params["opacities"].view(-1), noise],
        [noise_scale, 0.005, 100.0],
    )


def _sample_parents(weights: Tensor, n: int) -> Tensor:
    # torch.multinomial has a 2^24-category limit. Keep larger populations on
    # the device too, and use the same Torch RNG rather than NumPy's RNG.
    if len(weights) <= 2**24:
        return torch.multinomial(weights, n, replacement=True)
    cdf = weights.double().cumsum(0)
    draws = torch.rand(n, device=weights.device, dtype=cdf.dtype) * cdf[-1]
    return torch.searchsorted(cdf, draws, right=True).clamp_max_(len(weights) - 1)


def _redistribute(
    params: Params, sampled: Tensor, binoms: Tensor, min_opacity: float
) -> Tensor:
    # Evaluate and write each donor once, even when it is sampled repeatedly.
    parents, counts = sampled.unique(return_counts=True)
    opacity, scale = compute_relocation(
        params["opacities"][parents].sigmoid().flatten(),
        params["scales"][parents].exp(),
        counts + 1,
        binoms,
        min_opacity,
    )
    params["opacities"][parents] = opacity.logit().reshape_as(
        params["opacities"][parents]
    )
    params["scales"][parents] = scale.log()
    return parents


@torch.no_grad()
def relocate(
    params: Params,
    optimizers: Optimizers,
    dead: Tensor,
    binoms: Tensor,
    min_opacity: float,
) -> int:
    """Replace dead rows, reset donor/destination moments, retain Parameters."""
    destinations = torch.where(dead)[0]
    alive = torch.where(~dead)[0]
    if len(destinations) == 0 or len(alive) == 0:
        return 0
    weights = params["opacities"][alive].sigmoid().flatten()
    sampled = alive[_sample_parents(weights, len(destinations))]
    parents = _redistribute(params, sampled, binoms, min_opacity)
    for name, param in params.items():
        param[destinations] = param[sampled]
        # New identities must not inherit the dead Gaussian's Adam history.
        for key, value in optimizers[name].state.get(param, {}).items():
            if key != "step":
                value[parents] = 0
                value[destinations] = 0
    return len(destinations)


@torch.no_grad()
def sample_add(
    params: Params,
    optimizers: dict[str, torch.optim.Optimizer],
    n: int,
    binoms: Tensor,
    min_opacity: float,
) -> None:
    """Append opacity-weighted samples with zero moments for their new rows."""
    if n <= 0:
        return
    sampled = _sample_parents(params["opacities"].sigmoid().flatten(), n)
    _redistribute(params, sampled, binoms, min_opacity)

    def param_fn(name: str, param: Tensor) -> nn.Parameter:
        return nn.Parameter(
            torch.cat((param, param[sampled])), requires_grad=param.requires_grad
        )

    def optimizer_fn(key: str, value: Tensor) -> Tensor:
        return torch.cat((value, value.new_zeros((n, *value.shape[1:]))))

    update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
