# SPDX-License-Identifier: Apache-2.0
# Adapted from gsplat v1.4.0, strategy/mcmc.py (gsplat contributors).
# Modified for DSL Splat: Warp numerical ops, validation, all-dead handling,
# and optional noise cutoff. See README.md and LICENSE for provenance.
"""Renderer-independent MCMC population strategy with Warp numerical kernels."""

import math
from dataclasses import dataclass
from typing import Any

import torch

from .base import Strategy
from .mcmc_ops import (
    Params,
    binomial_table,
    inject_noise_to_position,
    relocate,
    sample_add,
)


@dataclass
class MCMCStrategy(Strategy):
    """3D Gaussian Splatting as Markov Chain Monte Carlo.

    Relocate low-opacity Gaussians, grow by 5% per refinement up to ``cap_max``,
    and perturb positions each step. There is no periodic opacity reset or
    gradient-based splitting. Parameters use world-space ``means [N,3]``, log
    ``scales [N,3]``, wxyz ``quats [N,4]`` and opacity logits ``[N]`` or ``[N,1]``.
    Additional per-Gaussian Parameters (SH/features, etc.) are copied as well.

    Call ``step_post_backward`` **after optimizer.step()**, passing the current
    means learning rate as ``lr``. All parameters need a corresponding optimizer
    with one group containing that parameter, as with DefaultStrategy. Float32
    CPU/CUDA tensors are supported. Torch's RNG controls sampling and noise.

    Refinement occurs at multiples of ``refine_every`` strictly between start
    and stop. Noise continues after refinement stops; ``noise_stop_iter=None``
    preserves that gsplat behavior, or set an exclusive noise cutoff.
    """

    cap_max: int = 1_000_000
    noise_lr: float = 5e5
    refine_start_iter: int = 500
    refine_stop_iter: int = 25_000
    refine_every: int = 100
    min_opacity: float = 0.005
    verbose: bool = False
    noise_stop_iter: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cap_max, int) or self.cap_max < 0:
            raise ValueError("cap_max must be a nonnegative integer.")
        if not isinstance(self.refine_every, int) or self.refine_every <= 0:
            raise ValueError("refine_every must be a positive integer.")
        if self.refine_start_iter < 0 or self.refine_stop_iter < self.refine_start_iter:
            raise ValueError("Expected 0 <= refine_start_iter <= refine_stop_iter.")
        if not math.isfinite(self.noise_lr) or self.noise_lr < 0:
            raise ValueError("noise_lr must be finite and nonnegative.")
        if not 0.0 < self.min_opacity < 1.0:
            raise ValueError("min_opacity must be between 0 and 1.")
        if self.noise_stop_iter is not None and self.noise_stop_iter < 0:
            raise ValueError("noise_stop_iter must be nonnegative or None.")

    def initialize_state(self) -> dict[str, Any]:
        """Small serializable state; contains no persistent Gaussian buffers."""
        return {"binoms": binomial_table(), "n_relocated": 0, "n_added": 0}

    def check_sanity(
        self, params: Params, optimizers: dict[str, torch.optim.Optimizer]
    ) -> None:
        super().check_sanity(params, optimizers)
        for key in ("means", "scales", "quats", "opacities"):
            if key not in params:
                raise ValueError(f"MCMC requires params[{key!r}].")
        n, device = len(params["means"]), params["means"].device
        if n > self.cap_max:
            raise ValueError("Initial Gaussian count exceeds cap_max.")
        shapes = {"means": (n, 3), "scales": (n, 3), "quats": (n, 4)}
        for name, shape in shapes.items():
            if params[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}.")
        if params["opacities"].shape not in ((n,), (n, 1)):
            raise ValueError("opacities must have shape [N] or [N,1].")
        for name, param in params.items():
            if param.ndim == 0 or len(param) != n or param.device != device:
                raise ValueError(f"{name} must have N rows on {device}.")
            if name in (*shapes, "opacities") and param.dtype != torch.float32:
                raise TypeError(f"{name} must be float32.")
            group = optimizers[name].param_groups[0]["params"]
            if len(group) != 1 or group[0] is not param:
                raise ValueError(
                    f"Optimizer for {name} must own exactly that Parameter."
                )

    @torch.no_grad()
    def step_post_backward(
        self,
        params: Params,
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Any],
        step: int,
        info: dict[str, Any],
        lr: float,
    ) -> None:
        """Refine then perturb; ``info`` is accepted for strategy compatibility."""
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("lr must be the finite, nonnegative current means LR.")
        n = len(params["means"])
        if n > self.cap_max:
            raise ValueError("Gaussian count exceeds cap_max.")
        if (
            self.refine_start_iter < step < self.refine_stop_iter
            and step % self.refine_every == 0
            and n
        ):
            binoms = state["binoms"].to(params["means"].device)
            state["binoms"] = binoms
            dead = params["opacities"].detach().sigmoid().flatten() <= self.min_opacity
            n_relocated = relocate(params, optimizers, dead, binoms, self.min_opacity)
            # With no live donor, skip growth instead of sampling an empty set.
            n_added = min(self.cap_max, int(1.05 * n)) - n
            if n_relocated == 0 and bool(dead.all()):
                n_added = 0
            sample_add(params, optimizers, n_added, binoms, self.min_opacity)
            state.update(n_relocated=n_relocated, n_added=n_added)
            if self.verbose:
                print(
                    f"Step {step}: Relocated {n_relocated}, added {n_added} GSs. "
                    f"Now having {len(params['means'])} GSs."
                )
        if self.noise_stop_iter is None or step < self.noise_stop_iter:
            inject_noise_to_position(params, lr * self.noise_lr)
