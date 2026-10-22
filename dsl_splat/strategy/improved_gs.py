# Copyright (c) 2026 ImprovedGS authors and Zhejiang University.
# Adapted for DSL Splat from Improved-GS: separated score collection, population
# policy and optimizer cadence; see IMPROVED_GS_LICENSE.txt and README.md.
# Inherits gsplat-derived DefaultStrategy (Apache-2.0; see LICENSE).
"""Improved-GS: EAS/LAS, recovery-aware pruning, growth control and MU cadence."""

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .default import DefaultStrategy
from .improved_gs_ops import split_long_axis
from .base import Params
from .ops import remove, reset_opa


@dataclass
class ImprovedGSStrategy(DefaultStrategy):
    """Full paper strategy with externally supplied per-Gaussian EAS scores.

    Rendering stays outside the strategy. After backward and optimizer updates,
    query ``needs_scores(state, step)`` and collect inference scores when True.
    Pass them to ``step_post_backward(..., scores=scores)``. After densification,
    use ``should_step_optimizer`` for MU and retain gradients between updates.
    See the demo and README for the complete ordering and paper-era defaults.
    """

    max_gs_num: int | None = 1_000_000
    absgrad: bool = True
    grow_grad2d: float = 0.0003
    prune_scale3d: float = math.inf
    split_distance: float = 0.45
    opacity_reduction: float = 0.6
    budget_warmup_offset: int = 500
    gradient_only_window: int = 500
    prune_stop_window: int = 100
    reset_opacity: float = 0.05
    initial_prune_iter: int | None = 300
    initial_prune_opa: float = 0.02
    rap_prune_ratio: float = 0.2
    rap_prune_offset: int = 300
    rap_rounds: int = 2
    mu_start_iter: int = 15_000
    mu_second_start_iter: int = 22_500
    mu_interval: int = 5
    mu_second_interval: int = 20

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.absgrad:
            raise ValueError("Improved-GS requires absolute screen gradients")
        if self.max_gs_num is None or self.max_gs_num <= 0:
            raise ValueError("Improved-GS requires a positive max_gs_num")
        if not 0 < self.split_distance <= 0.5 or not 0 < self.opacity_reduction <= 1:
            raise ValueError("invalid LAS split distance or opacity reduction")
        if not 0 < self.reset_opacity < 1 or not 0 <= self.rap_prune_ratio < 1:
            raise ValueError("invalid RAP reset opacity or pruning ratio")
        if self.refine_every <= 0 or self.reset_every <= 0:
            raise ValueError("refine_every and reset_every must be positive")
        if not 0 <= self.refine_start_iter < self.refine_stop_iter:
            raise ValueError("require 0 <= refine_start_iter < refine_stop_iter")
        for name in (
            "budget_warmup_offset",
            "gradient_only_window",
            "prune_stop_window",
            "rap_prune_offset",
            "rap_rounds",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.initial_prune_iter is not None and self.initial_prune_iter < 0:
            raise ValueError("initial_prune_iter must be nonnegative or None")
        if not 0 <= self.initial_prune_opa < 1:
            raise ValueError("initial_prune_opa must be in [0,1)")
        if not self.refine_stop_iter <= self.mu_start_iter < self.mu_second_start_iter:
            raise ValueError(
                "MU must start after densification, before its second stage"
            )
        if min(self.mu_interval, self.mu_second_interval) < 1:
            raise ValueError("MU intervals must be positive")
        if self.refine_scale2d_stop_iter != 0:
            raise ValueError(
                "Improved-GS does not use projected-size split/prune rules"
            )

    def initialize_state(self, scene_scale: float = 1.0) -> dict[str, Any]:
        state = super().initialize_state(scene_scale)
        state.update(
            rap_resets=0,
            rap_pending=[],
            initial_prune_done=False,
            n_split=0,
            n_pruned=0,
        )
        return state

    def check_sanity(
        self, params: Params, optimizers: dict[str, torch.optim.Optimizer]
    ) -> None:
        super().check_sanity(params, optimizers)
        n, device = len(params["means"]), params["means"].device
        for name, param in params.items():
            if param.ndim == 0 or len(param) != n or param.device != device:
                raise ValueError(f"{name} must have N Gaussian rows on {device}")
            group = optimizers[name].param_groups[0]["params"]
            if len(group) != 1 or group[0] is not param:
                raise ValueError(f"Optimizer {name} must own exactly its Parameter")
        for name, shape in (("means", (n, 3)), ("quats", (n, 4)), ("scales", (n, 3))):
            if params[name].shape != shape or params[name].dtype != torch.float32:
                raise ValueError(f"{name} must be float32 {shape}")
        if (
            params["opacities"].shape not in ((n,), (n, 1))
            or params["opacities"].dtype != torch.float32
        ):
            raise ValueError("opacities must be float32 logits [N] or [N,1]")

    def current_budget(self, step: int) -> int:
        end = self.refine_stop_iter - self.budget_warmup_offset
        progress = (
            1.0
            if end <= self.refine_start_iter
            else ((step - self.refine_start_iter) / (end - self.refine_start_iter))
        )
        assert self.max_gs_num is not None
        return int(self.max_gs_num * math.sqrt(min(max(progress, 0.0), 1.0)))

    def should_refine(self, state: dict[str, Any], step: int) -> bool:
        last = state.get("last_reset_step")
        paused = last is not None and step <= last + (
            self.pause_refine_after_reset or 0
        )
        return (
            self.refine_start_iter < step < self.refine_stop_iter
            and step % self.refine_every == 0
            and not paused
        )

    def needs_scores(self, state: dict[str, Any], step: int) -> bool:
        """Whether this step needs fresh no-grad inference scores (dense [N])."""
        return (
            self.should_refine(state, step)
            and step <= self.refine_stop_iter - self.gradient_only_window
        )

    def should_step_optimizer(self, step: int) -> bool:
        """Paper MU: single-view updates, then summed 5-view / 20-view updates."""
        if step <= self.mu_start_iter:
            return True
        interval = (
            self.mu_interval
            if step <= self.mu_second_start_iter
            else self.mu_second_interval
        )
        return step % interval == 0

    def step_pre_backward(
        self,
        params: Params,
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Any],
        step: int,
        info: dict[str, Any],
    ) -> None:
        if step < self.refine_stop_iter:
            super().step_pre_backward(params, optimizers, state, step, info)

    @torch.no_grad()
    def step_post_backward(
        self,
        params: Params,
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Any],
        step: int,
        info: dict[str, Any],
        packed: bool = False,
        *,
        scores: Tensor | None = None,
    ) -> None:
        n = len(params["means"])
        if self.needs_scores(state, step):
            if (
                scores is None
                or scores.shape != (n,)
                or scores.device != params["means"].device
            ):
                raise ValueError(
                    "Fresh dense EAS scores [N] are required on refinement steps"
                )
            if (
                scores.requires_grad
                or not bool(torch.isfinite(scores).all())
                or bool((scores < 0).any())
            ):
                raise ValueError("EAS scores must be finite, nonnegative and detached")
        state["n_split"], state["n_pruned"] = 0, 0
        if self.refine_start_iter < step < self.refine_stop_iter:
            self._update_state(params, state, info, packed=packed)
        if self.should_refine(state, step):
            n_split = self._grow_with_scores(params, optimizers, state, step, scores)
            if n_split:
                state["grad2d"].zero_()
                state["count"].zero_()
            state["n_split"] = n_split
            if step < self.refine_stop_iter - self.prune_stop_window:
                state["n_pruned"] += self._prune_opacity(
                    params, optimizers, state, self.prune_opa
                )
            if self.verbose:
                print(
                    f"Step {step}: {n_split} GSs split (LAS), budget {self.current_budget(step)}. "
                    f"Now having {len(params['means'])} GSs."
                )
        # RAP follows growth/pruning. No periodic resets outside densification.
        if (
            self.initial_prune_iter is not None
            and step == self.initial_prune_iter
            and not state["initial_prune_done"]
        ):
            state["n_pruned"] += self._prune_opacity(
                params, optimizers, state, self.initial_prune_opa
            )
            state["initial_prune_done"] = True
        self._prune_pending(params, optimizers, state, step)
        if (
            self.refine_start_iter < step < self.refine_stop_iter
            and step % self.reset_every == 0
        ):
            self._reset_opacities(params, optimizers, state, info)
            state["last_reset_step"] = step
            if state["rap_resets"] < self.rap_rounds and self.rap_prune_ratio > 0:
                due = step + self.rap_prune_offset
                if due < self.refine_stop_iter:
                    state["rap_pending"].append(due)
                    if due == step:
                        self._prune_pending(params, optimizers, state, step)
            state["rap_resets"] += 1
        if self.verbose and state["n_pruned"]:
            print(f"Step {step}: {state['n_pruned']} GSs pruned (Improved-GS).")

    def _prune_pending(
        self,
        params: Params,
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Any],
        step: int,
    ) -> None:
        pending = state["rap_pending"]
        for due in tuple(pending):
            if step >= due:
                if len(params["means"]):
                    opacity = params["opacities"].sigmoid().flatten()
                    threshold = torch.quantile(opacity, self.rap_prune_ratio)
                    state["n_pruned"] += self._remove(
                        params, optimizers, state, opacity < threshold
                    )
                pending.remove(due)

    def _grow_with_scores(
        self,
        params: Params,
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Any],
        step: int,
        scores: Tensor | None,
    ) -> int:
        if state["grad2d"] is None:
            return 0
        grads = torch.nan_to_num(
            state["grad2d"] / state["count"].clamp_min(1), nan=0, posinf=0, neginf=0
        )
        threshold = self.grow_grad2d
        if not self.needs_scores(state, step):
            scores = grads
            if len(grads) < self.current_budget(step):
                threshold /= 1.5
        assert scores is not None
        weights = torch.where(grads >= threshold, scores, 0)
        candidates = torch.where(weights > 0)[0]
        count = min(max(0, self.current_budget(step) - len(grads)), len(candidates))
        if count == 0:
            return 0
        chosen = candidates[
            torch.multinomial(weights[candidates], count, replacement=False)
        ]
        mask = torch.zeros(len(grads), dtype=torch.bool, device=grads.device)
        mask[chosen] = True
        split_long_axis(
            params, optimizers, state, mask, self.split_distance, self.opacity_reduction
        )
        return count

    @staticmethod
    def _remove(
        params: Params,
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Any],
        mask: Tensor,
    ) -> int:
        count = int(mask.sum().item())
        if count:
            remove(params, optimizers, state, mask)
        return count

    def _prune_opacity(
        self,
        params: Params,
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Any],
        threshold: float,
    ) -> int:
        return self._remove(
            params,
            optimizers,
            state,
            params["opacities"].sigmoid().flatten() < threshold,
        )

    def _reset_opacities(
        self,
        params: Params,
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Any],
        info: dict[str, Any],
    ) -> None:
        reset_opa(
            params,
            optimizers,
            state,
            self.reset_opacity,
            opacity_filter_variance=info.get("opacity_filter_variance"),
        )
