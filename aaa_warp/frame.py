"""Shared frame preparation and typed controls for compatibility adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .dispatch import FrameBindings, LaunchCache
from .preprocess import preprocess
from .settings import GaussianRasterizationSettings
from .types import GaussianInputs, PreprocessedGaussians


@dataclass(slots=True)
class PreparedFrame:
    state: PreprocessedGaussians
    bindings: FrameBindings


@dataclass(slots=True)
class AbsGradientTarget:
    """Publish a single view's statistics in its dense or packed destination."""

    target: Tensor
    gaussian_ids: Tensor | None = None

    @torch.no_grad()
    def write(self, gradient: Tensor) -> None:
        value = gradient if self.gaussian_ids is None else gradient[self.gaussian_ids]
        self.target.copy_(value)


@dataclass(slots=True)
class FrameOptions:
    principal_point: tuple[float, float] | None = None
    radius_clip: float = 0.0
    prepared: PreparedFrame | None = None
    screen_grad: bool = False
    absgrad_target: AbsGradientTarget | None = None


def prepare_frame(
    inputs: GaussianInputs,
    config: GaussianRasterizationSettings,
    launch_cache: LaunchCache,
    options: FrameOptions | None = None,
) -> PreparedFrame:
    """Preprocess once, preserving each frame's owners and clipping sentinels."""
    if options is not None and options.prepared is not None:
        return options.prepared
    bindings = FrameBindings(launch_cache)
    state = preprocess(
        **{name: value for name, value in inputs.items() if name != "means2D"},
        raster_settings=config,
        principal_point=None if options is None else options.principal_point,
        _bindings=bindings,
    )
    if options is not None and options.radius_clip > 0:
        keep = state["radii"] > options.radius_clip
        state["radii"] = torch.where(keep, state["radii"], 0)
        state["tiles_touched"] = torch.where(keep, state["tiles_touched"], 0)
        state["valid"] = state["valid"] & keep
    return PreparedFrame(state, bindings)
