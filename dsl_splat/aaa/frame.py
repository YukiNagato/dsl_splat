"""Shared frame preparation and typed controls for compatibility adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..runtime.dispatch import FrameBindings, LaunchCache
from ..runtime.gradients import AbsGradientTarget
from ..statistics import PreparedStatistics
from .preprocess import preprocess
from .config import FrameConfig
from .types import GaussianInputs, PreprocessedGaussians


@dataclass(slots=True)
class PreparedFrame:
    state: PreprocessedGaussians
    bindings: FrameBindings


@dataclass(slots=True)
class FrameOptions:
    principal_point: tuple[float, float] | None = None
    radius_clip: float = 0.0
    prepared: PreparedFrame | None = None
    screen_grad: bool = False
    absgrad_target: AbsGradientTarget | None = None
    statistics: PreparedStatistics | None = None


def prepare_frame(
    inputs: GaussianInputs,
    config: FrameConfig,
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
        principal_point=(
            config.principal_point
            if options is None or options.principal_point is None
            else options.principal_point
        ),
        _bindings=bindings,
    )
    if options is not None and options.radius_clip > 0:
        keep = state["radii"] > options.radius_clip
        state["radii"] = torch.where(keep, state["radii"], 0)
        state["tiles_touched"] = torch.where(keep, state["tiles_touched"], 0)
        state["valid"] = state["valid"] & keep
    return PreparedFrame(state, bindings)
