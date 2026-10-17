"""Lowered per-view parameters shared by AAA's kernel stages.

Public cameras use OpenCV world-to-camera matrices and K. These fields hold
the derived, transposed matrices required by the existing AAA arithmetic;
they are an internal kernel contract, not a second public camera convention.
"""

from typing import Any, NamedTuple

from torch import Tensor

from .settings import ExtendedSettings


class FrameConfig(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: Tensor
    scale_modifier: float
    viewmatrix: Tensor
    projmatrix: Tensor
    inv_viewprojmatrix: Tensor
    sh_degree: int
    campos: Tensor
    prefiltered: bool
    settings: ExtendedSettings
    render_depth: bool
    debug: bool
    principal_point: tuple[float, float] | None = None


def legacy_frame_config(config: Any) -> FrameConfig:
    """Adapt original 3DGS settings without copying any tensor storage."""
    if isinstance(config, FrameConfig):
        return config
    return FrameConfig(*(getattr(config, name) for name in FrameConfig._fields[:-1]))


def default_settings(*, antialiased: bool = True) -> ExtendedSettings:
    """AAA's recommended eval_3D, HIER 64/8/4 and culling configuration."""
    from .settings import CullingSettings, GlobalSortOrder, SortMode, SortSettings

    return ExtendedSettings(
        sort_settings=SortSettings(
            sort_mode=SortMode.HIER, sort_order=GlobalSortOrder.PTD_MAX
        ),
        culling_settings=CullingSettings(True, True, True, True),
        load_balancing=True,
        proper_ewa_scaling=antialiased,
        eval_3D=True,
        new_aabb=True,
    )


def validate_settings(settings: ExtendedSettings) -> None:
    """Validate the algorithm implemented by both differentiable entry points."""
    if not isinstance(settings, ExtendedSettings):
        raise ValueError("settings must be ExtendedSettings")
    if not settings.eval_3D or int(settings.sort_settings.sort_mode) != 3:
        raise ValueError("settings must use eval_3D=True and HIER sorting")
    queues = settings.sort_settings.queue_sizes
    if (queues.tile_4x4, queues.tile_2x2, queues.per_pixel) != (64, 8, 4):
        raise ValueError("settings require HIER queue sizes 64/8/4")
