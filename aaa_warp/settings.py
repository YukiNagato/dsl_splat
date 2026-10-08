"""Standalone Python settings compatible with the AAA forward interface.

Importing settings never imports the reference CUDA extension. Enum values,
defaults and serialized fields follow the reference Python settings.
"""

from copy import copy
from dataclasses import asdict, dataclass, field
from enum import IntEnum
import json
from pathlib import Path
from typing import Any, NamedTuple

import torch


class SortMode(IntEnum):
    GLOBAL = 0
    PPX_FULL = 1
    PPX_KBUFFER = 2
    HIER = 3


class GlobalSortOrder(IntEnum):
    Z_DEPTH = 0
    DISTANCE = 1
    PTD_CENTER = 2
    PTD_MAX = 3


@dataclass
class SortQueueSizes:
    tile_4x4: int = 64
    tile_2x2: int = 8
    per_pixel: int = 4


@dataclass
class SortSettings:
    queue_sizes: SortQueueSizes = field(default_factory=SortQueueSizes)
    sort_mode: SortMode = SortMode.GLOBAL
    sort_order: GlobalSortOrder = GlobalSortOrder.Z_DEPTH


@dataclass
class CullingSettings:
    rect_bounding: bool = False
    tight_opacity_bounding: bool = False
    tile_based_culling: bool = False
    hierarchical_4x4_culling: bool = False


@dataclass
class ExtendedSettings:
    sort_settings: SortSettings = field(default_factory=SortSettings)
    culling_settings: CullingSettings = field(default_factory=CullingSettings)
    load_balancing: bool = False
    proper_ewa_scaling: bool = False
    eval_3D: bool = False
    new_aabb: bool = True

    def copy(self) -> "ExtendedSettings":
        """Snapshot all mutable settings nodes without recursive deepcopy."""
        return copy_settings(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(
            self,
            dict_factory=lambda items: {
                key: int(value) if isinstance(value, IntEnum) else value
                for key, value in items
            },
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @staticmethod
    def from_dict(values: dict[str, Any]) -> "ExtendedSettings":
        values = dict(values)
        sort = dict(values.pop("sort_settings", {}))
        queues = SortQueueSizes(**sort.pop("queue_sizes", {}))
        if "sort_mode" in sort:
            sort["sort_mode"] = SortMode(sort["sort_mode"])
        if "sort_order" in sort:
            sort["sort_order"] = GlobalSortOrder(sort["sort_order"])
        culling = CullingSettings(**values.pop("culling_settings", {}))
        return ExtendedSettings(
            sort_settings=SortSettings(queue_sizes=queues, **sort),
            culling_settings=culling,
            **values,
        )

    @staticmethod
    def from_json(path: str | Path) -> "ExtendedSettings":
        with open(path) as source:
            return ExtendedSettings.from_dict(json.load(source))

    def set_value(self, key: str, value: Any) -> None:
        for settings in (
            self,
            self.culling_settings,
            self.sort_settings,
            self.sort_settings.queue_sizes,
        ):
            if key in settings.__dataclass_fields__:
                setattr(settings, key, value)
                return
        raise ValueError(f"unknown setting: {key}")


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    inv_viewprojmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    settings: ExtendedSettings
    render_depth: bool
    debug: bool


def copy_settings(settings: ExtendedSettings) -> ExtendedSettings:
    """Copy the four mutable nodes, including structurally compatible settings."""
    result = copy(settings)
    result.sort_settings = copy(settings.sort_settings)
    result.sort_settings.queue_sizes = copy(settings.sort_settings.queue_sizes)
    result.culling_settings = copy(settings.culling_settings)
    return result
