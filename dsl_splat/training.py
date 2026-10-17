"""Layout utilities for per-view densification metadata and gradient targets.

Renderers supply the actual geometry and define its meaning. These helpers
only pack visible rows and bind auxiliary gradient destinations.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .inputs import Inputs
from .runtime.gradients import AbsGradientTarget


@dataclass(slots=True)
class TrainingMetadata:
    values: dict[str, Any]
    screen: Tensor | None
    absolute_targets: list[AbsGradientTarget | None]


def prepare_training_metadata(
    dense: dict[str, Tensor], layout: Inputs, *, packed: bool, absgrad: bool
) -> TrainingMetadata:
    if packed:
        batch_ids, camera_ids, gaussian_ids = (
            (dense["radii"] > 0).all(-1).nonzero(as_tuple=True)
        )
        values = {
            name: value[batch_ids, camera_ids, gaussian_ids]
            for name, value in dense.items()
        }
    else:
        batch_ids = camera_ids = gaussian_ids = None
        values = {
            name: value.reshape(layout.batch_dims + tuple(value.shape[1:]))
            for name, value in dense.items()
        }
    values.update(batch_ids=batch_ids, camera_ids=camera_ids, gaussian_ids=gaussian_ids)
    screen = values["means2d"].detach().requires_grad_(layout.requires_screen_grad)
    values["means2d"] = screen
    targets: list[AbsGradientTarget | None] = [None] * (
        layout.batch_count * layout.cameras
    )
    if not layout.requires_screen_grad:
        return TrainingMetadata(values, None, targets)
    if packed:
        dense_screen = torch.zeros_like(dense["means2d"]).index_put(
            (batch_ids, camera_ids, gaussian_ids), screen
        )
    else:
        dense_screen = screen.reshape(layout.batch_count, layout.cameras, layout.n, 2)
    if absgrad:
        absolute = torch.zeros_like(screen)
        setattr(screen, "absgrad", absolute)
        if packed:
            counts = (
                [len(screen)]
                if len(targets) == 1
                else (dense["radii"] > 0).all(-1).sum(-1).flatten().tolist()
            )
            begin = 0
            for i, count in enumerate(counts):
                end = begin + count
                targets[i] = AbsGradientTarget(
                    absolute[begin:end], gaussian_ids[begin:end]
                )
                begin = end
        else:
            absolute_views = absolute.reshape(
                layout.batch_count * layout.cameras, layout.n, 2
            )
            targets = [AbsGradientTarget(value) for value in absolute_views]
    return TrainingMetadata(values, dense_screen, targets)
