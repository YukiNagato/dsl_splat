"""Frame-owned training metadata in the original CUDA pixel coordinates."""

import torch
from torch import Tensor
import warp as wp

from .projection import ProjectedGaussians
from ..runtime.dispatch import FrameBindings, kernel_arg, launch
from ..runtime.interop import current_stream


@wp.kernel(enable_backward=False)
def _prepare(
    radii: wp.array(dtype=int),
    centers: wp.array(dtype=wp.vec2),
    conic_opacity: wp.array(dtype=wp.vec4),
    means2d: wp.array(dtype=wp.vec2),
    opacities: wp.array(dtype=float),
    conics: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    if radii[i] > 0:
        value = conic_opacity[i]
        means2d[i] = centers[i]
        opacities[i] = value[3]
        conics[i] = wp.vec3(value[0], value[1], value[2])
    else:
        # Projection leaves culled centers/conics unwritten. Do not read them.
        means2d[i] = wp.vec2(0.0)
        opacities[i] = 0.0
        conics[i] = wp.vec3(0.0)


@torch.no_grad()
def prepare_metadata(
    state: ProjectedGaussians, bindings: FrameBindings
) -> dict[str, Tensor]:
    """Copy visible rows and zero culled rows in one pass, without autograd.

    means2d uses integer pixel centers, exactly as the CUDA rendering kernel.
    Its auxiliary gradient is attached separately by prepare_training_metadata.
    """
    n = len(state.radii)
    means2d = torch.empty_like(state.centers)
    opacities = state.conics.new_empty((n,))
    conics = state.conics.new_empty((n, 3))
    if n:
        launch(
            _prepare,
            dim=n,
            inputs=[
                kernel_arg(state.radii, bindings=bindings),
                kernel_arg(state.centers, wp.vec2, bindings),
                kernel_arg(state.conics, wp.vec4, bindings),
            ],
            outputs=[
                kernel_arg(means2d, wp.vec2, bindings),
                kernel_arg(opacities, bindings=bindings),
                kernel_arg(conics, wp.vec3, bindings),
            ],
            stream=current_stream(state.radii.device, bindings=bindings),
            bindings=bindings,
        )
    return dict(
        radii=state.radii[:, None].expand(-1, 2),
        means2d=means2d,
        opacities=opacities,
        tiles_per_gauss=state.tile_counts,
        conics=conics,
    )
