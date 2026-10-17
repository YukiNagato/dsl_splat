"""Differentiable single-view projection and rendering for original 3DGS.

The complete API keeps its fused Function. These independent Functions serve
callers who explicitly reuse geometry or compose rendering stages.
"""

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch.nn.functional import normalize

from . import projection, render
from .autograd import launch_cache
from .binning import bin_and_sort as _bin_and_sort
from .cameras import ViewConfig, prepare_camera
from ..cameras import PinholeCamera
from ..inputs import check_tensor, spherical_harmonics
from ..primitives.tiling import TileBins
from ..runtime.dispatch import FrameBindings
from ..statistics import PreparedStatistics, StatisticsRequest, prepare_statistics


@dataclass(slots=True)
class GaussianGeometry:
    """Only rows with radii>0 define centers and conics; bounds are discrete."""

    centers: Tensor
    conics: Tensor
    radii: Tensor
    tile_counts: Tensor
    depths: Tensor
    _config: ViewConfig = field(repr=False)

    def _state(self) -> projection.ProjectedGaussians:
        return projection.ProjectedGaussians(
            self.centers, self.conics, self.radii, self.tile_counts, self.depths
        )


@dataclass(slots=True)
class GaussianTiles:
    ranges: Tensor
    gaussian_ids: Tensor
    _geometry: GaussianGeometry = field(repr=False)


@dataclass(slots=True)
class PixelResult:
    features: Tensor
    alpha: Tensor
    statistics: dict[str, Tensor] = field(default_factory=dict)


class _Project(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        opacities: Tensor,
        view: Tensor,
        full: Tensor,
        intrinsics: Tensor,
        config: ViewConfig,
        grad_enabled: bool,
    ) -> tuple[Tensor, ...]:
        state = projection.prepare(
            means,
            quats,
            scales,
            opacities,
            view,
            full,
            intrinsics,
            config,
            FrameBindings(launch_cache),
        )
        if grad_enabled and any(ctx.needs_input_grad[:6]):
            ctx.save_for_backward(
                means,
                quats,
                scales,
                opacities,
                view,
                full,
                intrinsics,
                state.centers,
                state.conics,
                state.radii,
                state.tile_counts,
                state.depths,
            )
            ctx.config = config
        ctx.mark_non_differentiable(state.radii, state.tile_counts, state.depths)
        ctx.set_materialize_grads(False)
        return state.centers, state.conics, state.radii, state.tile_counts, state.depths

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any, centers: Tensor | None, conics: Tensor | None, *unused
    ) -> tuple[Tensor | None, ...]:
        if centers is None and conics is None:
            return (None,) * 9
        values = ctx.saved_tensors
        state = projection.ProjectedGaussians(*values[7:])
        config = ctx.config
        gradients = projection.backward(
            *values[:7],
            config,
            state,
            (
                torch.zeros_like(state.centers)
                if centers is None
                else centers.contiguous()
            ),
            torch.zeros_like(state.conics) if conics is None else conics.contiguous(),
            tuple(ctx.needs_input_grad[:6]),
            FrameBindings(launch_cache),
        )
        return (*gradients, None, None, None)


class _Pixels(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        centers: Tensor,
        conics: Tensor,
        features: Tensor,
        radii: Tensor,
        ranges: Tensor,
        ids: Tensor,
        background: Tensor,
        config: ViewConfig,
        statistics: PreparedStatistics | None,
        grad_enabled: bool,
    ) -> Tensor:
        state = projection.ProjectedFootprint(centers, conics)
        output = render.render(
            state,
            dict(ranges=ranges, point_list=ids),
            features,
            background,
            config.width,
            config.height,
            FrameBindings(launch_cache),
            statistics,
        )
        if grad_enabled and any(ctx.needs_input_grad[:3]):
            ctx.save_for_backward(
                centers,
                conics,
                features,
                radii,
                ranges,
                ids,
                background,
                output.transmittance,
                output.contributors,
            )
            ctx.config = config
        ctx.set_materialize_grads(False)
        return output.features

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, gradient: Tensor | None) -> tuple[Tensor | None, ...]:
        if gradient is None:
            return (None,) * 10
        (
            centers,
            conics,
            features,
            radii,
            ranges,
            ids,
            background,
            final_t,
            contributors,
        ) = ctx.saved_tensors
        config = ctx.config
        gradients = render.backward(
            projection.ProjectedFootprint(centers, conics),
            dict(ranges=ranges, point_list=ids),
            features,
            background,
            final_t,
            contributors,
            gradient,
            config.width,
            config.height,
            FrameBindings(launch_cache),
        )
        return (
            *(
                value if ctx.needs_input_grad[i] else None
                for i, value in enumerate(gradients[:3])
            ),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def prepare_gaussians(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    camera: PinholeCamera,
    *,
    radius_clip: float = 0.0,
) -> GaussianGeometry:
    if (
        not isinstance(means, Tensor)
        or not means.is_cuda
        or means.ndim != 2
        or means.shape[1] != 3
    ):
        raise ValueError("means must be CUDA float32 (N,3)")
    n, device = len(means), means.device
    for name, value, shape in (
        ("means", means, (n, 3)),
        ("quats", quats, (n, 4)),
        ("scales", scales, (n, 3)),
        ("opacities", opacities, (n,)),
    ):
        check_tensor(name, value, device, shape)
    view, full, intrinsics, config = prepare_camera(
        camera, radius_clip, FrameBindings(launch_cache)
    )
    if view.device != device:
        raise ValueError("camera and Gaussians must use the same CUDA device")
    output = _Project.apply(
        means.contiguous(),
        normalize(quats, dim=-1).contiguous(),
        scales.contiguous(),
        opacities.contiguous(),
        view,
        full,
        intrinsics,
        config,
        torch.is_grad_enabled(),
    )
    centers, conics, radii, tile_counts, depths = output
    return GaussianGeometry(centers, conics, radii, tile_counts, depths, config)


def bin_and_sort(geometry: GaussianGeometry) -> GaussianTiles:
    bins = _bin_and_sort(
        geometry._state(),
        geometry._config.width,
        geometry._config.height,
        FrameBindings(launch_cache),
    )
    return GaussianTiles(bins["ranges"], bins["point_list"], geometry)


def rasterize_to_pixels(
    geometry: GaussianGeometry,
    tiles: GaussianTiles,
    features: Tensor,
    background: Tensor | None = None,
    *,
    statistics: Sequence[StatisticsRequest] = (),
) -> PixelResult:
    if tiles._geometry is not geometry:
        raise ValueError("tiles must have been built from this geometry")
    n, device = len(geometry.radii), geometry.radii.device
    check_tensor("features", features, device)
    if features.ndim != 2 or features.shape[0] != n or features.shape[1] <= 0:
        raise ValueError("features must have shape (N,D), D>0")
    channels = features.shape[1]
    if background is not None:
        check_tensor("background", background, device, (channels,))
        if torch.is_grad_enabled() and background.requires_grad:
            raise NotImplementedError("background gradients are not supported")
    attributes = torch.cat((features, features.new_ones((n, 1))), -1)
    bg = features.new_zeros(channels + 1)
    if background is not None:
        bg[:channels] = background
    config = geometry._config
    prepared, outputs = (
        prepare_statistics(
            statistics,
            batch_dims=(),
            batch_count=1,
            cameras=1,
            n=n,
            width=config.width,
            height=config.height,
            device=device,
        )
        if statistics
        else ([], {})
    )
    image = _Pixels.apply(
        geometry.centers,
        geometry.conics,
        attributes,
        geometry.radii,
        tiles.ranges,
        tiles.gaussian_ids,
        bg,
        config,
        prepared[0] if statistics else None,
        torch.is_grad_enabled(),
    ).permute(1, 2, 0)
    return PixelResult(
        image[..., :channels],
        image[..., -1:],
        {name: value[0] for name, value in outputs.items()},
    )


__all__ = [
    "prepare_gaussians",
    "bin_and_sort",
    "rasterize_to_pixels",
    "spherical_harmonics",
    "GaussianGeometry",
    "GaussianTiles",
    "PixelResult",
]
