"""Differentiable single-view projection and rendering for RaDe-GS.

The complete API keeps its fused Function. These independent Functions serve
callers who explicitly reuse geometry or compose rendering stages.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch.nn.functional import normalize

from . import projection, render
from .autograd import launch_cache
from ..splatting.binning import bin_and_sort as _bin_and_sort
from .types import ViewConfig
from ..splatting.cameras import prepare_camera
from ..cameras import PinholeCamera
from ..inputs import check_tensor, spherical_harmonics
from ..runtime.dispatch import FrameBindings
from ..statistics import StatisticsRequest


@dataclass(slots=True)
class GaussianGeometry:
    """Only rows with radii>0 define centers and conics; bounds are discrete."""

    centers: Tensor
    conics: Tensor
    radii: Tensor
    tile_counts: Tensor
    depths: Tensor
    ray_planes: Tensor
    normals: Tensor
    _intrinsics: Tensor = field(repr=False)
    _config: ViewConfig = field(repr=False)

    def _state(self) -> projection.ProjectedGaussians:
        return projection.ProjectedGaussians(
            self.centers,
            self.conics,
            self.radii,
            self.tile_counts,
            self.depths,
            self.ray_planes,
            self.normals,
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
    depth: Tensor
    median_depth: Tensor
    normals: Tensor
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
                state.ray_planes,
                state.normals,
            )
            ctx.config = config
        ctx.mark_non_differentiable(state.radii, state.tile_counts, state.depths)
        ctx.set_materialize_grads(False)
        return (
            state.centers,
            state.conics,
            state.radii,
            state.tile_counts,
            state.depths,
            state.ray_planes,
            state.normals,
        )

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        centers: Tensor | None,
        conics: Tensor | None,
        unused_radii: Tensor | None,
        unused_tiles: Tensor | None,
        unused_depths: Tensor | None,
        planes: Tensor | None,
        normals: Tensor | None,
    ) -> tuple[Tensor | None, ...]:
        if all(value is None for value in (centers, conics, planes, normals)):
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
            (
                torch.zeros_like(state.ray_planes)
                if planes is None
                else planes.contiguous()
            ),
            (
                torch.zeros_like(state.normals)
                if normals is None
                else normals.contiguous()
            ),
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
        planes: Tensor,
        normals: Tensor,
        features: Tensor,
        geometry: GaussianGeometry,
        tiles: GaussianTiles,
        background: Tensor,
        grad_enabled: bool,
    ) -> tuple[Tensor, ...]:
        state = geometry._state()
        config = geometry._config
        output = render.render(
            state,
            dict(ranges=tiles.ranges, point_list=tiles.gaussian_ids),
            features,
            background,
            geometry._intrinsics,
            config.width,
            config.height,
            FrameBindings(launch_cache),
        )
        if grad_enabled and any(ctx.needs_input_grad[:5]):
            ctx.save_for_backward(
                centers,
                conics,
                planes,
                normals,
                features,
                background,
                output.features,
                output.auxiliary,
                output.contributors,
            )
            ctx.geometry, ctx.tiles = geometry, tiles
        ctx.set_materialize_grads(False)
        return output.images()

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, *gradients: Tensor | None) -> tuple[Tensor | None, ...]:
        if all(g is None for g in gradients):
            return (None,) * 9
        (
            centers,
            conics,
            planes,
            normals,
            features,
            background,
            image,
            aux,
            contributors,
        ) = ctx.saved_tensors
        geometry, tiles = ctx.geometry, ctx.tiles
        config = geometry._config
        gc, gq, gf, gp, gn, _ = render.backward(
            geometry._state(),
            dict(ranges=tiles.ranges, point_list=tiles.gaussian_ids),
            features,
            background,
            geometry._intrinsics,
            render.RenderedPixels(image, aux, contributors),
            gradients,
            config.width,
            config.height,
            FrameBindings(launch_cache),
        )
        return gc, gq, gp, gn, gf, None, None, None, None


def prepare_gaussians(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    camera: PinholeCamera,
    *,
    radius_clip: float = 0.0,
    kernel_size: float = 0.0,
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
    if not math.isfinite(kernel_size) or kernel_size < 0:
        raise ValueError("kernel_size must be finite and nonnegative")
    config = ViewConfig(
        config.width,
        config.height,
        config.near,
        config.far,
        config.radius_clip,
        kernel_size,
    )
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
    centers, conics, radii, tile_counts, depths, planes, normals = output
    return GaussianGeometry(
        centers, conics, radii, tile_counts, depths, planes, normals, intrinsics, config
    )


def bin_and_sort(geometry: GaussianGeometry) -> GaussianTiles:
    bins = _bin_and_sort(
        geometry.centers,
        geometry.radii,
        geometry.tile_counts,
        geometry.depths,
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
    if statistics:
        raise NotImplementedError("RaDe-GS contribution statistics are not implemented")
    bg = features.new_zeros(channels) if background is None else background
    images = _Pixels.apply(
        geometry.centers,
        geometry.conics,
        geometry.ray_planes,
        geometry.normals,
        features,
        geometry,
        tiles,
        bg,
        torch.is_grad_enabled(),
    )
    return PixelResult(*images)


__all__ = [
    "prepare_gaussians",
    "bin_and_sort",
    "rasterize_to_pixels",
    "spherical_harmonics",
    "GaussianGeometry",
    "GaussianTiles",
    "PixelResult",
]
