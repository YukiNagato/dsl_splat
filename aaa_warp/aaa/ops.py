"""Composable, differentiable AAA stages for one OpenCV camera.

Geometry and sorting can be reused with different feature tensors. The complete
high-level path retains its single autograd Function; these Functions are used
only when callers explicitly compose stages. Both paths use the same kernels.
"""

from dataclasses import dataclass, field
from typing import Any, Sequence, cast
import math

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch.nn.functional import normalize

from .backward_camera import backward_camera
from .backward_preprocess import backward_preprocess
from .backward_render import backward_render
from .binning import bin_and_sort as _bin_and_sort
from ..cameras import PinholeCamera
from .cameras import camera_config
from .config import FrameConfig, default_settings, validate_settings
from ..runtime.dispatch import FrameBindings
from .frame import FrameOptions
from ..inputs import check_tensor, spherical_harmonics
from .pipeline import RenderContext, default_context
from .settings import ExtendedSettings, copy_settings
from .types import GaussianInputs, PreprocessedGaussians, TileBins
from .render import render_hierarchical_3d, native_renderer, warp_renderer
from ..statistics import PreparedStatistics, StatisticsRequest, prepare_statistics

__all__ = [
    "prepare_gaussians",
    "bin_and_sort",
    "rasterize_to_pixels",
    "spherical_harmonics",
    "GaussianGeometry",
    "GaussianTiles",
    "PixelResult",
]


@dataclass(slots=True)
class GaussianGeometry:
    """AAA ray transforms and bounds; rejected rows are identified by radii=0.

    ray_transforms and opacities are differentiable. Bounds, visibility and tile
    membership are discrete, fixed during backward. Bounding centers are not
    gsplat's projected-center metadata. No feature tensor is retained here.
    """

    ray_transforms: Tensor
    opacities: Tensor
    radii: Tensor
    bounding_centers: Tensor
    half_extents: Tensor
    tiles_touched: Tensor
    valid: Tensor
    depths: Tensor
    _config: FrameConfig = field(repr=False)
    _context: RenderContext = field(repr=False)

    def _state(self, features: Tensor) -> PreprocessedGaussians:
        return dict(
            gauss2screen=self.ray_transforms,
            opacity=self.opacities,
            radii=self.radii,
            means2D=self.bounding_centers,
            rects2D=self.half_extents,
            tiles_touched=self.tiles_touched,
            valid=self.valid,
            depths=self.depths,
            rgb=features,
        )


@dataclass(slots=True)
class GaussianTiles:
    ranges: Tensor
    gaussian_ids: Tensor
    _geometry: GaussianGeometry = field(repr=False)


@dataclass(slots=True)
class PixelResult:
    """Owned (H,W,D) features and differentiable (H,W,1) alpha."""

    features: Tensor
    alpha: Tensor
    statistics: dict[str, Tensor] = field(default_factory=dict)


class _PrepareGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        opacities: Tensor,
        filters: Tensor | None,
        view: Tensor,
        projection: Tensor,
        center: Tensor,
        config: FrameConfig,
        context: RenderContext,
        radius_clip: float,
        grad_enabled: bool,
    ) -> tuple[Tensor, ...]:
        inputs: GaussianInputs = dict(
            means3D=means,
            rotations=quats,
            scales=scales,
            opacities=opacities,
            filter3D=filters,
            # Preprocessing does not read precomputed features. This view
            # satisfies its existing shape contract without allocating N values.
            colors_precomp=opacities.reshape(-1, 1),
        )
        state = context.prepare(
            inputs, config, FrameOptions(radius_clip=radius_clip)
        ).state
        output = tuple(
            state[name]
            for name in (
                "gauss2screen",
                "opacity",
                "radii",
                "means2D",
                "rects2D",
                "tiles_touched",
                "valid",
                "depths",
            )
        )
        ctx.mark_non_differentiable(*output[2:])
        ctx.set_materialize_grads(False)
        if grad_enabled and any(ctx.needs_input_grad[:8]):
            ctx.save_for_backward(
                means,
                quats,
                scales,
                opacities,
                filters,
                view,
                projection,
                center,
                state["radii"],
                state["gauss2screen"],
                state["opacity"],
            )
            ctx.config = config._replace(settings=copy_settings(config.settings))
            ctx.context = context
        return output

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        grad_transform: Tensor | None,
        grad_opacity: Tensor | None,
        *unused: Tensor | None,
    ) -> tuple[Tensor | None, ...]:
        if grad_transform is None and grad_opacity is None:
            return (None,) * 12
        (
            means,
            quats,
            scales,
            opacity,
            filters,
            view,
            projection,
            center,
            radii,
            transforms,
            effective_opacity,
        ) = ctx.saved_tensors
        config = ctx.config._replace(
            viewmatrix=view, projmatrix=projection, campos=center
        )
        inputs = dict(
            means3D=means,
            rotations=quats,
            scales=scales,
            opacities=opacity,
            filter3D=filters,
        )
        state = dict(radii=radii, gauss2screen=transforms, opacity=effective_opacity)
        gradients = dict(
            gauss2screen=(
                torch.zeros_like(transforms)
                if grad_transform is None
                else grad_transform.transpose(-1, -2).contiguous()
            ),
            opacity=(
                torch.zeros_like(effective_opacity)
                if grad_opacity is None
                else grad_opacity.contiguous()
            ),
            rgb=means.new_empty((len(means), 0)),
        )
        bindings = FrameBindings(ctx.context.launch_cache)
        model_gradients = (
            backward_preprocess(inputs, state, gradients, config, _bindings=bindings)
            if any(ctx.needs_input_grad[:4])
            else {}
        )
        camera_gradients = (
            backward_camera(inputs, state, gradients, config, _bindings=bindings)
            if any(ctx.needs_input_grad[5:8])
            else {}
        )
        result: list[Tensor | None] = [
            model_gradients[name] if ctx.needs_input_grad[i] else None
            for i, name in enumerate(("means3D", "rotations", "scales", "opacities"))
        ]
        result.append(None)  # filter3d is constant, following AAA.
        result.extend(
            camera_gradients[name] if ctx.needs_input_grad[5 + i] else None
            for i, name in enumerate(("viewmatrix", "projmatrix", "campos"))
        )
        return (*result, None, None, None, None)


class _RasterizePixels(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        transforms: Tensor,
        opacity: Tensor,
        features: Tensor,
        radii: Tensor,
        ranges: Tensor,
        ids: Tensor,
        background: Tensor,
        config: FrameConfig,
        context: RenderContext,
        grad_enabled: bool,
        statistics: PreparedStatistics | None,
    ) -> Tensor:
        state = cast(
            PreprocessedGaussians,
            dict(gauss2screen=transforms, opacity=opacity, rgb=features, radii=radii),
        )
        bins: TileBins = dict(ranges=ranges, point_list=ids)
        bindings = FrameBindings(context.launch_cache)
        if statistics is not None:
            if context.renderer not in (
                render_hierarchical_3d,
                native_renderer,
                warp_renderer,
            ):
                raise ValueError("contribution statistics require an AAA Warp renderer")
            output = warp_renderer(
                state, bins, config, _bindings=bindings, _statistics=statistics
            )
        else:
            output = context.renderer(state, bins, config, _bindings=bindings)
        if grad_enabled and any(ctx.needs_input_grad[:3]):
            ctx.save_for_backward(
                transforms,
                opacity,
                features,
                radii,
                ranges,
                ids,
                background,
                output["color"],
                output["final_T"],
                output["contributors"],
            )
            ctx.config = config._replace(settings=copy_settings(config.settings))
            ctx.context = context
        return output["color"]

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_color: Tensor) -> tuple[Tensor | None, ...]:
        (
            transforms,
            opacity,
            features,
            radii,
            ranges,
            ids,
            background,
            color,
            final_t,
            contributors,
        ) = ctx.saved_tensors
        gradients = backward_render(
            dict(gauss2screen=transforms, opacity=opacity, rgb=features, radii=radii),
            dict(ranges=ranges, point_list=ids),
            dict(color=color, final_T=final_t, contributors=contributors),
            grad_color,
            ctx.config._replace(bg=background),
            _bindings=FrameBindings(ctx.context.launch_cache),
        )
        return (
            (
                gradients["gauss2screen"].transpose(-1, -2)
                if ctx.needs_input_grad[0]
                else None
            ),
            gradients["opacity"] if ctx.needs_input_grad[1] else None,
            gradients["rgb"] if ctx.needs_input_grad[2] else None,
            None,
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
    filter3d: Tensor | None = None,
    radius_clip: float = 0.0,
    settings: ExtendedSettings | None = None,
) -> GaussianGeometry:
    """Prepare one view without features; supports Gaussian and pose gradients.

    Inputs are CUDA float32, shapes (N,3), (N,4), (N,3), (N,). Quaternion
    normalization is differentiable. Reuse requires unchanged camera, geometry,
    opacity, filtering and settings. Culling/bounds are not differentiated.
    """
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
    if filter3d is not None:
        check_tensor("filter3d", filter3d, device)
        if filter3d.shape not in ((n,), (n, 1)):
            raise ValueError("filter3d must have shape (N,) or (N,1)")
    if camera.viewmat.device != device:
        raise ValueError("camera and Gaussians must be on the same device")
    if radius_clip < 0 or not math.isfinite(radius_clip):
        raise ValueError("radius_clip must be finite and nonnegative")
    if settings is not None:
        validate_settings(settings)
    options = default_settings() if settings is None else settings.copy()
    config = camera_config(camera, means.new_zeros(1), options)
    values = _PrepareGaussians.apply(
        means,
        normalize(quats, dim=-1),
        scales,
        opacities,
        filter3d,
        config.viewmatrix,
        config.projmatrix,
        config.campos,
        config,
        default_context,
        radius_clip,
        torch.is_grad_enabled(),
    )
    return GaussianGeometry(
        ray_transforms=values[0],
        opacities=values[1],
        radii=values[2],
        bounding_centers=values[3],
        half_extents=values[4],
        tiles_touched=values[5],
        valid=values[6],
        depths=values[7],
        _config=config,
        _context=default_context,
    )


@torch.no_grad()
def bin_and_sort(geometry: GaussianGeometry) -> GaussianTiles:
    """Build AAA tile intersections and keys using its configured depth order."""
    bins = _bin_and_sort(
        geometry._state(geometry.opacities.reshape(-1, 1)),
        geometry._config,
        diagnostics=False,
        _bindings=FrameBindings(geometry._context.launch_cache),
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
    """Blend (N,D) attributes and alpha, reusing geometry and sorted tiles.

    Features can be signed and have any positive channel count. Background is
    constant; its gradient is unsupported. Rendering follows AAA HIER 64/8/4.
    """
    if tiles._geometry is not geometry:
        raise ValueError("tiles must have been built from this geometry")
    device, n = geometry.radii.device, len(geometry.radii)
    check_tensor("features", features, device)
    if features.ndim != 2 or features.shape[0] != n or features.shape[1] <= 0:
        raise ValueError("features must have shape (N,D), D > 0")
    channels = features.shape[1]
    if background is not None:
        check_tensor("background", background, device, (channels,))
        if torch.is_grad_enabled() and background.requires_grad:
            raise NotImplementedError("background gradients are not supported")
    attributes = torch.cat((features, features.new_ones(n, 1)), -1)
    bg = features.new_zeros(channels + 1)
    if background is not None:
        bg[:channels] = background
    config = geometry._config._replace(bg=bg)
    prepared, statistics_outputs = (
        prepare_statistics(
            statistics,
            batch_dims=(),
            batch_count=1,
            cameras=1,
            n=n,
            width=config.image_width,
            height=config.image_height,
            device=device,
        )
        if statistics
        else ([], {})
    )
    color = _RasterizePixels.apply(
        geometry.ray_transforms,
        geometry.opacities,
        attributes,
        geometry.radii,
        tiles.ranges,
        tiles.gaussian_ids,
        bg,
        config,
        geometry._context,
        torch.is_grad_enabled(),
        prepared[0] if statistics else None,
    ).permute(1, 2, 0)
    return PixelResult(
        color[..., :channels],
        color[..., -1:],
        {name: value[0] for name, value in statistics_outputs.items()},
    )
