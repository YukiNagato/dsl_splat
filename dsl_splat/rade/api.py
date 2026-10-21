"""RaDe-GS API with OpenCV cameras and channel-last geometry outputs."""

from dataclasses import dataclass, field
import math
from numbers import Integral
from typing import Any, Sequence

import torch
from torch import Tensor

from .autograd import Rasterize, launch_cache
from .types import ViewConfig
from ..splatting.cameras import prepare_cameras, prepare_cameras_with_centers
from .projection import prepare
from ..splatting.metadata import prepare_metadata
from .frame import PreparedView, ViewInputs
from ..cameras import validate_camera_options
from ..inputs import batch_item, normalize_inputs
from ..primitives.spherical_harmonics import spherical_harmonics
from ..runtime.dispatch import FrameBindings
from ..statistics import StatisticsRequest
from ..training import prepare_training_metadata


@dataclass(slots=True)
class RasterizationResult:
    """Channel-last views of frame-owned images and training metadata."""

    features: Tensor
    alpha: Tensor
    depth: Tensor
    median_depth: Tensor
    normals: Tensor
    meta: dict[str, Any]
    statistics: dict[str, Tensor] = field(default_factory=dict)

    @property
    def radii(self) -> Tensor:
        return self.meta["radii"]

    @property
    def means2d(self) -> Tensor:
        return self.meta["means2d"]

    @property
    def extra_features(self) -> Tensor | None:
        return self.meta.get("render_extra_signals")


def rasterize(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    *,
    features: Tensor | None = None,
    sh_coeffs: Tensor | None = None,
    sh_degree: int | None = None,
    backgrounds: Tensor | None = None,
    extra_features: Tensor | None = None,
    near_plane: float = 0.2,
    far_plane: float = math.inf,
    radius_clip: float = 0.0,
    kernel_size: float = 0.0,
    packed: bool = False,
    absgrad: bool = False,
    statistics: Sequence[StatisticsRequest] = (),
) -> RasterizationResult:
    """Render RaDe-GS features, alpha, expected/median Z depth and normals.

    Inputs use OpenCV world-to-camera matrices and channel-last images. Scales
    and opacities are activated and quaternions are normalized. Depth is the
    original ray-plane approximation, normals are unit camera-space vectors.
    kernel_size defaults to RaDe-GS's 0.0. Supply already 3D-filtered scales and
    opacities, or use apply_3d_filter before this call, as in the official frontend.
    Intrinsics/background gradients and higher derivatives are unsupported.
    """
    if not math.isfinite(kernel_size) or kernel_size < 0:
        raise ValueError("kernel_size must be finite and nonnegative")
    if statistics:
        raise NotImplementedError("RaDe-GS contribution statistics are not implemented")
    validate_camera_options(width, height, near_plane, far_plane, radius_clip)
    if (features is None) == (sh_coeffs is None):
        raise ValueError("provide exactly one of features or sh_coeffs")
    if sh_coeffs is not None:
        if not isinstance(sh_coeffs, Tensor) or sh_coeffs.ndim < 3:
            raise ValueError("sh_coeffs must have shape (...,N,K,3) or (...,C,N,K,3)")
        if sh_degree is None:
            sh_degree = min(3, math.isqrt(sh_coeffs.shape[-2]) - 1)
    elif sh_degree is not None:
        raise ValueError("sh_degree requires sh_coeffs")
    if sh_degree is not None and (
        isinstance(sh_degree, bool)
        or not isinstance(sh_degree, Integral)
        or sh_degree not in range(4)
    ):
        raise ValueError("sh_degree must be an integer in [0,3]")
    layout = normalize_inputs(
        means,
        quats,
        scales,
        opacities,
        features if sh_coeffs is None else sh_coeffs,
        viewmats,
        Ks,
        backgrounds,
        extra_features,
        None,
        sh_degree,
    )
    camera_args = (
        layout.viewmats,
        layout.Ks,
        width,
        height,
        FrameBindings(launch_cache),
    )
    if sh_degree is not None and sh_degree > 0:
        full, centers = prepare_cameras_with_centers(*camera_args)
    else:
        full, centers = prepare_cameras(*camera_args), None
    frames: list[PreparedView] = []
    dense: dict[str, list[Tensor]] = {
        name: []
        for name in ("radii", "means2d", "opacities", "tiles_per_gauss", "conics")
    }
    for b in range(layout.batch_count):
        batch_means = layout.means[b].contiguous()
        batch_quats = layout.quats[b].contiguous()
        batch_scales = layout.scales[b].contiguous()
        batch_opacities = layout.opacities[b].contiguous()
        for c in range(layout.cameras):
            config = ViewConfig(
                width, height, near_plane, far_plane, radius_clip, kernel_size
            )
            bindings = FrameBindings(launch_cache)
            view = batch_item(batch_item(layout.viewmats, b), c).contiguous()
            projection = batch_item(batch_item(full, b), c)
            intrinsics = batch_item(batch_item(layout.Ks, b), c).contiguous()
            state = prepare(
                batch_means,
                batch_quats,
                batch_scales,
                batch_opacities,
                view,
                projection,
                intrinsics,
                config,
                bindings,
            )
            values = layout.colors.at(b, c)
            if sh_degree is not None:
                values = spherical_harmonics(
                    values,
                    (
                        batch_means
                        if centers is None
                        else batch_means - batch_item(batch_item(centers, b), c)
                    ),
                    sh_degree,
                    radii=state.radii,
                    bindings=bindings,
                )
            inputs = ViewInputs(
                means=batch_means,
                quats=batch_quats,
                scales=batch_scales,
                opacities=batch_opacities,
                features=values,
                extra_features=(
                    None if layout.extras is None else layout.extras.at(b, c)
                ),
                view=view,
                projection=projection,
                intrinsics=intrinsics,
                background=(
                    None
                    if layout.backgrounds is None
                    else batch_item(batch_item(layout.backgrounds, b), c)
                ),
            )
            frames.append(PreparedView(b, c, inputs, state, config, bindings))
            for name, value in prepare_metadata(state, bindings).items():
                dense[name].append(value)
    geometry = {
        name: (rows[0] if len(rows) == 1 else torch.stack(rows)).reshape(
            (layout.batch_count, layout.cameras, layout.n) + tuple(rows[0].shape[1:])
        )
        for name, rows in dense.items()
    }
    training = prepare_training_metadata(
        geometry, layout, packed=packed, absgrad=absgrad
    )
    meta = training.values
    meta.update(
        width=width,
        height=height,
        tile_size=16,
        tile_width=(width + 15) // 16,
        tile_height=(height + 15) // 16,
        n_cameras=layout.cameras,
        packed=packed,
        backend="rade-warp",
        with_eval3d=False,
        supports_means2d_grad=True,
        screen_gradient_mode="projected_mean",
    )
    statistics_outputs: dict[str, Tensor] = {}
    images = []
    for i, frame in enumerate(frames):
        screen = (
            None
            if training.screen is None
            else batch_item(batch_item(training.screen, frame.batch), frame.camera)
        )
        inputs = frame.inputs
        image = Rasterize.apply(
            inputs.means,
            inputs.quats,
            inputs.scales,
            inputs.opacities,
            inputs.features,
            inputs.extra_features,
            inputs.view,
            inputs.projection,
            inputs.intrinsics,
            screen,
            inputs.background,
            frame.geometry,
            frame.config,
            frame.bindings,
            training.absolute_targets[i],
            torch.is_grad_enabled(),
        )
        images.append(image)
    shape = layout.batch_dims + (layout.cameras, height, width)
    # Each output has its own autograd edge. Splitting a single packed image
    # outside the Function would materialize a full packed gradient for every
    # output, followed by multiple full-image additions in backward.
    # Stack also permits ordinary in-place image processing on API results.
    features_image, alpha, depth, median, normals = (
        torch.stack(values).reshape(shape + (values[0].shape[-1],))
        for values in zip(*images)
    )
    channels = layout.colors.channels
    main_features = features_image
    if layout.extras is not None:
        meta["render_extra_signals"] = features_image[..., channels:]
        main_features = features_image[..., :channels]
    meta.update(depth=depth, median_depth=median, normals=normals)
    return RasterizationResult(
        main_features,
        alpha,
        depth,
        median,
        normals,
        meta,
        statistics_outputs,
    )
