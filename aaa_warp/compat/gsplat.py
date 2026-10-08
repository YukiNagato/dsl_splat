"""Typed gsplat API adapter; geometry and blending use the AAA core stages."""

import math
from numbers import Integral
from typing import Literal, cast

import torch
from torch import Tensor

from ..frame import FrameOptions, prepare_frame
from ..rasterizer import GaussianRasterizer
from ..settings import (
    CullingSettings,
    ExtendedSettings,
    GlobalSortOrder,
    SortMode,
    SortSettings,
)
from ..types import GaussianInputs
from .gsplat_cameras import Cameras, prepare_cameras
from .gsplat_inputs import (
    Inputs,
    ViewFeatures,
    RenderMode,
    normalize_inputs,
    spherical_harmonics,
)
from .gsplat_metadata import ExtendedMetadata, ViewFrame, build_metadata

_rasterizer = GaussianRasterizer()


def _default_settings(antialiased: bool) -> ExtendedSettings:
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


def _validate_options(
    width: int,
    height: int,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    eps2d: float,
    sh_degree: int | None,
    tile_size: int | None,
    render_mode: RenderMode,
    rasterize_mode: str,
    channel_chunk: int,
    camera_model: str,
    with_eval3d: bool | None,
    rolling_shutter: object | None,
    unsupported: dict[str, bool],
    unsupported_values: dict[str, object],
) -> None:
    for name, enabled in unsupported.items():
        if enabled:
            raise NotImplementedError(
                f"AAA/Warp gsplat adapter does not support {name}=True"
            )
    for name, value in unsupported_values.items():
        if value is not None:
            raise NotImplementedError(
                f"AAA/Warp gsplat adapter does not support {name}"
            )
    shutter = getattr(rolling_shutter, "name", rolling_shutter)
    if shutter not in (None, 0, "GLOBAL", "global"):
        raise NotImplementedError("only global shutter is supported")
    if camera_model != "pinhole":
        raise NotImplementedError("only pinhole cameras are supported")
    if with_eval3d is False:
        raise NotImplementedError(
            "this adapter always uses AAA eval_3D; omit with_eval3d or set it True"
        )
    if eps2d != 0.3:
        raise NotImplementedError("AAA's adaptive filter currently has fixed eps2d=0.3")
    if tile_size not in (None, 16):
        raise NotImplementedError("AAA's hierarchy requires tile_size=16")
    if render_mode not in ("RGB", "D", "ED", "RGB+D", "RGB+ED"):
        raise NotImplementedError(f"unsupported render_mode: {render_mode}")
    if rasterize_mode not in ("classic", "antialiased"):
        raise ValueError("rasterize_mode must be 'classic' or 'antialiased'")
    if any(
        isinstance(v, bool) or not isinstance(v, Integral) or v <= 0
        for v in (width, height, channel_chunk)
    ):
        raise ValueError("width, height and channel_chunk must be positive integers")
    if (
        not math.isfinite(near_plane)
        or near_plane <= 0
        or math.isnan(far_plane)
        or far_plane <= near_plane
    ):
        raise ValueError("require 0 < near_plane < far_plane")
    if not math.isfinite(radius_clip) or radius_clip < 0:
        raise ValueError("radius_clip must be finite and nonnegative")
    if sh_degree is not None and (
        isinstance(sh_degree, bool)
        or not isinstance(sh_degree, Integral)
        or sh_degree not in range(4)
    ):
        raise ValueError("sh_degree must be None or an integer in [0,3]")


def _prepare_frames(
    layout: Inputs,
    cameras: Cameras,
    render_mode: RenderMode,
    sh_degree: int | None,
    settings: ExtendedSettings,
    radius_clip: float,
    near_plane: float,
    far_plane: float,
) -> list[ViewFrame]:
    frames: list[ViewFrame] = []
    has_rgb, has_depth = "RGB" in render_mode, "D" in render_mode
    colors = cast(ViewFeatures, layout.colors)
    ones = torch.ones(layout.n, 1, device=layout.device, dtype=torch.float32)
    for batch in range(layout.batch_count):
        for camera in range(layout.cameras):
            view, means = cameras.views[batch, camera], layout.means[batch]
            positions = means if has_depth else means.detach()
            points = positions @ view[:3, :3].T + view[:3, 3]
            z = points[:, 2]
            opacity = torch.where(
                (z > near_plane) & (z < far_plane), layout.opacities[batch], 0
            )
            fx, fy, cx, cy = cameras.intrinsics[batch][camera]
            attributes: list[Tensor] = []
            if has_rgb:
                value = colors.at(batch, camera)
                if sh_degree is not None:
                    value = spherical_harmonics(
                        value, means - cameras.centers[batch, camera], sh_degree
                    )
                attributes.append(value)
            if has_depth:
                attributes.append(z[:, None])
            if layout.extras is not None:
                attributes.append(layout.extras.at(batch, camera))
            attributes.append(ones)
            features = torch.cat(attributes, dim=-1)
            background = torch.zeros(
                features.shape[-1], device=layout.device, dtype=torch.float32
            )
            if has_rgb and layout.backgrounds is not None:
                background[: colors.channels] = layout.backgrounds[batch, camera]
            config = cameras.config(batch, camera, background, settings)
            inputs: GaussianInputs = dict(
                means3D=means,
                rotations=layout.quats[batch],
                scales=layout.scales[batch],
                opacities=opacity,
                colors_precomp=features,
                filter3D=None if layout.filters is None else layout.filters[batch],
            )
            prepared = prepare_frame(
                inputs,
                config,
                _rasterizer._launch_cache,
                FrameOptions(principal_point=(cx, cy), radius_clip=radius_clip),
            )
            frames.append(
                ViewFrame(
                    batch,
                    camera,
                    config,
                    inputs,
                    prepared,
                    points,
                    cameras.intrinsics[batch][camera],
                )
            )
    return frames


def rasterization(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor | None,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 10000000000.0,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: int | None = None,
    packed: bool = True,
    tile_size: int | None = 16,
    backgrounds: Tensor | None = None,
    render_mode: RenderMode = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    distributed: bool = False,
    camera_model: str = "pinhole",
    segmented: bool = False,
    covars: Tensor | None = None,
    with_ut: bool = False,
    with_eval3d: bool | None = None,
    radial_coeffs: Tensor | None = None,
    tangential_coeffs: Tensor | None = None,
    thin_prism_coeffs: Tensor | None = None,
    ftheta_coeffs: object | None = None,
    rolling_shutter: object | None = None,
    viewmats_rs: Tensor | None = None,
    *,
    extra_signals: Tensor | None = None,
    extra_signals_sh_degree: int | None = None,
    aaa_settings: ExtendedSettings | None = None,
    filter3D: Tensor | None = None,
) -> tuple[Tensor, Tensor, ExtendedMetadata]:
    """Return (render_colors, render_alphas, meta) with gsplat tensor layouts.

    Supports arbitrary Gaussian batch dimensions, C pinhole cameras, shared or
    per-camera features/SH0–3, packed/dense metadata, radius/near/far culling,
    RGB/D/ED/RGB+D/RGB+ED, and first-order Gaussian/feature/alpha/depth gradients.
    Cameras are rendered sequentially on Torch's current CUDA stream.

    Inputs are activated scales/opacities and wxyz quaternions (normalized here).
    Output layouts are (...,C,H,W,D) and (...,C,H,W,1). Depth uses Gaussian
    center view-space Z; ED divides accumulated depth by alpha (floor 1e-10).
    Alpha is blended as a constant-one feature, so it remains differentiable.

    Always uses AAA eval_3D; explicitly requesting with_eval3d=False is rejected.
    eps2d is fixed at 0.3; rasterize_mode controls AAA opacity compensation.
    channel_chunk is an accepted performance hint: all channels are fused.
    Unsupported camera/background gradients, sparse gradients,
    distortion/UT/rolling shutter, covariance inputs and distributed rendering
    raise explicit errors.
    Projected-mean gradients measure rigid pixel-space translation of the AAA
    footprint. absgrad=True also records the sum of absolute per-pixel gradients
    at meta["means2d"].absgrad, for gsplat DefaultStrategy/AbsGS densification.
    """
    _validate_options(
        width,
        height,
        near_plane,
        far_plane,
        radius_clip,
        eps2d,
        sh_degree,
        tile_size,
        render_mode,
        rasterize_mode,
        channel_chunk,
        camera_model,
        with_eval3d,
        rolling_shutter,
        dict(
            sparse_grad=sparse_grad,
            distributed=distributed,
            segmented=segmented,
            with_ut=with_ut,
        ),
        dict(
            covars=covars,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            ftheta_coeffs=ftheta_coeffs,
            viewmats_rs=viewmats_rs,
            extra_signals_sh_degree=extra_signals_sh_degree,
        ),
    )
    layout = normalize_inputs(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        backgrounds,
        extra_signals,
        filter3D,
        render_mode,
        sh_degree,
    )
    cameras = prepare_cameras(layout, width, height, near_plane, far_plane)
    if aaa_settings is not None and not isinstance(aaa_settings, ExtendedSettings):
        raise ValueError("aaa_settings must use eval_3D=True and HIER sorting")
    settings = (
        _default_settings(rasterize_mode == "antialiased")
        if aaa_settings is None
        else aaa_settings.copy()
    )
    if (
        not isinstance(settings, ExtendedSettings)
        or not settings.eval_3D
        or int(settings.sort_settings.sort_mode) != 3
    ):
        raise ValueError("aaa_settings must use eval_3D=True and HIER sorting")
    frames = _prepare_frames(
        layout,
        cameras,
        render_mode,
        sh_degree,
        settings,
        radius_clip,
        near_plane,
        far_plane,
    )
    meta, screen, options = build_metadata(
        frames, layout, width, height, packed=packed, absgrad=absgrad
    )
    channels = cast(ViewFeatures, layout.colors).channels if "RGB" in render_mode else 0
    extra_channels = 0 if layout.extras is None else layout.extras.channels
    has_depth = "D" in render_mode
    outputs: list[Tensor] = []
    alphas: list[Tensor] = []
    extras: list[Tensor] = []
    for frame, option in zip(frames, options):
        actual = _rasterizer(
            **frame.inputs,
            means2D=None if screen is None else screen[frame.batch, frame.camera],
            raster_settings=frame.config,
            return_aux=True,
            _stage_options=option,
        )
        image = actual["color"].permute(1, 2, 0)
        if layout.n == 0:
            image = image + frame.config.bg
        alpha = image[..., -1:]
        rendered = image[..., : channels + int(has_depth)]
        if render_mode in ("ED", "RGB+ED"):
            rendered = torch.cat(
                (rendered[..., :-1], rendered[..., -1:] / alpha.clamp_min(1e-10)),
                dim=-1,
            )
        outputs.append(rendered)
        alphas.append(alpha)
        if layout.extras is not None:
            extras.append(image[..., channels + int(has_depth) : -1])
    image_shape = layout.batch_dims + (layout.cameras, height, width)
    render_colors = torch.stack(outputs).reshape(
        image_shape + (channels + int(has_depth),)
    )
    render_alphas = torch.stack(alphas).reshape(image_shape + (1,))
    if extras:
        meta["render_extra_signals"] = torch.stack(extras).reshape(
            image_shape + (extra_channels,)
        )
    return render_colors, render_alphas, meta


__all__ = ["rasterization"]
