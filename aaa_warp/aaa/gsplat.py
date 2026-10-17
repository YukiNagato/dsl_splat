"""Typed gsplat API adapter; geometry and blending use the AAA core stages."""

import math
from numbers import Integral
from typing import Literal, Sequence

from torch import Tensor

from .config import default_settings
from .api import rasterize
from ..inputs import RenderMode
from .metadata import ExtendedMetadata
from .settings import ExtendedSettings
from ..statistics import StatisticsRequest


def _default_settings(antialiased: bool) -> ExtendedSettings:
    return default_settings(antialiased=antialiased)


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
    if render_mode != "RGB":
        raise NotImplementedError(
            "AAA/Warp supports render_mode='RGB' for RGB or arbitrary features; "
            "depth rendering is not implemented"
        )
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
    statistics: Sequence[StatisticsRequest] = (),
) -> tuple[Tensor, Tensor, ExtendedMetadata]:
    """Return (render_colors, render_alphas, meta) with gsplat tensor layouts.

    Supports arbitrary Gaussian batch dimensions, C pinhole cameras, shared or
    per-camera features/SH0–3, packed/dense metadata, radius culling,
    RGB/arbitrary features, and first-order Gaussian/feature/alpha/pose gradients.
    Cameras are rendered sequentially on Torch's current CUDA stream.

    Inputs are activated scales/opacities and wxyz quaternions (normalized here).
    Output layouts are (...,C,H,W,D) and (...,C,H,W,1).
    Alpha is blended as a constant-one feature, so it remains differentiable.
    Only render_mode='RGB' is supported; depth modes raise NotImplementedError.
    near_plane/far_plane use AAA's frontend projection, mapping them to depth 0/1.
    Gaussian culling follows AAA's 3D frustum-contribution test, with no
    additional center-Z clipping; its (-1,1) test admits depths down to ~near/2.

    Always uses AAA eval_3D; explicitly requesting with_eval3d=False is rejected.
    eps2d is fixed at 0.3; rasterize_mode controls AAA opacity compensation.
    channel_chunk is an accepted performance hint: all channels are fused.
    viewmats supports gradients through the AAA transform, adaptive filter,
    opacity compensation and SH view direction. Intrinsic/background gradients,
    sparse gradients,
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
    result = rasterize(
        means,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        width,
        height,
        features=colors if sh_degree is None else None,
        sh_coeffs=colors if sh_degree is not None else None,
        sh_degree=sh_degree,
        backgrounds=backgrounds,
        extra_features=extra_signals,
        filter3d=filter3D,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        packed=packed,
        absgrad=absgrad,
        settings=(
            _default_settings(rasterize_mode == "antialiased")
            if aaa_settings is None
            else aaa_settings
        ),
        statistics=statistics,
    )
    if statistics:
        result.meta["statistics"] = result.statistics
    return result.features, result.alpha, result.meta


__all__ = ["rasterization"]
