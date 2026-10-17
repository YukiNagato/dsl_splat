"""gsplat-signature adapter for original 3DGS; no dependency on AAA."""

import math
from numbers import Integral
from typing import Any, Literal, Sequence

from torch import Tensor

from .api import rasterize
from ..inputs import RenderMode
from ..statistics import StatisticsRequest


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
                f"3DGS/Warp gsplat adapter does not support {name}=True"
            )
    for name, value in unsupported_values.items():
        if value is not None:
            raise NotImplementedError(
                f"3DGS/Warp gsplat adapter does not support {name}"
            )
    shutter = getattr(rolling_shutter, "name", rolling_shutter)
    if shutter not in (None, 0, "GLOBAL", "global"):
        raise NotImplementedError("only global shutter is supported")
    if camera_model != "pinhole":
        raise NotImplementedError("only pinhole cameras are supported")
    if with_eval3d is True:
        raise NotImplementedError(
            "original 3DGS uses projected 2D conics; with_eval3d=True is unsupported"
        )
    if eps2d != 0.3:
        raise NotImplementedError("original 3DGS currently has fixed eps2d=0.3")
    if tile_size not in (None, 16):
        raise NotImplementedError("original 3DGS currently requires tile_size=16")
    if render_mode != "RGB":
        raise NotImplementedError(
            "3DGS/Warp supports render_mode='RGB' for RGB or arbitrary features; "
            "depth rendering is not implemented"
        )
    if rasterize_mode != "classic":
        raise NotImplementedError(
            "original 3DGS supports rasterize_mode='classic' only"
        )
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
    statistics: Sequence[StatisticsRequest] = (),
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Return (features, alpha, metadata) using the original-3DGS Warp pipeline.

    Supports pinhole cameras, arbitrary feature channels, SH0-3, pose gradients
    and dense/packed training metadata. Packed changes metadata layout only.
    means2d retains CUDA's integer-center raster coordinates; its values are
    not shifted to gsplat's half-integer-center coordinate convention.
    near/far explicitly clip Gaussian centers; the native method entry defaults
    near to CUDA's 0.2, while this adapter retains gsplat's 0.01 signature default.
    Only RGB/features and classic covariance dilation are supported. Unsupported
    geometry, output modes and execution options raise explicit errors.
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
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        packed=packed,
        absgrad=absgrad,
        statistics=statistics,
    )
    if statistics:
        result.meta["statistics"] = result.statistics
    return result.features, result.alpha, result.meta


__all__ = ["rasterization"]
