"""Ordinary Python rasterizer with dynamic frames and analytic 3D backward."""

import torch

from .dispatch import LaunchCache
from .frame import FrameOptions
from .settings import GaussianRasterizationSettings
from .types import GaussianInputs, Renderer, RasterizationOutput
from .render import render_hierarchical_3d, native_renderer


class GaussianRasterizer(torch.nn.Module):
    """Render 3D HIER 64/8/4 frames without address or shape binding.

    Returns ``(color, radii)`` like the reference forward. Every call accepts
    new input tensors. Pass ``raster_settings=`` per frame to change camera,
    resolution or supported features; otherwise use the constructor settings.
    Outputs own their storage. No CUDA extension is imported or built.
    ``features`` or ``colors_precomp`` accepts (N,C) attributes and returns
    (C,H,W). ``channels=C`` optionally fixes the compiled channel count;
    otherwise each frame infers it. Kernels are cached by C and culling mode.
    The default selects the native baseline for RGB and Python/Warp for other
    channel counts. ``renderer=`` can explicitly select Python/Warp for all C.
    An empty Gaussian input returns a black image, matching the reference
    Python forward; a nonempty, fully culled scene returns the background.

    First-order backward follows the reference CUDA's 3D HIER conventions for
    means, scale, rotation, opacity and features/SH. Camera/background gradients
    are not supported; filter3D is constant, matching the reference API.
    """

    def __init__(
        self,
        raster_settings: GaussianRasterizationSettings | None = None,
        *,
        renderer: Renderer | None = None,
        channels: int | None = None,
    ) -> None:
        super().__init__()
        if channels is not None and (
            isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0
        ):
            raise ValueError("channels must be a positive integer or None")
        self.channels = channels
        self.raster_settings = raster_settings
        self._renderer = render_hierarchical_3d if renderer is None else renderer
        self._launch_cache = LaunchCache()

    def forward(
        self,
        means3D: torch.Tensor,
        means2D: torch.Tensor | None = None,
        opacities: torch.Tensor | None = None,
        filter3D: torch.Tensor | None = None,
        shs: torch.Tensor | None = None,
        colors_precomp: torch.Tensor | None = None,
        scales: torch.Tensor | None = None,
        rotations: torch.Tensor | None = None,
        cov3D_precomp: torch.Tensor | None = None,
        *,
        raster_settings: GaussianRasterizationSettings | None = None,
        return_aux: bool = False,
        features: torch.Tensor | None = None,
        _stage_options: FrameOptions | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | RasterizationOutput:
        config = (
            raster_settings if raster_settings is not None else self.raster_settings
        )
        if config is None:
            raise ValueError(
                "provide raster_settings at construction or for this frame"
            )
        if features is not None:
            if colors_precomp is not None:
                raise ValueError("provide features OR colors_precomp, not both")
            colors_precomp = features
        channels = 3
        if colors_precomp is not None:
            if (
                not isinstance(colors_precomp, torch.Tensor)
                or colors_precomp.ndim != 2
                or colors_precomp.shape[1] <= 0
            ):
                raise ValueError("features/colors_precomp must have shape (N,C), C > 0")
            channels = colors_precomp.shape[1]
        if self.channels is not None and channels != self.channels:
            raise ValueError(
                f"expected {self.channels} feature channels, got {channels}"
            )
        if self._renderer is native_renderer and channels != 3:
            raise ValueError(
                "the native RGB baseline requires 3 channels; use the default or Python/Warp renderer"
            )
        inputs: GaussianInputs = dict(
            means3D=means3D,
            opacities=opacities,
            filter3D=filter3D,
            shs=shs,
            colors_precomp=colors_precomp,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        if config.render_depth:
            raise ValueError("render_depth is not supported")
        if (
            not config.settings.eval_3D
            or int(config.settings.sort_settings.sort_mode) != 3
        ):
            raise ValueError("renderer currently supports eval_3D=True, sort_mode=HIER")
        from .autograd import rasterize

        output = rasterize(
            inputs,
            means2D,
            config,
            self._renderer,
            self._launch_cache,
            _stage_options,
        )
        if return_aux:
            return output
        return output["color"], output["radii"]
