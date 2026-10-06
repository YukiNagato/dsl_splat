"""Ordinary Python rasterizer with dynamic frames and analytic 3D backward."""

import torch

from .dispatch import FrameBindings, LaunchCache
from .binning import bin_and_sort
from .preprocess import preprocess
from .render import render_hierarchical_3d, native_renderer, warp_renderer


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

    def __init__(self, raster_settings=None, *, renderer=None, channels=None):
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
        means3D,
        means2D=None,
        opacities=None,
        filter3D=None,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        *,
        raster_settings=None,
        return_aux=False,
        features=None,
        _stage_options=None,
    ):
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
        inputs = dict(
            means3D=means3D,
            opacities=opacities,
            filter3D=filter3D,
            shs=shs,
            colors_precomp=colors_precomp,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        camera_inputs = (
            config.bg,
            config.viewmatrix,
            config.projmatrix,
            config.inv_viewprojmatrix,
            config.campos,
        )
        if torch.is_grad_enabled() and any(
            isinstance(value, torch.Tensor) and value.requires_grad
            for value in camera_inputs
        ):
            raise ValueError(
                "camera/background backward is not implemented; use torch.no_grad() for inference"
            )
        if config.render_depth:
            raise ValueError("render_depth is not supported")
        if (
            not config.settings.eval_3D
            or int(config.settings.sort_settings.sort_mode) != 3
        ):
            raise ValueError("renderer currently supports eval_3D=True, sort_mode=HIER")
        if torch.is_grad_enabled() and any(
            isinstance(value, torch.Tensor) and value.requires_grad
            for value in (*inputs.values(), means2D)
        ):
            from .autograd import rasterize_with_grad

            if self._renderer not in (
                render_hierarchical_3d,
                native_renderer,
                warp_renderer,
            ):
                raise ValueError(
                    "backward supports the native and Python/Warp hierarchical renderers"
                )
            output = rasterize_with_grad(
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
        options = {} if _stage_options is None else _stage_options
        bindings = options.get("bindings") or FrameBindings(self._launch_cache)
        state = options.get("preprocessed")
        if state is None:
            state = preprocess(
                **inputs,
                raster_settings=config,
                principal_point=options.get("principal_point"),
                _bindings=bindings,
            )
        if _stage_options is not None and "preprocessed" not in options:
            from .autograd import _stage_options as apply_stage_options

            with torch.no_grad():
                apply_stage_options(state, _stage_options)
        bins = bin_and_sort(state, config, diagnostics=False, _bindings=bindings)
        output = self._renderer(state, bins, config, _bindings=bindings)
        if state["radii"].numel() == 0:
            # rasterize_points.cu skips Rasterizer::forward for P=0 and leaves
            # its initially zero color intact. Low-level empty-tile rendering
            # still writes the background; apply the wrapper's special case here.
            output["color"].zero_()
        if return_aux:
            return dict(output, radii=state["radii"])
        return output["color"], state["radii"]
