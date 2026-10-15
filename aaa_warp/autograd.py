"""First-order Torch autograd bridge to the analytic Warp backward kernels."""

from typing import Any, cast

import torch
from torch.autograd.function import once_differentiable

from .backward_preprocess import backward_preprocess
from .backward_camera import backward_camera
from .backward_render import backward_render
from .binning import bin_and_sort
from .dispatch import FrameBindings, LaunchCache
from .frame import FrameOptions, prepare_frame
from .render import render_hierarchical_3d, native_renderer, warp_renderer
from .config import FrameConfig
from .settings import copy_settings
from .types import GaussianInputs, Renderer, RasterizationOutput

INPUT_NAMES = (
    "means3D",
    "means2D",
    "opacities",
    "filter3D",
    "shs",
    "colors_precomp",
    "scales",
    "rotations",
    "cov3D_precomp",
)
CAMERA_NAMES = ("bg", "viewmatrix", "projmatrix", "inv_viewprojmatrix", "campos")
DIFFERENTIABLE_CAMERAS = ("viewmatrix", "projmatrix", "campos")


class Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        means3D: torch.Tensor,
        means2D: torch.Tensor | None,
        opacities: torch.Tensor | None,
        filter3D: torch.Tensor | None,
        shs: torch.Tensor | None,
        colors_precomp: torch.Tensor | None,
        scales: torch.Tensor | None,
        rotations: torch.Tensor | None,
        cov3D_precomp: torch.Tensor | None,
        viewmatrix: torch.Tensor,
        projmatrix: torch.Tensor,
        campos: torch.Tensor,
        config: FrameConfig,
        renderer: Renderer,
        launch_cache: LaunchCache,
        stage_options: FrameOptions | None,
        grad_enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        values = (
            means3D,
            means2D,
            opacities,
            filter3D,
            shs,
            colors_precomp,
            scales,
            rotations,
            cov3D_precomp,
        )
        inputs = cast(GaussianInputs, dict(zip(INPUT_NAMES, values)))
        # Function.forward always runs without grad recording. needs_input_grad
        # describes its input tensors even under an outer no_grad/inference_mode,
        # so combine it with the caller's mode before preparing backward state.
        differentiable_count = len(INPUT_NAMES) + len(DIFFERENTIABLE_CAMERAS)
        needs_backward = grad_enabled and any(
            ctx.needs_input_grad[:differentiable_count]
        )
        if grad_enabled and any(
            isinstance(value, torch.Tensor) and value.requires_grad
            for value in (config.bg, config.inv_viewprojmatrix)
        ):
            raise ValueError(
                "background/inverse projection backward is not implemented; use torch.no_grad() for inference"
            )
        if needs_backward and renderer not in (
            render_hierarchical_3d,
            native_renderer,
            warp_renderer,
        ):
            raise ValueError(
                "backward supports the native and Python/Warp hierarchical renderers"
            )
        frame = prepare_frame(inputs, config, launch_cache, stage_options)
        state, bindings = frame.state, frame.bindings
        bins = bin_and_sort(state, config, diagnostics=False, _bindings=bindings)
        output = renderer(state, bins, config, _bindings=bindings)
        if means3D.shape[0] == 0:
            output["color"].zero_()
        result = (
            output["color"],
            state["radii"],
            output["final_T"],
            output["contributors"],
        )
        if not needs_backward:
            return result
        # Keep every frame's own intermediates. Save via Torch so in-place
        # changes to inputs/camera/auxiliary outputs are detected by autograd.
        named: list[tuple[str, str, torch.Tensor]] = []
        for namespace, mapping in (
            ("inputs", inputs),
            (
                "state",
                {
                    key: state[key]
                    for key in ("radii", "gauss2screen", "opacity", "rgb", "clamped")
                    if key in state
                },
            ),
            ("bins", {key: bins[key] for key in ("ranges", "point_list")}),
            ("output", output),
        ):
            named.extend(
                (namespace, key, value)
                for key, value in mapping.items()
                if isinstance(value, torch.Tensor)
            )
        named.extend(("camera", key, getattr(config, key)) for key in CAMERA_NAMES)
        ctx.names = [(namespace, key) for namespace, key, _ in named]
        ctx.save_for_backward(*(value for _, _, value in named))
        ctx.config = config._replace(settings=copy_settings(config.settings))
        ctx.launch_cache = launch_cache
        ctx.gaussian_grad = any(
            ctx.needs_input_grad[index]
            for index, name in enumerate(INPUT_NAMES)
            if name not in ("means2D", "filter3D", "cov3D_precomp")
        )
        ctx.camera_grad = any(
            ctx.needs_input_grad[len(INPUT_NAMES) : differentiable_count]
        )
        ctx.screen_grad = stage_options is not None and stage_options.screen_grad
        ctx.absgrad_target = (
            None if stage_options is None else stage_options.absgrad_target
        )
        ctx.mark_non_differentiable(
            state["radii"], output["final_T"], output["contributors"]
        )
        ctx.set_materialize_grads(False)
        return result

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        grad_color: torch.Tensor | None,
        grad_radii: torch.Tensor | None,
        grad_t: torch.Tensor | None,
        grad_contributors: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        if grad_color is None:
            return (None,) * len(ctx.needs_input_grad)
        mappings: dict[str, dict[str, torch.Tensor]] = {
            name: {} for name in ("inputs", "state", "bins", "output", "camera")
        }
        for (namespace, key), value in zip(ctx.names, ctx.saved_tensors):
            mappings[namespace][key] = value
        inputs = mappings["inputs"]
        config = ctx.config._replace(**mappings["camera"])
        bindings = FrameBindings(ctx.launch_cache)
        render_gradients = backward_render(
            mappings["state"],
            mappings["bins"],
            mappings["output"],
            grad_color,
            config,
            screen_grad=ctx.screen_grad,
            absgrad=ctx.absgrad_target is not None,
            _bindings=bindings,
        )
        if ctx.absgrad_target is not None:
            ctx.absgrad_target.write(render_gradients["means2D_abs"])
        gradients = (
            backward_preprocess(
                inputs, mappings["state"], render_gradients, config, _bindings=bindings
            )
            if ctx.gaussian_grad
            else {}
        )
        camera_gradients = (
            backward_camera(
                inputs, mappings["state"], render_gradients, config, _bindings=bindings
            )
            if ctx.camera_grad
            else {}
        )
        result: list[torch.Tensor | None] = []
        for index, name in enumerate(INPUT_NAMES):
            if (
                not ctx.needs_input_grad[index]
                or name not in inputs
                or name in ("filter3D", "cov3D_precomp")
            ):
                result.append(None)
            elif name == "means2D":
                # Original AAA 3D calls retain their zero dummy gradient. The
                # gsplat adapter explicitly opts into virtual screen translation.
                gradient = (
                    render_gradients["means2D"]
                    if ctx.screen_grad
                    else torch.zeros_like(inputs[name])
                )
                result.append(
                    gradient
                    if gradient.shape == inputs[name].shape
                    else torch.zeros_like(inputs[name])
                )
            else:
                result.append(gradients[name].reshape_as(inputs[name]))
        result.extend(
            (
                camera_gradients[name]
                if ctx.needs_input_grad[len(INPUT_NAMES) + index]
                else None
            )
            for index, name in enumerate(DIFFERENTIABLE_CAMERAS)
        )
        return (*result, None, None, None, None, None)


def rasterize(
    inputs: GaussianInputs,
    means2D: torch.Tensor | None,
    config: FrameConfig,
    renderer: Renderer,
    launch_cache: LaunchCache,
    stage_options: FrameOptions | None = None,
) -> RasterizationOutput:
    """One autograd entry for training and inference, preserving the caller mode."""
    values = dict(inputs, means2D=means2D)
    color, radii, final_t, contributors = Rasterize.apply(
        *(values.get(name) for name in INPUT_NAMES),
        *(getattr(config, name) for name in DIFFERENTIABLE_CAMERAS),
        config,
        renderer,
        launch_cache,
        stage_options,
        torch.is_grad_enabled(),
    )
    return dict(color=color, radii=radii, final_T=final_t, contributors=contributors)
