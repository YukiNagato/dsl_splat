"""First-order Torch autograd bridge to the analytic Warp backward kernels."""

from copy import copy

import torch
from torch.autograd.function import once_differentiable

from .backward_preprocess import backward_preprocess
from .backward_render import backward_render
from .binning import bin_and_sort
from .dispatch import FrameBindings
from .preprocess import preprocess

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


def _stage_options(state, options):
    """Optional adapter culling and read-only geometry metadata, before binning."""
    if options is None:
        return
    radius_clip = options.get("radius_clip", 0.0)
    if radius_clip > 0:
        keep = state["radii"] > radius_clip
        state["radii"] = torch.where(keep, state["radii"], 0)
        state["tiles_touched"] = torch.where(keep, state["tiles_touched"], 0)
        state["valid"] = state["valid"] & keep
    holder = options.get("state_out")
    if holder is not None:
        holder.update(state)


class Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        opacities,
        filter3D,
        shs,
        colors_precomp,
        scales,
        rotations,
        cov3D_precomp,
        config,
        renderer,
        launch_cache,
        stage_options,
    ):
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
        inputs = dict(zip(INPUT_NAMES, values))
        options = {} if stage_options is None else stage_options
        bindings = options.get("bindings") or FrameBindings(launch_cache)
        state = options.get("preprocessed")
        if state is None:
            state = preprocess(
                **{key: value for key, value in inputs.items() if key != "means2D"},
                raster_settings=config,
                principal_point=options.get("principal_point"),
                _bindings=bindings,
            )
            _stage_options(state, stage_options)
        bins = bin_and_sort(state, config, diagnostics=False, _bindings=bindings)
        output = renderer(state, bins, config, _bindings=bindings)
        if means3D.shape[0] == 0:
            output["color"].zero_()
        # Keep every frame's own intermediates. Save via Torch so in-place
        # changes to inputs/camera/auxiliary outputs are detected by autograd.
        named = []
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
        # Snapshot the four mutable dataclass nodes without recursively copying
        # their scalar/enum fields. Each outstanding frame retains its settings.
        settings = copy(config.settings)
        settings.sort_settings = copy(settings.sort_settings)
        settings.sort_settings.queue_sizes = copy(settings.sort_settings.queue_sizes)
        settings.culling_settings = copy(settings.culling_settings)
        ctx.config = config._replace(settings=settings)
        ctx.launch_cache = launch_cache
        ctx.screen_grad = bool(options.get("screen_grad", False))
        ctx.absgrad_target = options.get("absgrad_target")
        ctx.mark_non_differentiable(
            state["radii"], output["final_T"], output["contributors"]
        )
        ctx.set_materialize_grads(False)
        return (
            output["color"],
            state["radii"],
            output["final_T"],
            output["contributors"],
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_color, grad_radii, grad_t, grad_contributors):
        if grad_color is None:
            return (None,) * 13
        mappings = {
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
            target, rows, gaussian_ids = ctx.absgrad_target
            with torch.no_grad():
                if rows is None:
                    target.copy_(render_gradients["means2D_abs"])
                else:
                    target.index_copy_(
                        0, rows, render_gradients["means2D_abs"][gaussian_ids]
                    )
        gradients = backward_preprocess(
            inputs, mappings["state"], render_gradients, config, _bindings=bindings
        )
        result = []
        for name in INPUT_NAMES:
            if name not in inputs or name in ("filter3D", "cov3D_precomp"):
                result.append(None)
            elif name == "means2D":
                # Original AAA 3D calls retain their zero dummy gradient. The
                # gsplat adapter explicitly opts into virtual screen translation.
                result.append(
                    render_gradients["means2D"]
                    if ctx.screen_grad
                    else (
                        gradients[name]
                        if gradients[name].shape == inputs[name].shape
                        else torch.zeros_like(inputs[name])
                    )
                )
            else:
                result.append(gradients[name].reshape_as(inputs[name]))
        return (*result, None, None, None, None)


def rasterize_with_grad(
    inputs, means2D, config, renderer, launch_cache, stage_options=None
):
    values = dict(inputs, means2D=means2D)
    color, radii, final_t, contributors = Rasterize.apply(
        *(values.get(name) for name in INPUT_NAMES),
        config,
        renderer,
        launch_cache,
        stage_options,
    )
    return dict(color=color, radii=radii, final_T=final_t, contributors=contributors)
