# gsplat-style rasterization

For new integrations, see the [modern AAA API](API.md). This adapter translates
the gsplat signature into that core entry; it no longer calls the legacy 3DGS
rasterizer. Existing supported arguments, return layouts and defaults are kept.

The adapter follows the names and tensor layouts of
[gsplat 1.5.3 rasterization](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/rendering.py).
It renders with AAA 3D evaluation, adaptive filtering and StopThePop hierarchical
sorting. This is API compatibility for supported operations; images and
gradients are not expected to equal gsplat's 2D rasterizer. No gsplat package or
reference CUDA extension is required.

```python
# Replace: from gsplat import rasterization
from dsl_splat.aaa.gsplat import rasterization
# Also available: from dsl_splat import rasterization

render_colors, render_alphas, meta = rasterization(
    means=means,           # (N,3), world coordinates
    quats=quats,           # (N,4), wxyz, normalized by the adapter
    scales=scales,         # (N,3), activated positive scales
    opacities=opacities,   # (N,), activated opacities
    colors=features,      # (N,D), arbitrary signed attributes
    viewmats=viewmats,     # (C,4,4), world-to-camera, positive Z forward
    Ks=Ks,                # (C,3,3), fx/fy/cx/cy pinhole intrinsics
    width=width, height=height,
    rasterize_mode="antialiased", render_mode="RGB", packed=False,
)
# render_colors: (C,H,W,D), RGB or arbitrary attributes.
# render_alphas: (C,H,W,1)
loss = image_loss(render_colors) + mask_loss(render_alphas)
loss.backward()
```

Inputs are CUDA float32 on one device. Scales and opacities are activated values,
not logs/logits. Leading batch dimensions are supported: means `(...,N,3)`,
cameras `(...,C,4,4)`, Ks `(...,C,3,3)`, outputs `(...,C,H,W,D)`.
N may be zero; each batch must have at least one camera. Cameras are processed
sequentially on Torch's current stream. New allocations and noncontiguous
inputs are accepted; output storage belongs to each call.

## Features, SH and alpha

With `sh_degree=None`, colors are `(...,N,D)` or `(...,C,N,D)`, shared or
independent per camera. Any positive D is supported. Backgrounds have shape
`(...,C,D)` and default to zero. Features are not clamped or normalized.

With `sh_degree=0/1/2/3`, colors are `(...,N,K,3)` or `(...,C,N,K,3)`, with
at least `(sh_degree+1)**2` coefficients. SH is evaluated per view, offset by
0.5 and clamped below at zero. Fused Warp kernels provide first-order derivatives
through both SH coefficients and viewing directions to Gaussian means and poses.

Only `render_mode='RGB'` is supported; it blends all D feature channels with
AAA's renderer. `D`, `ED`, `RGB+D` and `RGB+ED` raise `NotImplementedError`.
The adapter does not synthesize a depth image by blending Gaussian center Z,
and does not return a `meta['depths']` field. `colors` is required.
Empty/fully culled scenes return the feature background and zero alpha.

Alpha is blended as a constant-one feature with zero background in the same
traversal. It supports opacity/geometry gradients. It is not computed from
the original backend's nondifferentiable `final_T`; summation can differ slightly
from `1-final_T`.

The newer `extra_signals` argument accepts `(...,N,E)` or `(...,C,N,E)`
post-activation attributes, with zero background. Output is
`meta['render_extra_signals']` of shape `(...,C,H,W,E)`; it does not change
`render_colors`. Extra-signal SH is not implemented.

## Options and camera conventions

The adapter converts OpenCV world-to-camera poses and zero-skew pinhole Ks to
AAA's transposed matrices, preserving fx/fy, arbitrary cx/cy and pixel centers.
Principal points can lie outside the image. AAA's view-plane AABB has been
extended to use cx/cy, so off-center cameras retain the default bounding method.
Centered cameras retain the original CUDA bounding arithmetic. `new_aabb=False` still
selects the projection-space bounds if explicitly requested in `aaa_settings`.

`near_plane` and `far_plane` are used to construct the camera projection matrix.
Culling follows AAA's 3D frustum-contribution test, bounding and tile culling;
the adapter does not discard Gaussians by comparing their center Z to those
planes. AAA's optional fixed `z < 0.2` center near-clipping is disabled by default.
These are AAA semantics, rather than gsplat's center near/far culling.
The adapter follows AAA's frontend camera utility: nominal `near_plane` and
`far_plane` map to normalized depth 0/1. For finite far, the projection depth row
is `[0, 0, far / (far - near), -far * near / (far - near)]`, with homogeneous W
equal to view-space Z. For infinite far, its limiting row is `[0, 0, 1, -near]`.
AAA's backend frustum helpers test normalized depth (-1,1). The adapter preserves
this frontend/backend combination: the helper's lower depth bound is
`near * far / (2 * far - near)` for finite far, or `near / 2` for infinite far;
the upper bound remains `far`. Thus `near_plane` is AAA's projection parameter,
rather than a strict center-Z cutoff. The contribution test also evaluates
Gaussian candidate positions. Core kernels and caller-supplied core projection
matrices are unchanged.
`radius_clip` uses AAA's maximum bounding radius after preprocessing.
`packed` compresses metadata only; the rendering buffers remain dense per view,
and images are identical. Tile size is fixed at 16. `channel_chunk` is accepted
as a positive performance hint; the backend fuses all channels into one traversal.

`rasterize_mode='classic'` disables AAA opacity compensation; `'antialiased'`
enables it. Both retain adaptive 3D dilation with fixed coefficient 0.3; other
`eps2d` values are rejected. Omit `with_eval3d` or set it True; this adapter always
uses AAA 3D evaluation and rejects an explicit False.

Adapter-specific `aaa_settings=ExtendedSettings(...)` overrides the default
AAA settings, including the compensation setting normally selected by
`rasterize_mode`. It must use eval_3D=True, HIER and supported queue sizes.
`filter3D` accepts `(...,N)` or `(...,N,1)` training filters and is constant
in backward.

## Metadata and training boundaries

Common fields include `radii`, `means2d`, `opacities`,
`tiles_per_gauss`, IDs and image/tile dimensions. Radii are the two rounded-up
AAA bounding extents. Means2d are projected Gaussian centers in pixel coordinates;
opacities include AAA compensation. Invalid dense rows are zeroed.

Packed geometry has first dimension M for visible Gaussian/camera pairs,
with int64 `batch_ids`, `camera_ids`, `gaussian_ids`. Batch IDs flatten leading
batch dimensions. Dense geometry has shape `(...,C,N,...)`; ID fields are None.
`conics` is None because AAA 3D evaluation uses ray transforms. gsplat-specific
intersection encoding fields are not fabricated.

First-order gradients support means, quats, scales, opacities, features, SH,
attribute producers, alpha and camera poses (`viewmats`). Gaussian filtering and
alpha-clamp gradients retain AAA's CUDA conventions. The camera adjoint also
differentiates adaptive filter variance using the forward `max(fx,fy)` focal,
opacity compensation and SH view direction, including the `filter3D` lower
bound on variance. Sorting, culling and alpha thresholds remain fixed during
backward. Intrinsic (`Ks`), background and second-order gradients are unsupported.

Set `viewmats.requires_grad_(True)` to receive `.grad` with shape `(...,C,4,4)`,
or construct `viewmats` from differentiable rotation/translation or SE(3)
parameters to propagate gradients to those parameters. Keep poses rigid during
optimization; gradients do not project arbitrary matrix updates back onto SE(3).
The pose convention remains OpenCV world-to-camera with positive Z forward.
AAA's frontend depth projection and all forward kernels are unchanged. The extra
camera adjoint runs only when a camera tensor requires gradients; fixed-camera
training and inference do not run it. `torch.no_grad()` still disables all training
state preparation, even for trainable camera inputs.

`meta['supports_means2d_grad']` is True. During training, `meta['means2d']`
is an auxiliary autograd input: image losses populate its `.grad`. It measures
the derivative of rigid pixel-space translation of each AAA footprint, with
sorting/culling held fixed. This is the 3D-renderer counterpart of gsplat's
projected-center derivative, rather than a derivative of a 2D conic approximation.

With `absgrad=True`, `.absgrad` records the componentwise sum of absolute
**per-pixel** derivatives. It is computed before Gaussian reduction, not as
`abs(means2d.grad)`. Both tensors use pixel units and the same packed/dense layout
as means2d. `.grad` follows ordinary Torch accumulation; `.absgrad` is replaced
per backward pass, following gsplat. Empty/clipped pairs have zero statistics.
Each outstanding frame owns its statistics. Under `torch.no_grad()` no screen
gradient graph is constructed.

The auxiliary tensor is independent of Gaussian parameters: the existing AAA
analytic backward already propagates their full gradients. Projecting this
statistic back into means3D would double-count derivatives. Do not use a loss
on `meta['means2d']` itself to optimize the model; use rendered-image losses.

gsplat 1.5.3 `DefaultStrategy` works with both `packed=False` and `packed=True`,
including `absgrad=True`. Keep its packed argument and absgrad option consistent
with the rendering call. Its Gaussian parameter/optimizer dictionaries retain
the normal gsplat convention (scales as logs and opacities as logits); activate
those values before calling rasterization. Batch rendering is supported, but
the upstream strategy manages one Gaussian parameter set, not independent
leading Gaussian batch dimensions.

```python
from gsplat.strategy import DefaultStrategy  # optional training dependency
from dsl_splat.aaa.gsplat import rasterization

strategy = DefaultStrategy(absgrad=True)
state = strategy.initialize_state()
strategy.check_sanity(params, optimizers)
# In the training loop:
colors, alphas, meta = rasterization(
    means=params['means'], quats=params['quats'],
    scales=params['scales'].exp(), opacities=params['opacities'].sigmoid(),
    colors=params['features'], viewmats=viewmats, Ks=Ks,
    width=width, height=height, packed=False, absgrad=strategy.absgrad,
)
strategy.step_pre_backward(params, optimizers, state, step, meta)
loss.backward()
strategy.step_post_backward(params, optimizers, state, step, meta, packed=False)
```

AAA's original eval_3D CUDA branch accumulates transform gradients rather than
means2D gradients. Its official trainer uses MCMC relocation/addition. The screen
statistics above are an extension for gradient-driven densification. Thresholds
such as `grow_grad2d` may need tuning for AAA's different rendering/filtering;
API interoperability is not a claim of equal training quality or gsplat gradients.

Explicitly unsupported: sparse gradients, distributed/segmented
rendering, covariance inputs, non-pinhole cameras, distortion, UT, rolling
shutter, depth rendering and newer gsplat hit-distance/normal modes.

## Validation

```bash
python -m tests.check_gsplat_interface
python -m tests.check_gsplat_screen_grad
python -m tests.check_camera_grad
# Optional actual upstream strategy integration, using a gsplat 1.5.3 checkout:
python -m tests.check_gsplat_screen_grad --gsplat-source /path/to/gsplat
python -m tests.check_features
python -m tests.check_backward --large --output /tmp/adapter-backward.json
```

The interface check uses independent native AAA RGB slices, camera/layout
checks including AAA's frontend depth mapping and infinite far, AAA culling parity for
centers outside near/far, depth-mode rejection,
batch comparisons, model finite differences, alpha-only gradients,
SH/extra-signal gradients, metadata layouts, empty/clipped scenes and stream
changes. It runs without gsplat or the reference CUDA extension.
The screen-gradient check uses a float64 world-ray/ellipsoid oracle, per-pixel
Jacobians, central differences through real principal-point shifts, cancellation
cases, packed/dense gradients and unchanged Gaussian gradients. The optional
upstream strategy check executes real callbacks and densification.
The camera-gradient check compares rotation and translation derivatives against
an independent float64 world-ray oracle and all six SE(3) finite differences,
including SH3, opacity compensation, adaptive/fixed filtering, packed/dense
batched cameras, unchanged model gradients, live frames, empty scenes and streams.
# Method-local adapters

AAA's adapter is `dsl_splat.aaa.gsplat.rasterization`; the top-level
`dsl_splat.rasterization` function is a direct alias. Original 3DGS has a
separate `dsl_splat.gs3d.gsplat.rasterization` adapter with its own supported
options. Import adapters from their method-local modules; the former root and
`compat` adapter modules have been removed. This document describes AAA.
