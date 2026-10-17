# DSL Splat: installation and AAA API

DSL Splat implements Gaussian splatting algorithms with NVIDIA Warp.
This page documents AAA, which lives in `dsl_splat.aaa`; original 3DGS is
available independently in `dsl_splat.gs3d`. Each method has its own `gsplat.py` adapter and low-level
`ops`. Top-level exports in this page remain compatible AAA entry points.
See [the library structure and contribution-statistics API](../docs/architecture.md).

NVIDIA Warp implementation of AAA-Gaussians preprocessing, binning, 3D
hierarchical rendering and first-order backward. Torch manages inputs and
outputs; Warp compiles and caches kernels automatically on first use. Rendering
and training do not import the reference CUDA extension or require CMake, GLM
headers, or a manual CUDA build.

The numerical reference is
[AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering](https://arxiv.org/abs/2504.12811)
and its CUDA implementation. License and attribution files are distributed with
the package: `LICENSE.md`, `NOTICE.md`, and `STOPTHEPOP_LICENSE.txt`.

## Installation

Use a CUDA-enabled Torch environment:

```bash
python -m pip install -e .
```

Runtime dependencies are PyTorch and `warp-lang>=1.17`. Initial kernel compilation
is excluded from warmed benchmark measurements.

## Dynamic rendering and training

The recommended entry uses OpenCV world-to-camera poses and pinhole K:

```python
from dsl_splat import rasterize

result = rasterize(
    means, quats, scales, opacities, viewmats, Ks, width, height,
    features=features, absgrad=True,
)
loss = image_loss(result.features) + mask_loss(result.alpha)
loss.backward()
```

Defaults enable AAA adaptive antialiasing and HIER 64/8/4. See
[the modern API and composable differentiable stages](API.md).

The original 3DGS/AAA interface remains available for existing callers:

```python
import torch
from dsl_splat import (
    ExtendedSettings, GaussianRasterizationSettings, GaussianRasterizer,
)

settings = ExtendedSettings.from_json("config.json")
config = GaussianRasterizationSettings(
    image_height=height, image_width=width,
    tanfovx=tanx, tanfovy=tany,
    bg=background, scale_modifier=1.0,
    viewmatrix=viewmatrix, projmatrix=projmatrix,
    inv_viewprojmatrix=torch.linalg.inv(projmatrix).contiguous(),
    sh_degree=3, campos=camera_position, prefiltered=False,
    settings=settings, render_depth=False, debug=False,
)
renderer = GaussianRasterizer(config)
color, radii = renderer(
    means3D=means3D, opacities=opacities,
    scales=scales, rotations=rotations,
    shs=shs, filter3D=filter3D,
)
# Trainable input tensors must have requires_grad=True.
loss = (color - target_image).square().mean()
loss.backward()

# Each frame accepts newly allocated inputs and changing N, camera or resolution.
with torch.no_grad():
    color, radii = renderer(**next_inputs, raster_settings=next_config)
```

Supply either `shs` (degree 0–3) or `colors_precomp` / `features`. Inputs are CUDA float32 on
one device. Kernels run on Torch's current stream; noncontiguous inputs are made
contiguous. Each ordinary call owns its outputs. `return_aux=True` returns
`color`, `radii`, `final_T`, and `contributors` in a dictionary. Empty input
produces a black image, matching the reference wrapper; nonempty, fully culled
input produces the background.

The default forward renderer uses the unchanged native baseline for three
channels and the specialized Python/Warp renderer for other channel counts.
To select Python/Warp explicitly for all channel counts:

```python
from dsl_splat.aaa.render_hierarchical_warp import render_hierarchical_3d
renderer = GaussianRasterizer(config, renderer=render_hierarchical_3d)
```

Both forward implementations support ordinary first-order training. Backward
replays the shared Python/Warp hierarchical traversal and applies analytic
adjoints. It supports means, opacity, scales, rotations, precomputed features and
SH0–3. It follows CUDA's filter and alpha-clamp gradient conventions: `filter3D`
is constant. First-order camera gradients support `viewmatrix`, `projmatrix`
and `campos`. The camera adjoint differentiates the forward adaptive filter,
opacity compensation and SH view direction while retaining existing CUDA
conventions for Gaussian gradients. Sorting, culling and alpha thresholds remain
fixed during backward. Background/inverse projection gradients and second
derivatives are unsupported.

## Arbitrary feature channels

`features` and the existing `colors_precomp` argument accept `(N,C)` tensors,
with any positive integer channel count. Pass exactly one of `features`,
`colors_precomp`, or `shs`. Background has shape `(C,)` and the returned image
has shape `(C,H,W)`; the legacy `color` output name is retained. SH evaluation
continues to generate three RGB channels.

```python
# RGB + normal + plane distance, for example; all are per-Gaussian attributes.
attributes = torch.cat((rgb, normals, plane_distance[:, None]), dim=1)  # (N,7)
feature_config = config._replace(bg=torch.zeros(7, device=attributes.device))
feature_renderer = GaussianRasterizer(feature_config, channels=7)
image, radii = feature_renderer(
    means3D=means3D, opacities=opacities,
    scales=scales, rotations=rotations, features=attributes,
)
rgb_image, normal_image, distance_image = image[:3], image[3:6], image[6:7]
loss = some_loss(rgb_image, normal_image, distance_image)
loss.backward()  # Includes attributes and the parameters used to create them.
```

`channels=C` is an optional channel-count constraint, similar to selecting a
template specialization. Without it, the count is inferred per frame. Warp
compiles a fixed-length vector accumulator and unrolled channel operations for
each `(C, hierarchical_4x4_culling)` combination, then caches the kernels.
Changing N, camera, or resolution does not require a new specialization. All
channels share one hierarchical traversal, sort, alpha and transmittance; no
per-channel render passes or per-channel geometry preprocessing are needed.
The native RGB baseline remains unchanged.

Features can be signed or exceed the RGB range; they are not clamped or
normalized. The result for channel c is `sum(T_i * alpha_i * feature_i[c]) +
T_final * bg[c]`, using the existing AAA sampling and termination rules.
Normalize composited normals or divide zero-background attributes by
`1 - final_T` in the caller if your application needs those semantics. This
provides attribute blending; it does not implement PGSR's geometry losses or
plane-derived depth by itself.

## Supported behavior

A [gsplat-style rasterization adapter](GSPLAT_COMPAT.md) is available as
`from dsl_splat.aaa.gsplat import rasterization`. It supports batched pinhole cameras,
features/SH, alpha outputs, arbitrary cx/cy, and signed/absolute
screen gradients for gsplat DefaultStrategy densification, and pose gradients
through OpenCV world-to-camera `viewmats`.
Its backend retains AAA evaluation and sorting; the compatibility document
describes metadata and unsupported gsplat options. Depth render modes are
rejected; the adapter applies AAA culling without extra center near/far clipping.

Full rendering and backward support `eval_3D=True`, `sort_mode=HIER`, and queue
sizes 64/8/4. Other queue sizes, full 2D rendering, PPX sorting modes, depth output
and debug visualization are not implemented. Preprocessing and binning retain
the reference feature branches needed for their stage-level use and validation.

Repository `config.json` enables adaptive 3D filtering, new AABB, tile/4x4
culling, load balancing, HIER 64/8/4 and PTD_MAX depth keys. Settings dataclass
field defaults follow the reference API; load `config.json` for this recommended
configuration.

The hierarchical renderer follows StopThePop queue organization: a 256-thread
block per 16x16 tile, shared 64-entry tails, 8-entry mids, and 4-entry per-pixel
heads. Python/Warp implements queue traversal, sorting, merging and gradient
formulas. Small native Warp intrinsics provide subgroup operations, aligned
loads, reciprocal rounding and global atomics. Scan/radix sorting uses Warp's
built-in operations. All native primitives are compiled automatically by Warp.

The native renderer's shared/staged diagnostic fallback remains available via
its existing options. The early fused renderer and fixed-shape
`PreparedRasterizer3D` experiment have been moved out of the core package.

## Separate stages

```python
from dsl_splat import preprocess, bin_and_sort, render_hierarchical_3d

state = preprocess(**inputs, raster_settings=config)
bins = bin_and_sort(state, config, diagnostics=False)
output = render_hierarchical_3d(state, bins, config)
```

`bin_and_sort` reads the pair count once on the host before allocating its
variable-sized arrays, as the CUDA reference does. Its default
`diagnostics=True` also returns offsets and unsorted/sorted keys for comparisons;
ordinary rendering uses `False` to avoid extra copies. Tile ranges delimit valid
entries; reserved invalid pairs have Gaussian ID -1.

Preprocess defines `radii`, `tiles_touched` and `valid` for every row. Geometry,
SH-generated RGB and SH clamp flags are meaningful only where `valid` is true;
rejected rows are not cleared. The legacy `state['rgb']` key holds `(N,C)`
features. Precomputed features alias the contiguous input, so treat this state
view as read-only; noncontiguous inputs are copied. SH-generated RGB owns a
fresh buffer. Quaternions are w,x,y,z and are not normalized by preprocessing.
Camera layout and transform bytes follow the reference's GLM conventions.
`contributors` is a diagnostic count, not CUDA HIER's unwritten `n_contrib`.

## Code organization and types

`aaa/api.py` exposes the modern AAA rendering/training API, with composable
differentiable stages in `aaa/ops.py`. `aaa/legacy.py` preserves the original
3DGS/AAA interface. `aaa/gsplat.py` owns AAA's adapter, while
`gs3d/gsplat.py` independently adapts original 3DGS. Top-level function exports
point directly to AAA; adapter imports use the method-local module. Shared input
normalization and camera definitions live
in `inputs.py` and `cameras.py`; methods own their camera lowering and geometry.

`aaa/frame.py` shares AAA frame preparation between inference, autograd and the adapter.
Ordinary AAA rendering enters `Rasterize.apply` in `aaa/autograd.py`; its forward
executes the same stages for training and inference. It saves backward state
only when the caller enables gradients and an input requires them. The caller's
grad mode is passed explicitly because `Function.forward` runs without grad
recording, and `ctx.needs_input_grad` alone does not distinguish `no_grad()`.
`aaa/types.py` describes AAA stage inputs, outputs and renderer callables using
`TypedDict` and `Protocol`; these annotations add no tensor wrappers or runtime
validation. The package includes `py.typed` for downstream type checkers.
Preprocessing, binning, rendering and their backward stages remain separate
modules. `runtime/dispatch.py` and `runtime/interop.py` handle launch caching and Torch/Warp
views. Native primitives and the native RGB baseline remain beside their Warp
counterparts; diagnostic renderers are retained because the baseline uses them.

For development, install the optional tools `mypy` and `black`. Run
`python -m mypy` from the repository root to check the host interfaces configured in
`pyproject.toml`. Warp's `wp.array(...)` annotations describe GPU compiler types;
kernel correctness is checked by compilation and numerical regression tests.
`python -m black dsl_splat` formats for Python 3.10 and excludes the native
baseline, which is deliberately kept unchanged.

## Validation and benchmarks

CUDA parity checks require an installed reference extension with the development
stage/debug bindings. It is a test dependency, not a runtime dependency.
AAA checks import `aaa_gaussian_rasterization`; original 3DGS checks import
`diff_gaussian_rasterization`.

```bash
python -m tests.check_gs3d
python -m tests.check_contribution_statistics
python -m tests.check_preprocess
python -m tests.check_preprocess_features
python -m tests.check_cooperative_culling
python -m tests.check_binning
python -m tests.check_default_config
python -m tests.check_native_math
python -m tests.check_native_render
python -m tests.check_warp_renderer
python -m tests.check_public_rasterizer
python -m tests.check_dispatch
python -m tests.check_uninitialized_geometry
python -m tests.check_features
python -m tests.check_gsplat_interface
python -m tests.check_gsplat_screen_grad
python -m tests.check_backward --large --output /tmp/backward-consistency.json
python -m tests.check_warp_renderer --audit --large --scene-matrix --check --output /tmp/forward-consistency.json
```

Forward comparisons check defined buffers and rendered outputs bitwise on the
covered cases. Backward uses tolerances for nondeterministic floating-point
atomic accumulation. Cases cover dynamic frames, saved-Tensor hooks, outstanding
frames, stream changes, queue boundaries, tied depths, SH, filtering and culling.

Ordinary forward and forward/backward benchmarks compare CUDA, native forward
and Python/Warp forward on synthetic room/front scenes and alternating cameras:

```bash
python -m tests.benchmark_warp_renderer --counts 100000 1000000 --resolutions 1280x720 --output /tmp/forward.json
python -m tests.benchmark_backward --counts 100000 1000000 --output /tmp/training.json
python -m tests.benchmark_features --counts 100000 1000000 --channels 3 7 16 --output /tmp/features.json
```

These are synthetic workloads, not trained-model performance guarantees. The
forward benchmark also reports render GPU timings separately from ordinary
full-forward timings. Historical probes, reports and measurements are kept in
the local ignored `research/` archive.

`check_features` covers 1/2/3/4/7/8/16/32/64 channels. It compares each channel
with independent three-channel CUDA renders and compares all geometric gradients
with the sum of those CUDA backward passes, plus finite differences for features
and opacity. `benchmark_features` measures Python/Warp forward and training with
all channels fused into one traversal, excluding initial JIT compilation.

Standalone wheel checks block reference imports and use a fresh Warp JIT cache:

```bash
python -m pip wheel . --no-deps --no-build-isolation -w /tmp/aaa-wheels
python -m tests.check_wheel_runtime /tmp/aaa-wheels/dsl_splat-0.1.0-py3-none-any.whl
python -m tests.check_wheel_runtime /tmp/aaa-wheels/dsl_splat-0.1.0-py3-none-any.whl --renderer python
```
