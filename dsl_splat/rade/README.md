# RaDe-GS

This package implements RaDe-GS's projected-Gaussian rasterizer in NVIDIA Warp.
It uses a 3DGS-style backend. The AAA and GEER backends are separate methods.
No C++ extension or manual compilation is needed to use this package.

## Rendering

```python
from dsl_splat import rade

result = rade.rasterize(
    means, quats, scales, opacities, viewmats, Ks, width, height,
    features=colors,              # or sh_coeffs=sh, sh_degree=3
    extra_features=attributes,    # optional, separate output
    absgrad=True,
)
rgb = result.features
alpha = result.alpha
depth = result.depth
median_depth = result.median_depth
normals = result.normals
extra = result.extra_features
```

Inputs follow the other method APIs: activated scales `(N,3)` and opacities
`(N,)`, normalized internally wxyz quaternions `(N,4)`, and OpenCV world-to-camera
`viewmats (C,4,4)` and intrinsics `Ks (C,3,3)`. Means are `(N,3)`. Features are
`(N,D)` or `(C,N,D)`. Shared Gaussian batch dimensions can prefix these layouts.
Outputs are channel-last: `(C,H,W,D)` for features, `(C,H,W,1)` for alpha and
both depths, and `(C,H,W,3)` for normals, with the same optional batch prefix.

Expected depth is the alpha-normalized sum of each contributor's ray-plane
distance, converted to camera Z. Median depth selects the last contributor
whose incoming transmittance exceeds 0.5; it is not a normalized expectation.
Normals are normalized alpha-weighted camera-space normals. Uncovered pixels
have zero depth and zero normal. Additional attributes use ordinary alpha
blending and are returned separately from the main features.

`kernel_size=0.0` matches the official default. A positive value dilates the
2D covariance and compensates opacity. Center clipping defaults to `z>0.2`;
`near_plane`, `far_plane` and `radius_clip` can override it. Tile size is 16.
Sorting uses distance from the camera to the Gaussian center, as in RaDe-GS.

The gsplat-style adapter is method-local:

```python
from dsl_splat.rade.gsplat import rasterization
image, alpha, info = rasterization(
    means, quats, scales, opacities, colors, viewmats, Ks, width, height,
    render_mode="RGB+ED", eps2d=0.0,
)
# info["depth"], info["median_depth"], info["normals"]
```

`ED` uses RaDe's expected Z depth; `D` multiplies it by alpha. `RGB`, `D`, `ED`,
`RGB+D` and `RGB+ED` are supported. The adapter keeps gsplat's near-plane default
0.01, while the native API defaults to 0.2. `eps2d` maps to `kernel_size`.
`packed` changes training metadata storage, with dense per-view rendering.
It does not select a different renderer or reproduce gsplat's depth algorithm.

## Explicit stages and shared components

`rade.ops.prepare_gaussians`, `rade.ops.bin_and_sort` and
`rade.ops.rasterize_to_pixels` expose differentiable single-view stages using
`dsl_splat.cameras.PinholeCamera`. Projection includes ray planes and normals;
rendering produces color and geometry in one traversal. The full API uses a
single rendering autograd boundary to avoid the overhead of composing stages.

`dsl_splat/splatting` holds shared camera preparation, projected footprint
storage, small geometry operations, square tile duplication/radix sorting and
training metadata preparation. Each method supplies its own projection policy,
sorting depth and pixel evaluation. `gs3d` keeps its existing public imports.
There is no renderer switch inside a universal `rasterize` function.
The clamp VJP is method-specific: RaDe differentiates saturated `limit*z`
through z, while original 3DGS retains its legacy saturated-Jacobian convention.
Offscreen large-Gaussian regressions compare each against its own CUDA source.

## 3D filtering

As in the official project, camera-dependent 3D filtering is a frontend step.
The rasterizer accepts the resulting scales and opacities:

```python
filtered_scales, filtered_opacities = rade.apply_3d_filter(
    scales, opacities, filter_3d,  # filter_3d: (...,N,1), width, not variance
)
result = rade.rasterize(
    means, quats, filtered_scales, filtered_opacities,
    viewmats, Ks, width, height, features=colors,
)
```

The helper uses `sqrt(scales**2 + filter_3d**2)` and the determinant opacity
compensation, with ordinary PyTorch autograd. The caller computes filter widths
from training cameras, refreshes them after densification and applies the
matching opacity-reset policy. The renderer does not own training-camera state
or silently substitute GEER's fixed filter. A full RaDe training pipeline,
integration/mesh extraction, sparse gradients, intrinsic/background gradients,
contribution statistics, distortion and higher derivatives are not implemented.

## Reference conventions

Source: [HKUST-SAIL/RaDe-GS](https://github.com/HKUST-SAIL/RaDe-GS), revision
`d72f20792005ae1d6555a82aa2d15345f247604e`, particularly
`submodules/diff-gaussian-rasterization/cuda_rasterizer/render_forward.cu`,
`render_backward.cu`, and `rasterizer_impl.cu`.

The first implementation preserves the CUDA reference's first-order backward
conventions, including inverse-conic regularization and its hand-written
normalization/opacity-compensation VJPs. In upstream `computeCov2DCUDA`, the
normal VJP divides by the *normalized* vector's length, and the determinant
coefficient VJP uses `-d_det0 * sqrt(det0/det1)` for the second determinant term.
These differ from the exact derivatives of the forward formulas. They are
explicit custom adjoints here, not assumptions that Warp AD reproduces CUDA.
Pose gradients extend these same conventions; this is not a claim that every
reference VJP passes mathematical finite differences.

Two deliberate extensions are arbitrary principal points in both projection
and pixel-ray conversion (the original pixel kernel assumes centered intrinsics),
and finite zero geometry/gradients for uncovered pixels. Empty scenes render
the background. Extremely ill-conditioned, undilated footprints can amplify
float32 arithmetic differences; strict VJP agreement is not claimed for
unfiltered near-singular footprints. The tests distinguish ordinary/filtered
scenes from this stress case.

Sorting keys are a stricter requirement: a one-ULP distance difference can
reverse nearly equal contributors and create a visible color error. Forward
uses the reference's `norm3df` and `--use_fast_math` behavior. CUDA comparisons
check visible sorting distances bit-for-bit, including both benchmark views.
The million-Gaussian front fixture exposed this issue; after the correction,
its first view's maximum RGB error fell from `0.108` to `1.9e-6`.

The renderer also contains hard alpha/visibility/termination tests and discrete
median selection. Last-bit differences can change isolated boundary pixels and
their normalized geometry outputs. The benchmark reports mean/max output errors
and counts exceeding `1e-4` across both views; it does not assume unconditional
per-pixel equality.

## Validation

```bash
python -m tests.check_rade --cuda-source RaDe-GS/submodules/diff-gaussian-rasterization
python -m tests.check_rade_interfaces
python -m tests.check_gs3d
```

The optional CUDA reference builds under a separate module name in the system
temporary directory and is only needed for comparison. Tests compare all five
outputs and their independent VJPs, SH0–3, ordinary and filtered thin Gaussians,
then check an independent double-precision Torch oracle, off-center cameras,
stage composition, changing Gaussian counts/channels, packed/batched metadata,
absgrad, extra attributes, empty pixels and pose gradients for color/depth.
The pixel kernels cooperatively load features, ray planes and normals. Backward
reduces gradients within each 32-thread warp before global atomic additions,
including the per-pixel magnitudes used by absgrad. Wide feature vectors fall
back to global feature reads to stay within the static shared-memory budget;
the public channel count is unrestricted. Rendering uses fast math like the
CUDA reference, and finished pixels leave the inner traversal immediately.
The small native helpers only expose shared storage and warp collectives;
traversal, blending and gradient formulas remain Python/Warp.

Each output has its own autograd edge. Backward reads the HWC upstream views
with their actual strides, treating unused outputs as zero without allocating
full-image gradient buffers. API images remain materialized and can be processed
in place. Camera inputs, feature layouts and public return values are unchanged.

A reproducible 4090 D baseline is recorded in
[the initial performance report](../../tests/results/rade_4090d_performance.md).
Controlled before/after results are recorded in
[the optimization report](../../tests/results/rade_4090d_optimization.md).

```bash
python -m tests.benchmark_rade --cuda-source RaDe-GS/submodules/diff-gaussian-rasterization \
    --counts 100000 1000000 --distributions room front --iterations 20 --repeats 6
python -m tests.profile_rade --cuda-source RaDe-GS/submodules/diff-gaussian-rasterization \
    --counts 100000 1000000 --distributions room front
```

The default benchmark supplies preallocated output gradients, excluding loss
construction and optimizer work. Add `--include-loss` to also measure a small
loss and its backward. CPU scheduling, affinity and frequency remain unchanged.
Profiler GPU durations are summed device activities, without Python idle gaps;
profiled host timings are kept separate from the ordinary benchmark.

## License

RaDe-GS's rasterizer derives from Inria/GRAPHDECO's Gaussian-Splatting code and
retains its research-use license. The upstream text is included in
[RADE_LICENSE.txt](RADE_LICENSE.txt); see also the package [NOTICE](../NOTICE.md).
