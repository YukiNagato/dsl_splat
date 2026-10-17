# Rendering library structure

The distribution is named `dsl-splat` and its Python package is `dsl_splat`.
Rendering methods have independent entry points, configurations and autograd
bridges:

```python
from dsl_splat import aaa, gs3d

aaa_result = aaa.rasterize(..., filter3d=filters)
gs3d_result = gs3d.rasterize(...)
```

There is no central algorithm selector. Each method validates its own inputs
and owns its projection, culling, traversal and backward semantics. PGSR and
RaDe-GS are future additions, not aliases of arbitrary feature blending.

## Calling a method

```python
from dsl_splat import gs3d

result = gs3d.rasterize(
    means, quats, scales, opacities,
    viewmats, Ks, width, height,
    features=features,
    absgrad=True,
)
loss = result.features.mean()
loss.backward()
screen_grad = result.means2d.grad
absolute_screen_grad = result.means2d.absgrad
```

For an unbatched Gaussian model, the inputs are:

| Input | Shape / meaning |
| --- | --- |
| `means` | `(N,3)`, world positions |
| `quats` | `(N,4)`, wxyz, normalized internally |
| `scales` | `(N,3)`, activated scales |
| `opacities` | `(N,)`, activated opacities |
| `viewmats` | `(C,4,4)`, OpenCV world-to-camera poses |
| `Ks` | `(C,3,3)`, zero-skew intrinsics, arbitrary cx/cy |
| `features` | `(N,D)` or `(C,N,D)` |

Use CUDA float32 tensors on the same device. A single camera still has `C=1`;
Gaussians need no extra batch dimension. Optional Gaussian batch dimensions
precede these axes, with matching batch dimensions on the cameras. Images have
shape `(...,C,H,W,D)`, alpha `(...,C,H,W,1)`. Supply either `features`, or
`sh_coeffs` with optional `sh_degree`. `extra_features` is blended separately
and returned as `result.extra_features`, without changing `result.features`.

Original 3DGS returns `means2d` in its CUDA raster coordinates, where pixel
centers have integer coordinates. Its values equal the K-projected positions
minus 0.5; culled rows are zero. The gsplat adapter keeps this native convention.
This metadata offset does not affect rendered images or screen gradients.

The analogous AAA call is `aaa.rasterize(...)`; its method-specific additions
include `filter3d` and `settings`. The full AAA contract is documented in
[API.md](../dsl_splat/API.md). Internal modules moved under their method or
shared component; private import paths are not compatibility guarantees.

## Responsibilities

| Location | Responsibility |
| --- | --- |
| `aaa/` | AAA geometry, hierarchical traversal, analytic backward and adapters |
| `gs3d/` | Original 3DGS projection, square bounds, center-Z order and backward |
| `runtime/` | Current-stream interop, frame-local descriptors, launch packets and gradient destinations |
| `primitives/` | Shared scans, radix sorting, tile ranges, SH, quaternion math and small GPU intrinsics |
| `cameras.py` | OpenCV camera definition and intrinsic validation |
| `inputs.py` | Tensor layout checks, activated parameters and SH evaluation |
| `training.py` | Packing metadata and binding auxiliary gradient destinations |
| `statistics.py` | Optional reductions from accepted pixel contributions to Gaussians |

Methods supply their own geometry to the shared metadata utilities. Sharing
packing code does not make AAA bounding extents equivalent to original 3DGS
square radii, or change the meaning of either method's screen-space gradients.

Each method exposes a separate `gsplat.py` adapter:

```python
from dsl_splat.aaa.gsplat import rasterization
# or: from dsl_splat.gs3d.gsplat import rasterization
```

Adapters translate signatures and results and reject unsupported options.
Projection, clipping, geometry outputs and differentiation belong to the native
method implementation. Existing top-level functions such as
`dsl_splat.rasterize`, `dsl_splat.rasterization`, and the `ops` attribute continue
to select AAA through direct imports in `__init__.py`. The redundant root
forwarding modules and `compat/` package have been removed. Import adapters as
`dsl_splat.aaa.gsplat` or `dsl_splat.gs3d.gsplat`.

## Differentiation and low-level operations

The complete entry points use a Torch rendering Function per view, together
with differentiable input/camera preparation. They save frame-owned state only
when backward is needed. Neither launch caches nor
compiled kernel variants retain old input tensors or impose fixed N/pointers.

Launch commands are bounded to 64 entries per execution thread and updated
through Warp's parameter API on every submission. Descriptor layout templates
are separately bounded to 64 entries; Tensor owners live only in the current
frame's bindings. Users do not manage these caches or temporary GPU buffers.

Input preparation stores Gaussian and feature batches as Python tuples of
Tensor views. Unbatched inputs retain their original Tensor; real batches share
one `unbind` operation per attribute. Indexing a model therefore needs no
artificial singleton batch or gradient scatter. The public Tensor layouts are
unchanged. See the [batch-view measurements](gs3d-batch-views.md).

`aaa.ops` and `gs3d.ops` additionally expose independently differentiable
`prepare_gaussians`, `bin_and_sort` and `rasterize_to_pixels` operations. These
reuse the same method kernels and let callers render multiple attribute sets
from unchanged geometry and intersections.

Original 3DGS uses a Warp-generated projection adjoint with explicit custom
adjoints for CUDA's inverse-covariance regularization and clamped-Jacobian
convention. Local VJPs take camera matrices as values and reduce camera
contributions within each block before its leader atomically updates the final
matrices. No per-Gaussian camera-gradient buffer is allocated; see the
[camera reduction and measurements](gs3d-camera-reduction.md).
Its blending backward is an explicit reverse traversal. AAA keeps
its existing hierarchical replay and analytic adjoints. No reference extension
is used at runtime.

Original 3DGS also preserves GLM's matrix accumulation order, lower-triangle
covariance entry and double-precision viewport intermediates. These rounding
details can change which samples survive the alpha threshold in large scenes.

Original 3DGS prepares image-plane projection matrices from current GPU
`Ks/viewmats` in one batched camera kernel. Projection kernels read focal
lengths directly from GPU K; a small camera VJP preserves pose gradients.
Recoverable intrinsic validation reads back status codes only. See the
[GPU camera implementation and measurements](gs3d-gpu-camera.md).

Original 3DGS generates visible/zero-masked training metadata in one independent
Warp kernel, then uses the shared packing and gradient destinations. Its
`means2d` retains CUDA raster coordinates; see the
[metadata implementation and measurements](gs3d-metadata.md).

Shared CUDA float32 SH0–3 uses one Warp evaluation kernel and one first-order
backward kernel. Local basis derivatives come from Warp's differentiation of
the polynomial; dense coefficient gradients are initialized to zero. Original
3DGS projects geometry first and only evaluates SH on rows with positive radii,
matching CUDA's culling order. AAA's modern/low-level SH calls also use the shared
primitive, without a visibility mask; its legacy preprocessing remains unchanged.
CPU/double standalone SH retains the Torch reference. Camera centers retain the
differentiable matrix inverse, preserving the existing pose gradient semantics.
See the [SH implementation and measurements](gs3d-sh-optimization.md).

The native AAA RGB reference moved to `aaa/render_hierarchical_native.py`
without content changes. Its SHA256 remains
`992b3d65ac51a8ad83f8c854faf4247f90b626c6ec2bf973e0a211353f8da9ed`.

## Contribution statistics

```python
from dsl_splat.statistics import MaskedContributionCount, WeightedContributionSum

result = aaa.rasterize(
    ...,
    statistics=(
        MaskedContributionCount(name="hits", mask=mask),
        WeightedContributionSum(name="score", pixel_weights=weights),
    ),
)
```

Requests use `(H,W)` maps shared by all views or explicit `(...,C,H,W)` maps.
Masks are bool, weights are CUDA float32 constants. The results have shape
`(...,C,N)` even with packed rendering metadata; low-level single-view pixel
rendering returns `(N,)` statistics. Names must be unique within a call.

An accepted contribution has survived alpha thresholding and early-termination
checks and is about to be blended in the method's actual final order. With
`w=T*alpha`, masked counts add one for an accepted masked pixel; weighted sums
add `pixel_weights[p]*w`. Masks never alter image rendering. These statistics
are non-differentiable and are not recollected during backward replay.

The requested formulas specialize into the forward kernel. Without requests,
the ordinary kernel has no statistics arrays or atomics. Cache keys contain
formula choices, not map data, pointers or Gaussian counts.

Training strategies own cross-view/frame accumulation and update their state
when Gaussians are split, cloned or pruned. A current-render-derived pixel map
requires a later traversal; a retained-frame statistics replay API is not part
of this first implementation.

## Current method differences

- AAA retains its adaptive 3D filtering, contribution-based clipping and HIER
  64/8/4 ordering. Its near/far behavior is unchanged.
- Original 3DGS defaults to center-depth clipping at `z>0.2`, matching the CUDA
  reference. Explicit near/far values change that center clipping. Its gsplat
  adapter keeps the gsplat signature's `near_plane=0.01` default.
- Original 3DGS currently supports classic fixed `0.3` covariance dilation;
  its adapter rejects antialiased mode. Both methods support features, SH0-3,
  alpha, pose gradients and signed/absolute screen-space statistics.
- Packed metadata does not imply packed GPU geometry. Intrinsic/background
  gradients, second derivatives and built-in depth outputs are unsupported.

## Verification

Saved benchmark snapshots from before the package rename still use the
`aaa_warp` directory name. The benchmark loaders accept a package directory
independently of its name, so those historical paths remain valid.

```bash
python -m tests.check_modern_interface
python -m tests.check_gs3d
python -m tests.check_contribution_statistics
python -m tests.check_library_interfaces
python -m tests.check_backward --output /tmp/aaa-backward.json
python -m tests.benchmark_refactor --baseline-dir /tmp/aaa-warp-library-baseline/aaa_warp \
    --output /tmp/library-refactor-timing.json
```

CUDA references are separate optional test dependencies:
`aaa_gaussian_rasterization` and `diff_gaussian_rasterization`.
See [refactor validation and measured performance](refactor-validation.md) for
the current test coverage and the new original-3DGS backend's performance limits.
The subsequent [original-3DGS bottleneck analysis](gs3d-performance.md) separates
host submission, GPU kernels and the optional pose-gradient hotspot.
