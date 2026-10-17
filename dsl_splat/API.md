# Modern AAA API

The canonical entry is `dsl_splat.aaa.rasterize`; `dsl_splat.rasterize` remains
its compatibility export. Original 3DGS has an independent `dsl_splat.gs3d` API
and `dsl_splat.gs3d.gsplat` adapter. See the repository's
[library architecture](../docs/architecture.md) for shared components and
optional per-Gaussian contribution statistics.

The main API uses OpenCV cameras: rigid world-to-camera transforms, X right,
Y down, Z forward, and zero-skew pinhole K with arbitrary principal points.
Inputs are CUDA float32. Scales/opacities are activated values; quaternions
use wxyz and are normalized differentiably.

## Complete rendering

```python
from dsl_splat import rasterize

result = rasterize(
    means, quats, scales, opacities, viewmats, Ks, width, height,
    features=features, absgrad=True,
)
loss = image_loss(result.features) + mask_loss(result.alpha)
loss.backward()
# features: (...,C,H,W,D); alpha: (...,C,H,W,1), differentiable
# radii: (...,C,N,2), with packed=False (default)
# result.means2d.grad / result.means2d.absgrad: screen translation statistics
```

Features have shape `(...,N,D)` or `(...,C,N,D)`. For SH, pass
`sh_coeffs=coefficients` instead; coefficients have shape `(...,N,K,3)` or
`(...,C,N,K,3)`. `sh_degree` selects degree 0–3 or defaults to the highest
complete degree up to 3. `extra_features` adds independently rendered attributes
available as `result.extra_features`. Backgrounds have shape `(...,C,D)`.

Defaults enable AAA 3D evaluation, adaptive antialiasing with opacity
compensation, HIER 64/8/4, maximum-contribution tile depth, tight rectangle/tile
culling and load balancing. Customize with `default_settings()` or
`ExtendedSettings.from_json(...)`; settings are snapshotted per frame.

Gaussian batch dimensions and multiple cameras are supported. Cameras execute
sequentially. Calls accept changing N, fresh allocations and strided inputs.
Each call owns its outputs. There is no fixed-address contract.

Image outputs use channel-last layouts for both single and multiple cameras.
Features, alpha and extra features may be views of a shared per-call image
buffer; individual fields need not be contiguous. Ordinary Torch operations
work directly. Use `.reshape(...)` for general reshaping, or `.contiguous()`
when an external consumer requires contiguous storage. In-place image
processing follows normal Torch autograd rules.

`packed=True` packs training metadata only; rendering buffers remain dense.
`result.meta` retains the training metadata contract. Means2d gradients measure
rigid translation of the AAA footprint with sorting/culling fixed; absgrad is
the sum of absolute per-pixel derivatives. Neither statistic is a second route
for propagating Gaussian parameter gradients.

## Composable stages

```python
from dsl_splat import PinholeCamera, ops

camera = PinholeCamera(viewmat, K, width, height)
geometry = ops.prepare_gaussians(means, quats, scales, opacities, camera)
tiles = ops.bin_and_sort(geometry)
rgb = ops.rasterize_to_pixels(geometry, tiles, colors, rgb_background)
attributes = ops.rasterize_to_pixels(geometry, tiles, extra_attributes)
loss = image_loss(rgb.features) + attribute_loss(attributes.features)
loss.backward()
```

These stages operate on one camera and one Gaussian set. Both renders share
geometry and sorted intersections and may have different channel counts and
backgrounds. Reuse requires unchanged camera, geometry, opacity, filtering and
settings. Joint losses propagate both renders' gradients through preparation;
separate backward calls need Torch's normal `retain_graph=True` handling.

`GaussianGeometry` exposes differentiable `ray_transforms` and effective
`opacities`. Its bounds, visibility and tile membership are discrete outputs.
Only rows with radii > 0 have defined geometry; rejected rows are not cleared.
Mask invalid rows in direct geometry losses. Bounding centers are AAA bounds,
not gsplat projected centers. `GaussianTiles` exposes ranges and sorted
Gaussian IDs. `PixelResult` exposes `(H,W,D)` features and `(H,W,1)` alpha.
The low-level path does not construct densification statistics.

`ops.spherical_harmonics(coefficients, directions, degree)` evaluates real RGB
SH with first-order gradients through coefficients and directions, and AAA's
+0.5 offset/lower clamp. CUDA float32 uses fused Warp kernels; CPU/double
inputs retain the Torch implementation for numerical work.

## Compatibility and execution

`dsl_splat.aaa.gsplat.rasterization` retains the gsplat signature and
`(colors, alphas, meta)` result. Its `classic` and `packed=True` defaults remain
unchanged; the modern API defaults to AAA antialiasing and dense metadata.
See [the compatibility contract](GSPLAT_COMPAT.md).

`GaussianRasterizer` / `GaussianRasterizationSettings` remain for original
3DGS/AAA callers. Both adapters use the core; the modern path never constructs
or calls the legacy rasterizer.

| Module | Responsibility |
| --- | --- |
| `aaa/api.py` | Modern AAA entry, batches and typed result |
| `cameras.py`, `aaa/cameras.py` | Public OpenCV cameras and AAA-specific lowering |
| `inputs.py`, `aaa/metadata.py`, `training.py` | Layout checks, SH and frame-owned statistics |
| `aaa/pipeline.py`, `runtime/` | Execution context and host launch-packet ownership |
| `aaa/autograd.py` | One complete-render Function for training/inference |
| `aaa/ops.py` | Independently differentiable preparation and pixel rendering |
| `aaa/gsplat.py`, `aaa/legacy.py` | Method-local compatibility signatures/translations |

Both autograd paths reuse the same analytic kernels. Complete rendering retains
one Function and phase-local descriptor reuse. Internal `FrameConfig` holds
lowered kernel parameters, not a public camera convention. Lowering currently
retains a small CPU read of K and the established arithmetic for numerical parity.

Near/far retain AAA's frontend projection and contribution-based culling,
including its effective lower clip bound near near/2, rather than strict
Gaussian-center clipping. First-order Gaussian and pose gradients are supported.
Intrinsic/background gradients, second derivatives, depth rendering,
non-pinhole cameras and sparse/distributed rendering are unsupported.

## Validation

```bash
python -m tests.check_modern_interface
python -m tests.check_gsplat_interface
python -m tests.check_gsplat_screen_grad
python -m tests.check_camera_grad
python -m tests.check_backward --output /tmp/backward.json
python -m tests.benchmark_refactor --baseline-dir /path/to/saved/aaa_warp \
    --output /tmp/refactor-timing.json
```
