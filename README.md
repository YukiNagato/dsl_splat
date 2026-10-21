# DSL Splat

This is a personal learning project for exploring domain-specific languages
(DSLs), primarily **NVIDIA Warp**, using **3D Gaussian Splatting** as a starting
point. I use it to study how Gaussian projection, sorting, rendering and
differentiation can be implemented with a GPU DSL and managed from Python.

I have also worked to bring performance as close as practical to the reference
CUDA implementations, and have started using this code in my own projects.

## Implemented methods

| Method | Implementation |
| --- | --- |
| [Original 3DGS](dsl_splat/gs3d/) | Gaussian projection, tile-based depth sorting, alpha blending, and backward propagation. |
| [AAA-Gaussians](dsl_splat/aaa/) | Adaptive 3D antialiasing, 3D Gaussian evaluation, StopThePop hierarchical sorting, and backward propagation. |
| [3DGEER](dsl_splat/geer/) | PBF association, exact ray–Gaussian evaluation, radial sorting, and backward propagation for pinhole, BEAP and Kannala–Brandt cameras. Initial implementation; performance is still being evaluated. |
| [RaDe-GS](dsl_splat/rade/) | 3DGS-style splatting with radial sorting, ray-plane expected/median depth, camera-space normals, and backward propagation. |
| [PGSR](dsl_splat/pgsr/) | Plane geometry, flattening and normal consistency losses, and multiview geometry/NCC supervision using the 3DGS or AAA renderer. |

The 3DGS, AAA and RaDe renderers support RGB and SH0–3 rendering, alpha blending of arbitrary
feature channels, first-order Gaussian/feature/camera-pose gradients, and
screen-space gradient metadata for densification, including absolute gradients.
They accept OpenCV-style world-to-camera `viewmats` and pinhole intrinsics `Ks`.

Each renderer has its own rendering API and low-level operations:

```python
from dsl_splat import aaa, gs3d, geer, rade

result = gs3d.rasterize(...)
result = aaa.rasterize(...)
result = geer.rasterize(...)
result = rade.rasterize(...)
```

## gsplat-style interfaces

I frequently use splatting in my work and want to reuse code written around
gsplat's calling convention. Each renderer therefore provides a separate
gsplat-style `rasterization` adapter:

```python
from dsl_splat.gs3d.gsplat import rasterization  # Original 3DGS
# Or select AAA-Gaussians:
# from dsl_splat.aaa.gsplat import rasterization
# from dsl_splat.geer.gsplat import rasterization
# from dsl_splat.rade.gsplat import rasterization
```

**This is not an implementation of gsplat.** The adapters support a subset of
its interface; rendering and gradients come from the selected method
implementation. Supported options and numerical behavior depend
on the method. The gsplat package is not required for rendering.

## Installation

Use **Python 3.10+** and an NVIDIA GPU. First install a CUDA-enabled PyTorch
build suitable for your environment, then install from the repository root:

```bash
python -m pip install -e .
```

The declared runtime dependencies are `torch>=2.0` and `warp-lang>=1.17`.
The distribution is named `dsl-splat`, and the Python import name is
`dsl_splat`.


## Minimal training demo

See [demo/README.md](demo/README.md) for a minimal 3DGS training example and its
setup instructions. The demo focuses on API usage; its training parameters may
need tuning for your dataset and task.

## Licenses and attribution

See [NOTICE.md](dsl_splat/NOTICE.md) for
the renderer licenses and upstream attribution. The strategy code adapted from
gsplat v1.4.0 has its own [source and license notice](dsl_splat/strategy/README.md)
and [Apache-2.0 license](dsl_splat/strategy/LICENSE).
