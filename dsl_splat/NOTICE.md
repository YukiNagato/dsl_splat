# License

This project is derived from [StopThePop-Rasterization](https://github.com/r4dl/StopThePop-Rasterization), which is based on [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization).

The `gs3d` method ports the original Inria rasterizer's projection, bounds,
compositing and backward conventions to NVIDIA Warp. Its reference source is
Copyright (C) 2023, Inria / GRAPHDECO. The package's `LICENSE.md` includes the
Gaussian-Splatting license applying to this derivative work.

Gaussian-Splatting is licensed under a custom non-commercial research license.  
This project includes additional modifications by Thomas Köhler, Michael Steiner and contributors (c) 2025.

This derivative work remains under the same research-only usage restrictions.

The contents of the files ```cuda_rasterizer/utils.*``` are based on publicly available code from [3D Gaussian Splatting as Markov Chain Monte Carlo](https://github.com/ubc-vision/3dgs-mcmc) authored by Shakiba Kheradmand and contributors, which falls under the same research-only usage restrictions. 

See `LICENSE.md` for details.

# Files subject to permissive licenses
The contents of the files ```cuda_rasterizer/stopthepop``` are based on publicly available code authored by Lukas Radl, Michael Steiner and contributors, which falls under the permissive MIT license. 

Title: StopThePop\
Project code: https://github.com/r4dl/StopThePop\
Copyright Graz University of Technology, 2024\
License: https://github.com/r4dl/StopThePop-Rasterization/blob/main/cuda_rasterizer/stopthepop/LICENSE.txt (MIT)

## gsplat strategy code (Apache-2.0)

The gsplat-derived files in `strategy/` are copied or adapted from
[gsplat v1.4.0](https://github.com/nerfstudio-project/gsplat/tree/v1.4.0/gsplat/strategy),
commit `4d3a3b69db4de0326f983ccf7b7b255271a17b01`, by the gsplat authors and
contributors. These files retain their upstream Apache License 2.0 terms.
The component's full license is included in [strategy/LICENSE](strategy/LICENSE);
[strategy/README.md](strategy/README.md) records the source files and local
adaptations. This component-specific license does not replace the renderer
license described above or relicense the rest of this package.

## PGSR plane supervision and RaDe-GS NCC

The `pgsr/` geometry and loss helpers adapt PGSR and RaDe-GS. Their source
revisions, adapted files, and local changes are recorded in
[pgsr/README.md](pgsr/README.md). The PGSR authors' terms are retained in
[pgsr/PGSR_LICENSE.txt](pgsr/PGSR_LICENSE.txt); RaDe-GS's Gaussian-Splatting
license is included in [LICENSE.md](LICENSE.md).

## Fused SSIM

The Warp SSIM metric adapts the filtering structure and coefficients from
Rahul Goel's fused-ssim (MIT, 2024). Source revision and local changes are
recorded in [metrics/README.md](metrics/README.md); the upstream terms are
retained in [metrics/FUSED_SSIM_LICENSE.txt](metrics/FUSED_SSIM_LICENSE.txt).

## 3DGEER

The `geer/` component adapts Bosch Research's 3DGEER PBF and ray-evaluation
rasterizer. Source revision, adapted files, and local changes are recorded in
[geer/README.md](geer/README.md). Upstream's root GNU AGPLv3 license is preserved
in [geer/GEER_LICENSE.txt](geer/GEER_LICENSE.txt); the rasterizer subdirectory's
Inria/MPII Gaussian-Splatting license is preserved in
[geer/GEER_RASTERIZER_LICENSE.txt](geer/GEER_RASTERIZER_LICENSE.txt).
These component-specific notices do not replace the upstream license texts
or relicense other components.

## Speedy-Splat tile localization

`splatting/intersections.py` adapts SnugBox and AccuTile from
[Speedy-Splat](https://github.com/j-alex-hanson/speedy-splat), by Alex Hanson,
Allen Tu, Geng Lin, Vasu Singla, Matthias Zwicker and Tom Goldstein.
Reference: `speedy-splat-rasterizer`, revision
`b38abb6da7a7217999f8c5ec0e9219284464ffe0`, `cuda_rasterizer/auxiliary.h`.
The reference is derived from Inria/GRAPHDECO's rasterizer and retains the
Gaussian-Splatting research-use license included in [LICENSE.md](LICENSE.md).
This port uses Python/Warp, shares count/emission traversal, retains legacy
3DGS candidate limits and radii, and conservatively expands the alpha cutoff
for float32 rounding. It does not implement Speedy-Splat's pruning strategy.

## RaDe-GS

The `rade/` renderer ports HKUST-SAIL's RaDe-GS CUDA rasterizer to Warp.
It shares the Inria-derived projected-splat components in `splatting/` with
`gs3d/`. Source revision, implemented stages and numerical conventions are
recorded in [rade/README.md](rade/README.md). The upstream rasterizer license is
preserved in [rade/RADE_LICENSE.txt](rade/RADE_LICENSE.txt).
