# Minimal 3DGS training demo

This demo primarily shows how to use DSL Splat's rendering interfaces in a
training loop. **The training parameters are examples and are not necessarily
optimal; tune them for your own dataset and task.**

Start with [train_3dgs.py](train_3dgs.py) for rendering calls, optimization,
densification/pruning, and PLY export. Gaussian initialization is in
[model.py](model.py), and COLMAP data loading is in
[colmap_dataset.py](colmap_dataset.py).

## Installation

Use a CUDA-enabled PyTorch environment as described in the
[project installation instructions](../README.md#installation). From the
repository root, install DSL Splat and the demo's additional dependencies:

```bash
python -m pip install numpy opencv-python scikit-learn tqdm tyro plyfile
```

## Usage

The demo expects a COLMAP scene with `PINHOLE` cameras and the binary
reconstruction in `sparse/0/`. For original images stored in `images/`, run
from the repository root:

```bash
python demo/train_3dgs.py --data-root /path/to/scene --im-stride 1
```

With `--im-stride 8`, the loader prefers downsampled images in `images_8/`.
If that directory is missing, it loads `images/` and resizes each image while
caching it in memory. Intrinsics are scaled to the actual output dimensions;
no downsampled files are written to disk. This applies to all training demos.
The default backend is original 3DGS; `--rasterizer-type aaa` selects AAA and
`--rasterizer-type geer` selects GEER. The trained Gaussians are saved to
`splat.ply` by default; change `--ply-save-path` to choose another output path.

See the script's `Config` dataclass and `--help` for command-line options:

```bash
python demo/train_3dgs.py --help
```

## PGSR-style training

[train_pgsr.py](train_pgsr.py) adds plane flattening, single-view normal
consistency, and multi-view geometry/NCC supervision. Its training outline
follows PGSR; plane geometry and losses live in
[`dsl_splat.pgsr`](../dsl_splat/pgsr/README.md), including the RaDe-GS-style
[Warp patch NCC](../dsl_splat/pgsr/patch_ncc.py). It uses DSL Splat's existing
3DGS, AAA or GEER renderer and DefaultStrategy, rather than reproducing every
PGSR training option.

```bash
python -m demo.train_pgsr --data-root /path/to/scene --im-stride 8
```

**Although the PGSR demo supports GEER, its use is currently not recommended:
PGSR's flatten loss can shrink the minimum Gaussian scale until GEER's 3D
filter compensation makes the Gaussian invisible.**

Set `--debug-save-path /path/to/debug` to save a PGSR debug grid every 200
steps, including step 0. The top row contains ground truth, rendered RGB,
rendered normals, and plane distance; the bottom row contains multi-view
geometry weights, plane depth, depth normals, and image-gradient weights.

Example debug output from the garden scene:

![PGSR-style debug grid for the garden scene](assets/pgsr_garden_debug_example.jpg)

## RaDe-GS training

[train_radegs.py](train_radegs.py) uses the native RaDe renderer's expected and
median Z depth and camera-space normals. It trains RGB with L1/SSIM, adds cosine
consistency between rendered and depth-derived normals, and reuses the
multi-view geometry/Warp NCC helpers. It does not apply PGSR's flatten loss. Note it
uses `DefaultStrategy` instead of upstream's GOF/quantile densification.

```bash
python -m demo.train_radegs --data-root /path/to/scene --im-stride 8 \
    --ply-save-path radegs.ply --debug-save-path debug/radegs
```
