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
python -m pip install -e .
python -m pip install numpy opencv-python scikit-learn tqdm tyro plyfile
```

## Usage

The demo expects a COLMAP scene with `PINHOLE` cameras and the binary
reconstruction in `sparse/0/`. For original images stored in `images/`, run
from the repository root:

```bash
python demo/train_3dgs.py --data-root /path/to/scene --im-stride 1
```

Use `--im-stride 8` if your scene provides downsampled images in `images_8/`.
The default backend is original 3DGS; `--rasterizer-type aaa` selects AAA and
`--rasterizer-type geer` selects GEER. The trained Gaussians are saved to
`splat.ply` by default; change `--ply-save-path` to choose another output path.

Both training scripts use seed `42` by default for Python, NumPy and PyTorch
(CPU and CUDA), set before data loading and model initialization. Pass
`--seed 123` to use a different seed.

Both scripts can pause cloning, splitting and pruning after opacity resets:

- `--pause-refine-after-reset None` (default), or `0`: no pause.
- `--pause-refine-after-reset 200`: pause for the next 200 training iterations.
- `--pause-refine-after-reset auto`: use `len(dataset)` iterations, one epoch
  in these single-image training loops.

Normal optimization and densification-statistic accumulation continue during
the pause. A reset at step 1000 with a 200-iteration pause freezes refinement
at steps 1001–1200; it resumes at the first scheduled `refine_every` step after
1200. Refinement on the reset step still happens before the reset. Each further
reset restarts the pause; a pause at least as long as `reset_every` prevents
refinement between resets. When using `DefaultStrategy` directly, pass an
integer or `None`; resolve `"auto"` to `len(dataset)` in the training code.

See the script's `Config` dataclass and `--help` for command-line options:

```bash
python demo/train_3dgs.py --help
```

Review the learning rates, densification/pruning thresholds and schedules in
`Config`, as well as the `scene_scale` passed to `strategy.initialize_state()`
in the training script. These settings may need adjustment for your scene.

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

Both demos accept `--rasterizer-type geer`, for example:

```bash
python -m demo.train_3dgs --data-root /path/to/scene --im-stride 8 --rasterizer-type geer
python -m demo.train_pgsr --data-root /path/to/scene --im-stride 8 --rasterizer-type geer
```

GEER blends the same extra plane attributes used by the PGSR demo. PGSR's
camera-Z depth is recovered from those attributes; it does not use GEER's
inverse radial depth. GEER's densification gradient is a canonical-space proxy,
so `grow_grad2d` may need different tuning from 3DGS/AAA. See the
[GEER gradient conventions](../dsl_splat/geer/README.md#gradients-and-metadata).

`DefaultStrategy` uses GEER's proxy directly, without the pixel-gradient
`width/2`, `height/2` normalization used for 3DGS/AAA. Earlier versions applied
that normalization to GEER as well, greatly increasing growth scores.

To control growth, increase `--grow-grad2d` (affects both cloning and splitting),
or increase `--refine-every` to grow less often. `--grow-scale3d` separates
clone candidates from split candidates; it is not a limit on total growth.
Both demos accept `--max-gs-num 500000` to cap growth. Existing Gaussians above
that budget are not automatically pruned. Leave headroom for temporary
split/optimizer allocations and tile intersections;
the count limit is not a GPU-memory limit. GEER's original trainer uses a
300-step densification interval; this is a useful starting point to try with
`--refine-every 300`, rather than the demos' 100-step default.

Images are RGB HWC, and camera poses are ordinary OpenCV world-to-camera
matrices. Rendered plane attributes come from `info["render_extra_signals"]`.
The renderer uses half-integer pixel centers; geometry and NCC use integer
array indices with a matching principal-point adjustment inside this demo.

View supervision starts after step 7000 by default. Neighbor distance limits
use the reconstruction's world units; inspect the `pgsr_*` options and tune
them for your scene. This script also primarily demonstrates usage: its
training parameters are examples, not established optimal settings.

NCC defaults to PGSR's one-pixel patch spacing
(`pgsr_multi_view_half_step=False`). Enable `--pgsr-multi-view-half-step` to
use RaDe-GS's half-pixel spacing instead.

The demo enables `torch.compile` for Gaussian plane attributes, depth/normal
recovery, and dense multiview reprojection. Initial calls include compilation
time. Use `--no-pgsr-compile` to run those blocks in eager mode. Image weights,
dynamic patch selection, Warp rendering/NCC, and debug I/O stay outside compilation.

Set `--debug-save-path /path/to/debug` to save a PGSR debug grid every 200
steps, including step 0. The top row contains ground truth, rendered RGB,
rendered normals, and plane distance; the bottom row contains multi-view
geometry weights, plane depth, depth normals, and image-gradient weights.
Before multi-view supervision starts, or when no neighbor is available,
the geometry weights are zero. Filenames include the step and frame name.

Example debug output from the garden scene:

![PGSR-style debug grid for the garden scene](assets/pgsr_garden_debug_example.jpg)
