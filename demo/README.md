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

## MCMC strategy

`train_3dgs.py` can use the renderer-independent Warp MCMC strategy:

```bash
python -m demo.train_3dgs --data-root /path/to/scene --im-stride 8 \
    --strategy-type mcmc --max-gs-num 1000000 --refine-stop-iter 25000
```

## Improved-GS training

[train_3dgs.py](train_3dgs.py) supports Improved-GS EAS, LAS, RAP, MU and
Gaussian-budget growth control via `--strategy-type improvedgs`. It uses the shared `Trainer`;
[improvedgs.py](improvedgs.py) handles sampled-view no-grad EAS inference. The
strategy receives scores and owns no renderer or dataset. It excludes MiniGS
and the reference repository's other methods.

```bash
python -m demo.train_3dgs --data-root /path/to/scene --im-stride 8 \
    --strategy-type improvedgs --max-gs-num 1000000 --ply-save-path improvedgs.ply
```

Select `--rasterizer-type aaa` for AAA, or keep the 3DGS default.
`--improvedgs.score-views -1` scores all training views. MU uses
single-view updates through 15k, then 5-view and 20-view summed-gradient
updates, switching at 22.5k. When shortening densification, adjust the
budget/late-stage windows and MU thresholds too. The final partial batch is
flushed.

Model and optimizer settings use the same flags for all strategies. For an
Improved-GS example with a decaying position learning rate, add
`--means-lr 0.00004 --means-lr-final 0.000002 --scales-lr 0.005
--opacities-lr 0.025 --sh-degree-interval 1000`. Without `--means-lr-final`,
`train_3dgs.py` keeps the position learning rate constant.

The PGSR and RaDe demos can use the same complete strategy with their own
geometry losses and rendering:

```bash
python -m demo.train_pgsr --data-root /path/to/scene --im-stride 8 \
    --strategy-type improvedgs --max-gs-num 1000000
python -m demo.train_radegs --data-root /path/to/scene --im-stride 8 \
    --strategy-type improvedgs --max-gs-num 1000000
```

All three demos default to `--strategy-type default`.
Improved-GS settings are grouped under `--improvedgs.*`, for example
`--improvedgs.score-views 5`, `--improvedgs.grow-grad2d 0.0003`,
`--improvedgs.prune-opa 0.005`, `--improvedgs.reset-every 3000` and
`--improvedgs.reset-opacity 0.05`. The unprefixed counterparts configure the
default strategy. Shared `--max-gs-num`, `--refine-start-iter`,
`--refine-stop-iter`, `--refine-every`, `--absgrad` and reset-pause settings apply
to either strategy; Improved-GS requires absgrad and uses a 1M budget if no cap
is specified. Model, optimizer and geometry-loss settings retain each demo's
defaults, so this is a combination of methods rather than a paper reproduction.

EAS inference stays outside the strategy and runs after optimizer updates,
before population changes. PGSR supports EAS with the 3DGS/AAA backends; GEER
does not yet expose contribution statistics. RaDe collects statistics from its
own filtered rendering, recomputes filter widths for LAS children, preserves
filter alignment on pruning, and caps filtered opacity during RAP resets. The
shared `Trainer` handles MU accumulation and the final partial update for all
three demos. Improved-GS mode uses steps 1 through `train_iters`; 3DGS and PGSR
retain their existing step-0 schedules for the other strategies.

See [the strategy documentation](../dsl_splat/strategy/IMPROVED_GS.md) for
interfaces, defaults and differences from the current upstream revision. This
is a usage example; tune parameters for your data. Full-scene quality and
paper benchmark reproduction have not been evaluated.

## PGSR-style training

[train_pgsr.py](train_pgsr.py) adds plane flattening, single-view normal
consistency, and multi-view geometry/NCC supervision. Its training outline
follows PGSR; plane geometry and losses live in
[`dsl_splat.pgsr`](../dsl_splat/pgsr/README.md), including the RaDe-GS-style
[Warp patch NCC](../dsl_splat/pgsr/patch_ncc.py). It uses DSL Splat's existing
3DGS, AAA or GEER renderer and DefaultStrategy by default, rather than reproducing every
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
uses `DefaultStrategy` by default instead of upstream's GOF/quantile densification.
`--strategy-type improvedgs` selects the alternative described above.

```bash
python -m demo.train_radegs --data-root /path/to/scene --im-stride 8 \
    --ply-save-path radegs.ply --debug-save-path debug/radegs
```
