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
The default backend is original 3DGS; `--rasterizer-type aaa` selects the AAA
adapter. The trained Gaussians are saved to `splat.ply` by default; change
`--ply-save-path` to choose another output path.

See the script's `Config` dataclass and `--help` for command-line options:

```bash
python demo/train_3dgs.py --help
```

Review the learning rates, densification/pruning thresholds and schedules in
`Config`, as well as the `scene_scale` passed to `strategy.initialize_state()`
in the training script. These settings may need adjustment for your scene.
