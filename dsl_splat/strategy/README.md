# Strategy code: source and license

This directory contains selected strategy code copied or adapted from
[gsplat](https://github.com/nerfstudio-project/gsplat), maintained by the gsplat
authors and contributors credited in its
[upstream README](https://github.com/nerfstudio-project/gsplat/blob/v1.4.0/README.md).

## Upstream version

- Release: **v1.4.0**.
- Commit: [`4d3a3b69db4de0326f983ccf7b7b255271a17b01`](https://github.com/nerfstudio-project/gsplat/commit/4d3a3b69db4de0326f983ccf7b7b255271a17b01).
- Source directory: [`gsplat/strategy/`](https://github.com/nerfstudio-project/gsplat/tree/4d3a3b69db4de0326f983ccf7b7b255271a17b01/gsplat/strategy).

| Local file | Upstream source | Local changes |
| --- | --- | --- |
| `base.py` | [`gsplat/strategy/base.py`](https://github.com/nerfstudio-project/gsplat/blob/4d3a3b69db4de0326f983ccf7b7b255271a17b01/gsplat/strategy/base.py) | Body unchanged; final newline differs. |
| `default.py` | [`gsplat/strategy/default.py`](https://github.com/nerfstudio-project/gsplat/blob/4d3a3b69db4de0326f983ccf7b7b255271a17b01/gsplat/strategy/default.py) | Adds an optional `max_gs_num` growth limit with joint random sampling of duplicate/split candidates. Uses GEER's canonical gradient proxy without projected-mean screen scaling and passes renderer filter metadata to opacity resets. Accepts `None` to disable the post-reset refinement pause and tracks actual opacity resets to define its window. |
| `ops.py` | [`gsplat/strategy/ops.py`](https://github.com/nerfstudio-project/gsplat/blob/4d3a3b69db4de0326f983ccf7b7b255271a17b01/gsplat/strategy/ops.py) | Locally edited subset: imports and helpers removed, optimizer helper renamed, opacity reset optionally accounts for a 3D filter's determinant ratio; adaptation is in progress. |
| `__init__.py` | Local package marker | Does not copy upstream's strategy exports. |

The current code uses PyTorch. Warp implementations of MCMC and ImprovedGS
strategies are planned additions; this import does not include those
implementations. Record further adaptations and their sources alongside the
affected files.

## License

The gsplat-derived code listed above is provided under **Apache License 2.0**
(`SPDX-License-Identifier: Apache-2.0`). [LICENSE](LICENSE) is an unmodified copy
of the [upstream license at the pinned commit](https://github.com/nerfstudio-project/gsplat/blob/4d3a3b69db4de0326f983ccf7b7b255271a17b01/LICENSE).
Upstream authorship and applicable attribution notices are retained; the local
copy does not transfer authorship to DSL Splat.

When redistributing these files or derivatives, include the Apache-2.0 license,
retain applicable upstream notices, and mark modified files as required by the
license. The provenance headers in `default.py` and `ops.py` identify local
modifications.

This component's license is separate from the Gaussian-Splatting research
license used by the renderer code in [../LICENSE.md](../LICENSE.md). Adding
gsplat strategy code does not change the license of the rest of DSL Splat.
See [../NOTICE.md](../NOTICE.md) for the package's third-party attributions.
