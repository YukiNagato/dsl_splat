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
| `default.py` | [`gsplat/strategy/default.py`](https://github.com/nerfstudio-project/gsplat/blob/4d3a3b69db4de0326f983ccf7b7b255271a17b01/gsplat/strategy/default.py) | Adds an optional `max_gs_num` growth limit with joint random sampling of duplicate/split candidates. Uses GEER's canonical gradient proxy without projected-mean screen scaling and passes renderer filter metadata to opacity resets. Accepts `None` to disable the post-reset refinement pause and tracks actual opacity resets to define its window. Provides a post-growth reset-policy hook for methods with per-Gaussian filters. |
| `ops.py` | [`gsplat/strategy/ops.py`](https://github.com/nerfstudio-project/gsplat/blob/4d3a3b69db4de0326f983ccf7b7b255271a17b01/gsplat/strategy/ops.py) | Selected helpers/imports removed, optimizer helper renamed, opacity reset optionally accounts for a 3D filter's determinant ratio, with scalar or per-Gaussian variance. MCMC operations now live in `mcmc_ops.py`. |
| `mcmc.py` | [`gsplat/strategy/mcmc.py`](https://github.com/nerfstudio-project/gsplat/blob/4d3a3b69db4de0326f983ccf7b7b255271a17b01/gsplat/strategy/mcmc.py) | Same opacity-weighted relocation, 5% growth schedule and noise policy; uses Warp numerical operations. Adds validation, empty/all-dead handling and an optional noise cutoff. |
| `mcmc_ops.py`, `_mcmc_kernels.py` | [`gsplat/strategy/ops.py`](https://github.com/nerfstudio-project/gsplat/blob/4d3a3b69db4de0326f983ccf7b7b255271a17b01/gsplat/strategy/ops.py), [`gsplat/relocation.py`](https://github.com/nerfstudio-project/gsplat/blob/4d3a3b69db4de0326f983ccf7b7b255271a17b01/gsplat/relocation.py) and Equation 9 of the MCMC paper | Python population management and native-free Warp relocation/noise kernels. Replaces the incomplete relocation helper previously in `ops.py`. Differences are documented below. |
| `__init__.py` | Local exports | Exports `DefaultStrategy` and `MCMCStrategy`. |

`DefaultStrategy` uses PyTorch. `MCMCStrategy` uses PyTorch for sampling and
parameter/optimizer bookkeeping, and Warp for scale/opacity redistribution and
position noise. The latter fuses quaternion normalization, opacity gating and
covariance application without allocating an `[N,3,3]` covariance tensor. It
uses **covariance times noise**, following upstream MCMC, rather than a
covariance square root.

## MCMC usage

```python
from dsl_splat.strategy import MCMCStrategy

strategy = MCMCStrategy(cap_max=1_000_000)
strategy.check_sanity(params, optimizers)
state = strategy.initialize_state()

for step in range(train_iters):
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    image, alpha, info = rasterization(...)
    loss = photometric_loss(image, target)
    # These regularizers belong in the differentiable training loss.
    loss += 0.01 * params["opacities"].sigmoid().mean()
    loss += 0.01 * params["scales"].exp().mean()
    loss.backward()
    for optimizer in optimizers.values():
        optimizer.step()
    strategy.step_post_backward(
        params, optimizers, state, step, info,
        lr=optimizers["means"].param_groups[0]["lr"],
    )
```

All parameter tensors have a leading Gaussian dimension. `means [N,3]`, log
`scales [N,3]`, wxyz `quats [N,4]` and opacity **logits** `[N]` or `[N,1]` must
be float32 on the same CPU/CUDA device. Extra SH/features are copied during
relocation and growth, preserving their dtype and `requires_grad`. Every
parameter, including frozen ones, needs its own single-parameter optimizer,
following the existing strategy convention. Standard Adam/SGD per-row states
are supported; optimizer-wide scalar `step` counters are preserved. Call the
strategy after the optimizer updates, before the next forward pass. It does
not consume projected-mean gradients or reset opacity periodically.

Defaults follow gsplat v1.4.0: refine every 100 steps strictly between steps
500 and 25,000, relocate opacity `<=0.005`, grow by `floor(1.05*N)-N` capped at
one million, and perturb with `means_lr * 5e5`. Populations below 20 do not grow
under that rounding rule. `cap_max` must be at least the initial population;
this is a growth cap, not an implicit pruning request. Noise continues after
refinement stops; set `noise_stop_iter` for an exclusive cutoff. Calling
`torch.manual_seed` controls both parent selection and perturbations; this
does not promise identical random streams across devices or PyTorch versions.

Implementation differences from the old gsplat operations:

- Clamp the new opacity **before** computing the scale correction, matching
  newer gsplat revision
  [`512d366b67073d77ca099ede742683c165dfc23b`](https://github.com/nerfstudio-project/gsplat/commit/512d366b67073d77ca099ede742683c165dfc23b),
  `gsplat/cuda/csrc/RelocationCUDA.cu` (gsplat contributors and NVIDIA, Apache-2.0).
  The opacity floor intentionally relaxes Equation 9's transparency invariant.
- Evaluate the alternating binomial sum in float64, and cap saturated input
  opacity at `1-float32_eps` when deriving the new opacity. This avoids severe
  cancellation for high multiplicities and keeps opacity-one donors finite.
  Outputs and the per-step noise kernel remain float32. Multiplicity is capped
  at 51, as in gsplat; above this cap redistribution is an approximation.
- Update repeated donors once. Relocation retains Parameter objects and clears
  both donor and replaced destination optimizer moments; the old helper only
  cleared donor moments. Growth replaces Parameters, preserves existing moments
  and initializes appended moments to zero, following gsplat.
- Empty populations and populations with no live donor skip refinement/growth.
  For over `2^24` sampling categories, use a device-side float64 CDF and Torch's
  RNG instead of falling back to NumPy/CPU.
- Use Torch's current stream for all Warp launches; do not call
  `torch.cuda.empty_cache()` during refinement. The strategy state contains only
  the small binomial table and last-refinement counts, with no saved Gaussian
  pointers or temporary population buffers.

Validation: `python -m tests.check_strategy_mcmc` checks Equation 9, noise,
optimizer alignment, deterministic seeded runs, empty/all-dead populations,
non-default CUDA streams and short training with 3DGS, AAA, GEER and RaDe.

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
