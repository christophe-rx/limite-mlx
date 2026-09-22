# LM Studio: why `install.sh` does what it does

`./install.sh` handles all of this. This page explains the three gates it
clears, because each produces a misleading error if you hit it by hand.

## 1. MLX *format*, not just MLX-loadable

LM Studio picks a runtime from the **safetensors metadata**, before it ever
calls mlx-lm. The official upload is PyTorch-saved and carries
`{"format": "pt"}`, which LM Studio classifies as `torchSafetensors` — a format
with no Mac runtime:

> 🥲 Failed to load the model
> No LM Runtime found for model format 'torchSafetensors'!

`mlx_lm.convert` re-saves with `{"format": "mlx"}`, which routes it to the MLX
runtime. Converting also folds the per-layer projection scales once and drops
them from the checkpoint (`sanitize` is a no-op on the result, so nothing gets
double-folded).

## 2. An mlx-lm new enough to do that conversion

The `model_file` hook arrived in **mlx-lm 0.31.0**, which requires
`transformers>=5`, which requires **Python >= 3.10**. On an older interpreter
the resolver silently falls back to a pre-0.31 mlx-lm, and the failure looks
like it is about the architecture rather than the version:

> ValueError: Model type limite not supported.

`install.sh` builds a pinned environment (Python 3.12, `mlx-lm>=0.31`) and
verifies the resolved version before converting. Note that `uv run --with
mlx-lm` alone is not enough — uv may pick the 3.9 CommandLineTools Python and
quietly resolve an old mlx-lm.

## 3. Custom-code permission

mlx-lm can load an architecture from the model folder via the `model_file`
config key, but newer versions gate that behind a `trust_remote_code=True`
**function argument**:

> ValueError: The model at ... requires importing and running a custom module
> ('limite.py') to build its architecture. This is disabled by default.

LM Studio calls `mlx_lm.utils.load(self.model_path)` with no such argument
(`mlx_engine/model_kit/model_kit.py`), and there is **no config-level opt-in** —
`trust_remote_code` is a parameter, not a config key. So inside LM Studio the
architecture cannot travel with the model, however the folder is arranged.

`install.sh` therefore puts the module where mlx-lm keeps *built-in*
architectures — `mlx_lm/models/limite.py` inside LM Studio's bundled runtime —
which needs no permission at all, and removes `model_file` from the installed
model so no custom code is requested.

## Why this needs re-running

LM Studio bundles its own copy of mlx-lm. Updating the MLX runtime replaces
that copy and takes `limite.py` with it, and the model starts failing with
`Model type limite not supported` again. Re-run `./install.sh`; it skips the
download and conversion if they are already done, so it is quick.

The only way to remove this step permanently would be for the architecture to
ship in mlx-lm itself.

## Outside LM Studio

None of gate 3 applies. `run.py`, `eval.py` and `mlx_lm.load()` in your own
code all accept the architecture from the model folder via `model_file`, which
is what the README's command-line section uses.
