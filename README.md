# limite-mlx

Run Paradigma's **Limite 1B** models on Apple Silicon with MLX.

[Limite 1B Violetto](https://huggingface.co/paradigma-inc/limite-1b-violetto) is
a 1B mathematical reasoning model. Its [reference
implementation](https://github.com/paradigma-inc/limite-violetto) is a vLLM
plugin requiring **Linux x86-64 + NVIDIA + CUDA 13**, so it does not run on a
Mac. `limite_mlx.py` is a port of that implementation to MLX — same numerics,
Metal instead of CUDA.

Nothing else loads these models: `model_type: "limite"` is unknown to
transformers, llama.cpp, Ollama and stock mlx-lm.

## Use it with LM Studio

```bash
git clone <this repo> && cd limite-mlx
./install.sh                      # or: ./install.sh limite-1b-base
```

Then restart LM Studio and load `paradigma-inc/limite-1b-violetto-mlx`.

The script downloads the weights (~2 GB), converts them to MLX format,
installs them where LM Studio looks, and installs the architecture into LM
Studio's MLX runtime. It needs `uv` (or any Python with `mlx-lm>=0.31`) and
LM Studio's MLX runtime already present.

**One caveat:** LM Studio bundles its own copy of mlx-lm, so updating its MLX
runtime removes the architecture file. If the model later fails with
`Model type limite not supported`, re-run `./install.sh`. See
[docs/LMSTUDIO.md](docs/LMSTUDIO.md) for why this is necessary.

## Use it from the command line

```bash
uv sync
uv run hf download paradigma-inc/limite-1b-violetto --local-dir models/limite-1b-violetto
cp limite_mlx.py models/limite-1b-violetto/limite.py
python -c "import json;p='models/limite-1b-violetto/config.json';c=json.load(open(p));c['model_file']='limite.py';json.dump(c,open(p,'w'),indent=2)"

uv run python run.py "What is the remainder when 7^100 is divided by 13?"
uv run python eval.py --model models/limite-1b-violetto
```

`model_file` tells mlx-lm to load the architecture from the model folder.
(That works everywhere *except* LM Studio, which is why `install.sh` takes the
other route — see [docs/LMSTUDIO.md](docs/LMSTUDIO.md).)

Violetto reasons inside `<think>`…`</think>` before answering, so give it room
— `--max-tokens 4096` is the default and hard problems want more.

## Models

| checkpoint | loss ↓ | top-1 ↑ | notes |
|---|---|---|---|
| `limite-1b-base` | 3.21 | 42.4% | general base model |
| `limite-1b-violetto` | 6.06 | 26.1% | maths specialist; prose is out of domain |

Held-out English prose, uniform baseline 11.93 nats over the 151680-token
vocab. Violetto scoring worse on prose is expected — it is a reasoning model
with deliberately light instruction tuning, not a general chat model.

~2 GB in bfloat16. 4-bit quantization works (`mlx_lm.convert -q`, 583 MB) but
costs a lot at this size — loss 6.615 → 7.787, top-1 24.7% → 16.4% — so prefer
bf16 unless you are tight on memory.

## The architecture

48 layers, d_model 1280, 10 query / 2 KV heads, head_dim 128, SwiGLU, tied
embeddings, parameter-free RMSNorm, 131072 context. On top of that it combines:

- **XSA** ([arXiv:2603.09078](https://arxiv.org/abs/2603.09078)) — subtracts the
  component of the attention output along the token's own value vector.
- **MUDD** ([arXiv:2502.12170](https://arxiv.org/abs/2502.12170)) — at layers 24
  and 47 the attention input and residual base are input-dependent mixes of
  retained earlier states.
- Value embeddings injected into V at 16 layers, gated from the layer input.
- Every 4th layer is full attention with **no positional encoding**; the rest
  use an inclusive 1024-key sliding window with partial adjacent-pair RoPE.
- Per-head attention output gates, per-layer projection scales, learned
  residual/branch lambdas, and a `23·sigmoid((x+5)/7.5)` logit softcap.

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for the details that bite
when porting.

## Verification

The port is checked against a PyTorch implementation of the same reference
(`tests/reference_torch.py`):

| check | result |
|---|---|
| float32 parity, per layer (all 48) | rel. err ~1e-6, cosine similarity 1.000000 |
| float32 parity, final logits | max diff 1.1e-4, argmax agreement 100% |
| bfloat16 parity | argmax 98%, loss 6.568 vs 6.562 |
| incremental decode vs full forward | bit-exact |
| decode with cache rotation | argmax 100% |
| sliding-window convention (forced to 8) | argmax 100% |

```bash
uv sync --extra dev && uv run pytest tests/ -v
```

The bfloat16 gap is rounding, not different maths — the same comparison in
float32 agrees to ~1e-6 through all 48 layers.

## Licence

Apache-2.0, matching the reference implementation and the model weights. See
`LICENSE` and `NOTICE`. Weights are not redistributed here; they are downloaded
from Hugging Face.
