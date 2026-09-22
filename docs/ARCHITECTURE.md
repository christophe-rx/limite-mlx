# The `limite` architecture, and what bites when porting it

Shapes and names come from `config.json` and the 744-tensor checkpoint; the
semantics follow the [reference vLLM
plugin](https://github.com/paradigma-inc/limite-violetto).

## Shape

48 layers · d_model 1280 · 10 query heads / 2 KV heads (GQA) · head_dim 128 ·
SwiGLU 3328 · tied embeddings · vocab 151680 · parameter-free RMSNorm
everywhere · 131072 context.

## The unusual parts

**XSA — Exclusive Self Attention** ([arXiv:2603.09078](https://arxiv.org/abs/2603.09078)).
Per head, before `o_proj`:

    z = y - alpha_h * (y · v̂) v̂,    v̂ = normalize(v_self)

removing the component of the attention output along the token's *own* value
vector. `xsa_alpha` is `tanh`-bounded per head, and runs in fp32.

**MUDD — Multiway Dynamic Dense connections** ([arXiv:2502.12170](https://arxiv.org/abs/2502.12170)).
At layers 24 and 47 only, a two-layer net maps the current hidden state to
mixing weights over three retained history states, producing separate attention
inputs and residual bases. Accumulation is ordered left to right — the config
pins `mudd_accumulation` precisely so a reimplementation cannot silently use a
stacked contraction instead.

**Value embeddings.** A second table added into V at 16 of the 48 layers.

**Local/global interleaving.** Every 4th layer (3, 7, … 47) is full attention
with **no positional encoding**; the rest use a sliding window with partial
RoPE. RoPE base 1024 is small because it only has to span the window.

**Logit softcap.** `23 · sigmoid((x + 5) / 7.5)` — `b` a shift, `c` a divisor.

## Details that bite

**Both gates read the layer input**, not the attention output:

```python
gate = attn_gate_scale * sigmoid(linear(attn_in[..., :attn_gate_channels], attn_gate))
```

`attn_gate` is `[n_heads, 128]`, which looks exactly like a per-channel gate on
the head output — reading it that way saturates the sigmoid into a near-binary
gate. The same applies to the value-embedding gate, which explains the
otherwise inexplicable `ve_gate: [2, 12]` when `ve_dim` is 128: the 12 is
`ve_gate_channels` of **`attn_in`**, unrelated to the value width.

**RoPE frequencies use an inclusive linspace:**

```python
freq = (1.0 / base) ** mx.linspace(0, 1, n_pairs)   # spacing 1/(n-1), not 1/n
freq = mx.repeat(freq, 2)                           # adjacent pairs
freq = concat([freq, zeros(head_dim - 2 * n_pairs)])
```

The exponent reaches 1.0. Using `base ** (-i / n_pairs)` puts every frequency
slightly off. The odd lanes carry a negated sine, giving
`(a, b) -> (a cos + b sin, b cos - a sin)`, and the cos/sin tables are rounded
to bfloat16 before use — that rounding is part of the trained function.

**The RMSNorm epsilon is not what the reference comment says.** The reference
calls `F.rms_norm(x, (d,))` with no epsilon and notes that torch then uses
`finfo(dtype).eps` — 0.0078 in bfloat16. Measured, that is not what happens:
torch computes in fp32 with an fp32-sized epsilon. Results are identical for
eps in {0, 1.19e-7, 1e-6} and 1.6e-2 off for the bfloat16 value. Using the
bfloat16 epsilon costs **~3% relative error per layer** and drops argmax
agreement to 96.6%. `_RMS_EPS` pins the fp32 value.

**The per-layer scalars fold in bfloat16.** The reference multiplies a 0-dim
fp32 tensor into a bf16 weight; PyTorch treats a 0-dim tensor as a wrapped
scalar, so the product stays **bf16** and the scalar is rounded first. A
shape-`(1,)` tensor would promote to fp32 and diverge. This port folds them
once in `sanitize()` at the checkpoint's dtype, which is equivalent and leaves
plain `nn.Linear` modules so quantization still works.

**The sliding window is inclusive of the query token**, so a serialized window
of 1024 admits 1025 keys. In mlx-lm terms that is `window_size = sliding_window
+ 1` and `RotatingKVCache(max_size=sliding_window + 1)`.

**`lm_head_precision_mode: "oracle_exact"`** means the head runs in the
activation dtype rather than upcasting to fp32.
