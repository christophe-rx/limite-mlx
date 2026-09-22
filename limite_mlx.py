# Copyright © 2026
"""MLX implementation of the `limite` architecture (`LimiteForCausalLM`).

Ported from the reference vLLM plugin
(https://github.com/paradigma-inc/limite-violetto), preserving its numerics:
parameter-free RMSNorm with a dtype-dependent epsilon, bfloat16 rotary tables,
bfloat16 folding of the per-layer projection scalars, and fp32 XSA.

Install either way:
  * copy to `mlx_lm/models/limite.py`, or
  * drop next to the weights and add `"model_file": "limite.py"` to config.json
Absolute imports keep both routes working.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs, create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "limite"
    hidden_size: int = 1280
    num_hidden_layers: int = 48
    num_attention_heads: int = 10
    num_key_value_heads: int = 2
    head_dim: int = 128
    intermediate_size: int = 3328
    vocab_size: int = 151680
    mlp_type: str = "swiglu"
    attention_softmax_scale: float = 0.1
    tie_word_embeddings: bool = True
    # rotary
    rope_n_pairs: int = 32
    rope_base_local: float = 1024.0
    rope_per_layer: bool = False
    # local / global interleaving
    global_layers: List[int] = field(default_factory=list)
    global_nope: bool = True
    global_window: int = -1
    sliding_window: int = 1024
    # value embeddings
    ve_layers: List[int] = field(default_factory=list)
    ve_dim: int = 128
    ve_stored_heads: int = 2
    ve_gate_channels: int = 12
    ve_gate_scale: float = 2.0
    # attention output gate
    attn_gate_channels: int = 128
    attn_gate_scale: float = 2.0
    # XSA
    xsa: bool = True
    xsa_layers: List[int] = field(default_factory=list)
    xsa_normalize_eps: float = 1e-4
    # MUDD
    mudd_taps: int = 3
    mudd_inter: int = 32
    mudd_mlp: bool = True
    mudd_layers: List[int] = field(default_factory=list)
    mudd_tap_idx: Dict[str, List[int]] = field(default_factory=dict)
    # head
    softcap_logits: Dict[str, Any] = field(default_factory=dict)
    final_softcap: float = 0.0
    lm_head_precision_mode: str = "oracle_exact"


# The reference calls `F.rms_norm(x, (d,))` with no epsilon. Despite what its
# comment suggests, torch does *not* apply finfo(bf16).eps there: measured
# against this torch build it computes in fp32 with an fp32-sized epsilon
# (identical results for eps in {0, 1.19e-7, 1e-6}, and 1.6e-2 off for
# finfo(bf16).eps = 0.0078). Matching that is worth ~3% per layer.
_RMS_EPS = float(mx.finfo(mx.float32).eps)


def rms_norm(x: mx.array) -> mx.array:
    """Parameter-free RMSNorm, computed in fp32 like the reference."""
    return mx.fast.rms_norm(x, None, _RMS_EPS)


class Rotary(nn.Module):
    """Partial, adjacent-pair-interleaved rotary with an odd-lane sign flip."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        if args.rope_per_layer:
            raise NotImplementedError("rope_per_layer=true needs a per-layer table")
        n_pairs, head_dim = args.rope_n_pairs, args.head_dim
        # note the inclusive linspace: the exponent reaches 1.0, so the
        # spacing is 1/(n_pairs - 1), not 1/n_pairs
        freq = (1.0 / args.rope_base_local) ** mx.linspace(0, 1, n_pairs, dtype=mx.float32)
        freq = mx.repeat(freq, 2)
        self._freq = mx.concatenate([freq, mx.zeros(head_dim - 2 * n_pairs)])

    def __call__(self, q, k, offset: int):
        L = q.shape[1]
        pos = (mx.arange(L, dtype=mx.float32) + offset).reshape(-1, 1)
        theta = pos * self._freq.reshape(1, -1)
        cos = mx.cos(theta).astype(mx.bfloat16).reshape(1, L, 1, -1)
        sin = mx.sin(theta).astype(mx.bfloat16).reshape(1, L, 1, -1)
        # negate the odd lanes, giving (a, b) -> (a cos + b sin, b cos - a sin)
        lane = mx.tile(mx.array([1.0, -1.0], dtype=mx.bfloat16), [sin.shape[-1] // 2])
        sin = sin * lane.reshape(1, 1, 1, -1)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    @staticmethod
    def _rotate(x, cos, sin):
        pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        flip = mx.stack([pairs[..., 1], pairs[..., 0]], axis=-1).reshape(x.shape)
        return cos.astype(x.dtype) * x + sin.astype(x.dtype) * flip


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.n_heads = args.num_attention_heads
        self.n_kv = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.groups = self.n_heads // self.n_kv
        self.scale = args.attention_softmax_scale

        self.is_global = layer_idx in set(args.global_layers)
        self.uses_rope = not (args.global_nope and self.is_global)
        self.uses_ve = layer_idx in set(args.ve_layers)
        self.uses_xsa = bool(args.xsa) and layer_idx in set(args.xsa_layers)

        h, kv = args.hidden_size, self.n_kv * self.head_dim
        # the per-layer qkv_scale / o_scale are folded into these weights in
        # sanitize(), exactly as the reference folds them at runtime
        self.q_proj = nn.Linear(h, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(h, kv, bias=False)
        self.v_proj = nn.Linear(h, kv, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, h, bias=False)

        self.xsa_alpha = mx.zeros(self.n_heads, dtype=mx.float32)
        if self.uses_ve:
            self.ve_gate = mx.zeros((args.ve_stored_heads, args.ve_gate_channels),
                                    dtype=mx.float32)
        if args.attn_gate_channels:
            self.attn_gate = mx.zeros((self.n_heads, args.attn_gate_channels),
                                      dtype=mx.float32)

    def __call__(self, attn_in, value_embeds, inputs, rotary, mask, cache):
        a = self.args
        B, L, _ = attn_in.shape
        q = self.q_proj(attn_in).reshape(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(attn_in).reshape(B, L, self.n_kv, self.head_dim)
        v = self.v_proj(attn_in).reshape(B, L, self.n_kv, self.head_dim)

        if self.uses_ve:
            ve = value_embeds(inputs).astype(v.dtype).reshape(
                B, L, a.ve_stored_heads, a.ve_dim)
            if a.ve_dim < self.head_dim:
                ve = mx.pad(ve, [(0, 0), (0, 0), (0, 0), (0, self.head_dim - a.ve_dim)])
            gw = self.ve_gate.astype(attn_in.dtype)
            if a.ve_stored_heads > self.n_kv:
                ve, gw = ve[:, :, : self.n_kv], gw[: self.n_kv]
            # the gate is driven by the *layer input*, sliced to ve_gate_channels
            gate = a.ve_gate_scale * mx.sigmoid(attn_in[..., : gw.shape[-1]] @ gw.T)
            v = v + gate[..., None].astype(v.dtype) * ve

        # qk_norm: rms_pre_rope, after the value-embedding step
        q, k = rms_norm(q), rms_norm(k)
        offset = cache.offset if cache is not None else 0
        if self.uses_rope:
            q, k = rotary(q, k, offset)

        v_self = v                                    # current tokens', for XSA
        qT = q.transpose(0, 2, 1, 3)
        kT = k.transpose(0, 2, 1, 3)
        vT = v.transpose(0, 2, 1, 3)
        if cache is not None:
            kT, vT = cache.update_and_fetch(kT, vT)

        y = mx.fast.scaled_dot_product_attention(qT, kT, vT, scale=self.scale, mask=mask)
        y = y.transpose(0, 2, 1, 3)                   # [B, L, n_heads, head_dim]

        if self.uses_xsa:
            vx = mx.repeat(v_self, self.groups, axis=2).astype(mx.float32)
            vn = vx * mx.rsqrt(mx.maximum(mx.sum(vx * vx, axis=-1, keepdims=True),
                                          a.xsa_normalize_eps ** 2))
            proj = mx.sum(y.astype(mx.float32) * vn, axis=-1, keepdims=True)
            alpha = mx.tanh(self.xsa_alpha.astype(mx.float32)).reshape(1, 1, -1, 1)
            y = y - (alpha * proj * vn).astype(y.dtype)

        if a.attn_gate_channels:
            gate = a.attn_gate_scale * mx.sigmoid(
                attn_in[..., : a.attn_gate_channels] @ self.attn_gate.astype(attn_in.dtype).T)
            y = y * gate[..., None].astype(y.dtype)

        return self.o_proj(y.reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        h, i = args.hidden_size, args.intermediate_size
        self.mlp_type = args.mlp_type
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)
        if self.mlp_type == "swiglu":
            self.gate_proj = nn.Linear(h, i, bias=False)

    def __call__(self, x):
        up = self.up_proj(x)
        if self.mlp_type == "relu2":
            act = mx.square(nn.relu(up))
        else:
            act = nn.silu(self.gate_proj(x)) * up
        return self.down_proj(act)


class Mudd(nn.Module):
    """Dynamic dense mixing over a few retained residual-stream states."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        L, t, i, h = (args.num_hidden_layers, args.mudd_taps,
                      args.mudd_inter, args.hidden_size)
        self.dense1 = mx.zeros((i, h), dtype=mx.float32)
        self.dense2 = mx.zeros((L, t, i), dtype=mx.float32)
        self.bias = mx.zeros((L, t), dtype=mx.float32)
        self.uses_r_way = bool(args.mudd_mlp)
        if self.uses_r_way:
            self.dense2_mlp = mx.zeros((L, t, i), dtype=mx.float32)
            self.bias_mlp = mx.zeros((L, t), dtype=mx.float32)

    def combine(self, values, x_cur, layer_idx, r_way=False):
        n = len(values)
        inner = nn.gelu(rms_norm(x_cur) @ self.dense1.astype(x_cur.dtype).T)
        d2, b = ((self.dense2_mlp, self.bias_mlp) if r_way
                 else (self.dense2, self.bias))
        w = inner @ d2[layer_idx, :n].astype(inner.dtype).T
        w = w + b[layer_idx, :n].astype(w.dtype)
        # ordered left-to-right accumulation, as the trainer does it
        out = w[..., 0:1].astype(values[0].dtype) * values[0]
        for j in range(1, n):
            out = out + w[..., j : j + 1].astype(values[j].dtype) * values[j]
        return out


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args)
        self.resid_lambda_attn = mx.zeros((), dtype=mx.float32)
        self.post_lambda_attn = mx.zeros((), dtype=mx.float32)
        self.resid_lambda_mlp = mx.zeros((), dtype=mx.float32)
        self.post_lambda_mlp = mx.zeros((), dtype=mx.float32)

    def __call__(self, x, attn_in, residual_base, value_embeds, inputs,
                 rotary, mask, cache):
        attn_out = self.self_attn(attn_in, value_embeds, inputs, rotary, mask, cache)
        x = (self.resid_lambda_attn.astype(x.dtype) * residual_base
             + self.post_lambda_attn.astype(x.dtype) * attn_out)
        mlp_out = self.mlp(rms_norm(x))
        return (self.resid_lambda_mlp.astype(x.dtype) * x
                + self.post_lambda_mlp.astype(x.dtype) * mlp_out)


class LimiteModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.value_embeds = nn.Embedding(args.vocab_size,
                                         args.ve_stored_heads * args.ve_dim)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.mudd = Mudd(args)
        self.rotary = Rotary(args)
        self.tap_idx = {int(k): [int(i) for i in v] for k, v in args.mudd_tap_idx.items()}
        self.retained = {i for taps in self.tap_idx.values() for i in taps}
        self.global_set = set(args.global_layers)

    def __call__(self, inputs, cache=None):
        h = rms_norm(self.embed_tokens(inputs))
        if cache is None:
            cache = [None] * len(self.layers)

        first_global = next((i for i in range(len(self.layers)) if i in self.global_set), 0)
        first_local = next((i for i in range(len(self.layers)) if i not in self.global_set), 0)
        global_mask = create_attention_mask(h, cache[first_global])
        # the serialized window is inclusive of the query token, so the runtime
        # span is sliding_window + 1 keys
        local_mask = create_attention_mask(h, cache[first_local],
                                           window_size=self.args.sliding_window + 1)

        history = {0: h} if self.tap_idx else {}
        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            if i in self.tap_idx:
                vals = [history[t] for t in self.tap_idx[i]]
                attn_in = rms_norm(self.mudd.combine(vals, h, i))
                residual_base = (self.mudd.combine(vals, h, i, r_way=True)
                                 if self.mudd.uses_r_way else h)
            else:
                attn_in = rms_norm(h)
                residual_base = h
            mask = global_mask if i in self.global_set else local_mask
            h = layer(h, attn_in, residual_base, self.value_embeds, inputs,
                      self.rotary, mask, c)
            if i + 1 in self.retained:
                history[i + 1] = h

        if self.args.final_softcap > 0:
            cap = self.args.final_softcap
            h = cap * mx.tanh(h / cap)
        return rms_norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LimiteModel(args)
        sc = dict(args.softcap_logits)
        if sc.get("kind") != "sigmoid":
            raise ValueError(f"unsupported softcap kind {sc.get('kind')!r}")
        self.sc_a, self.sc_b, self.sc_c = float(sc["a"]), float(sc["b"]), float(sc["c"])

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = self.model(inputs, cache)
        # oracle_exact: the head runs in the activation dtype, not upcast fp32
        raw = self.model.embed_tokens.as_linear(h)
        return self.sc_a * mx.sigmoid((raw.astype(mx.float32) + self.sc_b) / self.sc_c)

    def sanitize(self, weights):
        """Strip the `model.` prefix mismatch and fold the per-layer scalars.

        The reference multiplies a 0-dim fp32 scalar into a bf16 weight, which
        PyTorch resolves in bf16 (the scalar is rounded first). Doing the same
        fold once at load time is bit-identical and leaves plain nn.Linear
        modules, so `mlx_lm.convert -q` still works.
        """
        out = {}
        scales = {k: v for k, v in weights.items()
                  if k.endswith(("self_attn.qkv_scale", "self_attn.o_scale"))}
        folded = {"qkv_scale": ("q_proj", "k_proj", "v_proj"), "o_scale": ("o_proj",)}
        for k, v in weights.items():
            if k.endswith(("self_attn.qkv_scale", "self_attn.o_scale")):
                continue
            if k.endswith(".weight") and ".self_attn." in k:
                proj = k.rsplit(".", 2)[-2]
                for sname, projs in folded.items():
                    if proj in projs:
                        sk = k.rsplit(".", 2)[0] + "." + sname
                        if sk in scales:
                            v = scales[sk].astype(v.dtype) * v
            out[k] = v
        if "lm_head.weight" in out:
            out.pop("lm_head.weight")
        return out

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for i in range(self.args.num_hidden_layers):
            if i in set(self.args.global_layers):
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window + 1))
        return caches
