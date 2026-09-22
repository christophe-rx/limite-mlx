"""The `limite` architecture (`LimiteForCausalLM`) in plain PyTorch.

Faithful to the reference implementation in
https://github.com/paradigma-inc/limite-violetto (a vLLM plugin, which needs
Linux + CUDA). This port keeps the same numerics but runs anywhere PyTorch
does, including Apple-Silicon MPS.

Architecture notes, all following the reference:
  * RMSNorm has no learnable gain and no explicit epsilon, so torch uses
    finfo(dtype).eps -- the epsilon is dtype dependent and part of the
    trained function.
  * Rotary is partial (first 2*rope_n_pairs channels), adjacent-pair
    interleaved, with an odd-lane sign flip. cos/sin are rounded to bfloat16
    before use, which the reference treats as part of the function.
  * Global layers (every 4th) use full attention and NO positional encoding;
    the rest use an inclusive sliding window of `sliding_window` + 1 keys.
  * Value embeddings and the attention output gate are both driven by the
    *layer input* `attn_in`, sliced to their configured channel counts.
  * XSA (arXiv:2603.09078) removes the component of the attention output
    along the token's own value vector, with a per-head tanh-bounded alpha.
  * MUDD (arXiv:2502.12170) mixes a few retained history states into the
    attention input and the residual base at the tapped layers, accumulating
    left to right.
  * Both residual joins are `resid_lambda * stream + post_lambda * branch`.
"""

from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as F
from torch import nn


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """No learnable gain, no explicit eps (torch falls back to finfo.eps)."""
    return F.rms_norm(x, (x.size(-1),))


class LimiteConfig:
    def __init__(self, d):
        self.__dict__.update(d)
        self.tap_idx = {int(k): [int(i) for i in v] for k, v in self.mudd_tap_idx.items()}
        self.global_set = {int(x) for x in self.global_layers}
        self.ve_set = {int(x) for x in self.ve_layers}
        self.xsa_set = {int(x) for x in self.xsa_layers}
        self.retained = {i for taps in self.tap_idx.values() for i in taps}

    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            return cls(json.load(f))


class Rotary(nn.Module):
    """Partial adjacent-pair rotary with an odd-lane sign flip."""

    def __init__(self, cfg):
        super().__init__()
        if bool(cfg.rope_per_layer):
            raise NotImplementedError("rope_per_layer=true needs a per-layer table")
        n_pairs, head_dim = int(cfg.rope_n_pairs), int(cfg.head_dim)
        base = float(cfg.rope_base_local)
        freq = (1.0 / base) ** torch.linspace(0, 1, steps=n_pairs, dtype=torch.float32)
        freq = freq.repeat_interleave(2)
        freq = torch.cat([freq, freq.new_zeros(head_dim - 2 * n_pairs)])
        self.register_buffer("freq", freq, persistent=False)

    def tables(self, positions):
        theta = positions.to(torch.float32).view(-1, 1) * self.freq.view(1, -1)
        # the bfloat16 rounding of the tables is part of the trained function
        cos = theta.cos().to(torch.bfloat16).view(-1, 1, self.freq.numel())
        sin = theta.sin().to(torch.bfloat16).view(-1, 1, self.freq.numel())
        sin = sin.clone()
        sin[:, :, 1::2] *= -1
        return cos, sin

    @staticmethod
    def rotate(x, cos, sin):
        flip = x.view(*x.shape[:-1], x.shape[-1] // 2, 2).flip(-1).view(x.shape)
        return cos.to(x.dtype) * x + sin.to(x.dtype) * flip


class LimiteAttention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.cfg, self.layer_idx = cfg, layer_idx
        self.nh, self.nkv = int(cfg.num_attention_heads), int(cfg.num_key_value_heads)
        self.hd = int(cfg.head_dim)
        self.groups = self.nh // self.nkv
        self.uses_ve = layer_idx in cfg.ve_set
        self.uses_xsa = bool(cfg.xsa) and layer_idx in cfg.xsa_set
        self.is_global = layer_idx in cfg.global_set
        self.uses_rope = not (bool(cfg.global_nope) and self.is_global)

        h, kv = int(cfg.hidden_size), self.nkv * self.hd
        self.q_proj = nn.Linear(h, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(h, kv, bias=False)
        self.v_proj = nn.Linear(h, kv, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, h, bias=False)
        self.qkv_scale = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.o_scale = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.xsa_alpha = nn.Parameter(torch.empty(self.nh, dtype=torch.float32))
        if self.uses_ve:
            self.ve_gate = nn.Parameter(torch.empty(int(cfg.ve_stored_heads),
                                                    int(cfg.ve_gate_channels),
                                                    dtype=torch.float32))
        if int(cfg.attn_gate_channels):
            self.attn_gate = nn.Parameter(torch.empty(self.nh, int(cfg.attn_gate_channels),
                                                      dtype=torch.float32))

    @staticmethod
    def _fold(scale, weight):
        """Fold a learned scalar into a projection *in the weight's dtype*.

        The reference multiplies a 0-dim fp32 tensor by a bf16 weight, which
        PyTorch treats as a wrapped scalar: the product stays bf16 and the
        scalar is rounded first. Keeping the cast explicit preserves that.
        """
        return scale.to(weight.dtype) * weight

    def forward(self, input_ids, positions, attn_in, value_embeds, rotary, mask,
                cache=None, layer_idx=None):
        cfg = self.cfg
        B, T, _ = attn_in.shape
        q = F.linear(attn_in, self._fold(self.qkv_scale, self.q_proj.weight))
        k = F.linear(attn_in, self._fold(self.qkv_scale, self.k_proj.weight))
        v = F.linear(attn_in, self._fold(self.qkv_scale, self.v_proj.weight))
        q = q.view(B, T, self.nh, self.hd)
        k = k.view(B, T, self.nkv, self.hd)
        v = v.view(B, T, self.nkv, self.hd)

        if self.uses_ve:
            ve = value_embeds(input_ids).to(v.dtype).view(
                B, T, int(cfg.ve_stored_heads), int(cfg.ve_dim))
            ve = F.pad(ve, (0, self.hd - int(cfg.ve_dim)))
            gw = self.ve_gate.to(attn_in.dtype)
            if int(cfg.ve_stored_heads) > self.nkv:
                ve, gw = ve[:, :, : self.nkv], gw[: self.nkv]
            gate = float(cfg.ve_gate_scale) * torch.sigmoid(
                F.linear(attn_in[..., : gw.size(-1)], gw))
            v = v + gate.unsqueeze(-1).to(v.dtype) * ve

        # qk_norm: rms_pre_rope -- after the value-embedding step
        q, k = rms_norm(q), rms_norm(k)
        if self.uses_rope:
            cos, sin = rotary.tables(positions)
            cos = cos.view(1, T, 1, self.hd)
            sin = sin.view(1, T, 1, self.hd)
            q, k = rotary.rotate(q, cos, sin), rotary.rotate(k, cos, sin)

        if cache is not None:
            past = cache[layer_idx]
            if past is not None:
                k = torch.cat([past[0], k], dim=1)
                v_full = torch.cat([past[1], v], dim=1)
            else:
                v_full = v
            cache[layer_idx] = (k, v_full)
        else:
            v_full = v

        qh = q.transpose(1, 2)
        kh, vh = k.transpose(1, 2), v_full.transpose(1, 2)
        kh = kh.repeat_interleave(self.groups, dim=1)
        vh = vh.repeat_interleave(self.groups, dim=1)
        scores = torch.matmul(qh, kh.transpose(-1, -2)).float()
        scores = scores * float(cfg.attention_softmax_scale)
        scores = scores.masked_fill(mask, float("-inf"))
        y = torch.matmul(scores.softmax(-1).to(vh.dtype), vh)
        y = y.transpose(1, 2)                                  # [B, T, nh, hd]

        if self.uses_xsa:
            vx = v.repeat_interleave(self.groups, dim=2).float()   # current tokens only
            vn = F.normalize(vx, dim=-1, eps=float(cfg.xsa_normalize_eps))
            proj = (y.float() * vn).sum(-1, keepdim=True)
            alpha = torch.tanh(self.xsa_alpha.float()).view(1, 1, self.nh, 1)
            y = y - (alpha * proj * vn).to(y.dtype)

        if int(cfg.attn_gate_channels):
            gate = float(cfg.attn_gate_scale) * torch.sigmoid(
                F.linear(attn_in[..., : int(cfg.attn_gate_channels)],
                         self.attn_gate.to(attn_in.dtype)))
            y = y * gate.to(y.dtype).unsqueeze(-1)

        y = y.reshape(B, T, self.nh * self.hd)
        return F.linear(y, self._fold(self.o_scale, self.o_proj.weight))


class LimiteMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, i = int(cfg.hidden_size), int(cfg.intermediate_size)
        self.mlp_type = str(cfg.mlp_type)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)
        if self.mlp_type == "swiglu":
            self.gate_proj = nn.Linear(h, i, bias=False)

    def forward(self, h):
        up = self.up_proj(h)
        if self.mlp_type == "relu2":
            act = F.relu(up).square()
        elif self.mlp_type == "swiglu":
            act = F.silu(self.gate_proj(h)) * up
        else:
            raise AssertionError(f"unsupported mlp_type {self.mlp_type!r}")
        return self.down_proj(act)


class LimiteMudd(nn.Module):
    """Dynamic dense mixing over a few retained residual-stream states."""

    def __init__(self, cfg):
        super().__init__()
        L, taps = int(cfg.num_hidden_layers), int(cfg.mudd_taps)
        inter, hidden = int(cfg.mudd_inter), int(cfg.hidden_size)
        f32 = torch.float32
        self.dense1 = nn.Parameter(torch.empty(inter, hidden, dtype=f32))
        self.dense2 = nn.Parameter(torch.empty(L, taps, inter, dtype=f32))
        self.bias = nn.Parameter(torch.empty(L, taps, dtype=f32))
        self.uses_r_way = bool(getattr(cfg, "mudd_mlp", False))
        if self.uses_r_way:
            self.dense2_mlp = nn.Parameter(torch.empty(L, taps, inter, dtype=f32))
            self.bias_mlp = nn.Parameter(torch.empty(L, taps, dtype=f32))

    def combine(self, values, x_cur, layer_idx, r_way=False):
        n = len(values)
        inner = F.gelu(F.linear(rms_norm(x_cur), self.dense1.to(x_cur.dtype)))
        dense2, bias = ((self.dense2_mlp, self.bias_mlp) if r_way
                        else (self.dense2, self.bias))
        w = F.linear(inner, dense2[layer_idx, :n].to(inner.dtype))
        w = w + bias[layer_idx, :n].to(w.dtype)
        # ordered left-to-right accumulation, as the trainer does it
        out = w[..., 0:1].type_as(values[0]) * values[0]
        for i in range(1, n):
            out = out + w[..., i : i + 1].type_as(values[i]) * values[i]
        return out


class LimiteLayer(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.self_attn = LimiteAttention(cfg, layer_idx)
        self.mlp = LimiteMLP(cfg)
        f32 = torch.float32
        for n in ("resid_lambda_attn", "post_lambda_attn",
                  "resid_lambda_mlp", "post_lambda_mlp"):
            setattr(self, n, nn.Parameter(torch.empty((), dtype=f32)))

    def forward(self, input_ids, positions, x, attn_in, value_embeds,
                residual_base, rotary, mask, cache=None, layer_idx=None):
        attn_out = self.self_attn(input_ids, positions, attn_in, value_embeds,
                                  rotary, mask, cache, layer_idx)
        x = (self.resid_lambda_attn.to(x.dtype) * residual_base
             + self.post_lambda_attn.to(x.dtype) * attn_out)
        mlp_out = self.mlp(rms_norm(x))
        return (self.resid_lambda_mlp.to(x.dtype) * x
                + self.post_lambda_mlp.to(x.dtype) * mlp_out)


class LimiteForCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        V, H = int(cfg.vocab_size), int(cfg.hidden_size)
        self.embed_tokens = nn.Embedding(V, H)
        self.value_embeds = nn.Embedding(V, int(cfg.ve_stored_heads) * int(cfg.ve_dim))
        self.layers = nn.ModuleList(
            [LimiteLayer(cfg, i) for i in range(int(cfg.num_hidden_layers))])
        self.mudd = LimiteMudd(cfg)
        self.rotary = Rotary(cfg)
        sc = dict(cfg.softcap_logits)
        if sc.get("kind") != "sigmoid":
            raise ValueError(f"unsupported softcap kind {sc.get('kind')!r}")
        self.sc_a, self.sc_b, self.sc_c = float(sc["a"]), float(sc["b"]), float(sc["c"])

    def _mask(self, positions, total, layer_idx, device):
        """Causal, plus an inclusive sliding window on non-global layers."""
        q = positions.view(-1, 1)
        k = torch.arange(total, device=device).view(1, -1)
        m = k > q
        window = (int(self.cfg.global_window) if layer_idx in self.cfg.global_set
                  else int(self.cfg.sliding_window))
        if window >= 0:
            m = m | (k < q - window)
        return m[None, None]

    def hidden_states(self, input_ids, positions=None, cache=None):
        cfg = self.cfg
        B, T = input_ids.shape
        device = input_ids.device
        if positions is None:
            positions = torch.arange(T, device=device)
        total = int(positions[-1].item()) + 1
        if cache is None:
            cache_ref = None
        else:
            cache_ref = cache

        x = rms_norm(self.embed_tokens(input_ids))
        history = {0: x} if cfg.tap_idx else {}
        for i, layer in enumerate(self.layers):
            if i in cfg.tap_idx:
                vals = [history[t] for t in cfg.tap_idx[i]]
                attn_in = rms_norm(self.mudd.combine(vals, x, i))
                residual_base = (self.mudd.combine(vals, x, i, r_way=True)
                                 if self.mudd.uses_r_way else x)
            else:
                attn_in = rms_norm(x)
                residual_base = x
            mask = self._mask(positions, total, i, device)
            x = layer(input_ids, positions, x, attn_in, self.value_embeds,
                      residual_base, self.rotary, mask, cache_ref, i)
            if i + 1 in cfg.retained:
                history[i + 1] = x
        cap = float(cfg.final_softcap)
        if cap > 0:
            x = cap * torch.tanh(x / cap)
        return rms_norm(x)

    def compute_logits(self, h):
        w = self.embed_tokens.weight
        if str(self.cfg.lm_head_precision_mode) == "oracle_exact":
            raw = F.linear(h, w.to(h.dtype))
        else:
            raw = F.linear(h.float(), w.float())
        return self.sc_a * torch.sigmoid((raw.float() + self.sc_b) / self.sc_c)

    def forward(self, input_ids, positions=None, cache=None):
        return self.compute_logits(self.hidden_states(input_ids, positions, cache))

    def new_cache(self):
        return [None] * int(self.cfg.num_hidden_layers)


def load_model(model_dir="model", dtype=torch.bfloat16, device="cpu"):
    from safetensors.torch import load_file

    cfg = LimiteConfig.from_file(f"{model_dir}/config.json")
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            model = LimiteForCausalLM(cfg)
    finally:
        torch.set_default_dtype(prev)

    sd = {}
    for k, t in load_file(f"{model_dir}/model.safetensors").items():
        k = k[len("model."):] if k.startswith("model.") else k
        # fp32 scalars/gates stay fp32; the bf16 matrices take the run dtype
        sd[k] = t if t.dtype == torch.float32 else t.to(dtype)
    model.load_state_dict(sd, strict=True, assign=True)
    model.rotary.freq = Rotary(cfg).freq
    return model.to(device=device).eval()
