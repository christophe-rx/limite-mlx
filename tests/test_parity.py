"""Verify the MLX port against the PyTorch reference implementation.

    uv run --extra dev pytest tests/ -v

Needs a downloaded checkpoint in ./model (see README) and the dev extra for
torch. Skips cleanly if either is missing.
"""
import json
import pathlib
import sys

import numpy as np
import pytest

import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import limite_mlx

torch = pytest.importorskip("torch", reason="parity test needs the dev extra")
import reference_torch  # noqa: E402  (same directory)

_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL = next((p for p in [_ROOT / "models" / "limite-1b-violetto",
                          _ROOT / "models" / "limite-1b-base",
                          _ROOT / "model"]
              if (p / "model.safetensors").exists()), _ROOT / "models" / "limite-1b-violetto")
pytestmark = pytest.mark.skipif(
    not (MODEL / "model.safetensors").exists(),
    reason=f"no checkpoint at {MODEL}",
)

TEXT = ("The history of the printing press is one of the clearest examples of how a "
        "single technology can reshape an entire civilisation. Before Johannes "
        "Gutenberg began experimenting with movable type in Mainz around 1440, books "
        "in Europe were copied by hand, usually by monks working in scriptoria.")


def _ids():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(MODEL / "tokenizer.json")).encode(TEXT).ids


def _mlx(dtype, fold_dtype=None, cfg_overrides=None):
    cfg = json.loads((MODEL / "config.json").read_text())
    cfg.update(cfg_overrides or {})
    m = limite_mlx.Model(limite_mlx.ModelArgs.from_dict(cfg))
    raw = mx.load(str(MODEL / "model.safetensors"))
    if fold_dtype is not None:                     # fold the scales at this precision
        raw = {k: v.astype(fold_dtype) for k, v in raw.items()}
    m.load_weights(list(m.sanitize(raw).items()), strict=True)
    m.set_dtype(dtype)
    m.eval()
    return m


def _torch(dtype, **overrides):
    m = reference_torch.load_model(str(MODEL), dtype=dtype)
    for k, v in overrides.items():
        setattr(m.cfg, k, v)
    return m


def test_float32_parity():
    """The strong check: identical maths agrees to fp32 rounding."""
    ids = _ids()
    a = _torch(torch.float32)(torch.tensor(ids)[None])[0].detach().float().numpy()
    b = np.array(_mlx(mx.float32, mx.float32)(mx.array([ids]))[0].astype(mx.float32))
    assert (a.argmax(-1) == b.argmax(-1)).all()
    assert np.abs(a - b).max() < 1e-3, np.abs(a - b).max()


def test_bfloat16_argmax_agreement():
    """bf16 differs only by rounding; most argmaxes still agree."""
    ids = _ids()
    a = _torch(torch.bfloat16)(torch.tensor(ids)[None])[0].detach().float().numpy()
    b = np.array(_mlx(mx.bfloat16)(mx.array([ids]))[0].astype(mx.float32))
    assert (a.argmax(-1) == b.argmax(-1)).mean() > 0.95


def test_incremental_decoding_matches_full_forward():
    """Prefill + one-token steps must equal a single full forward."""
    ids = _ids()
    m = _mlx(mx.bfloat16)
    full = np.array(m(mx.array([ids]))[0].astype(mx.float32))
    cache = m.make_cache()
    out = [np.array(m(mx.array([ids[:20]]), cache=cache)[0].astype(mx.float32))]
    for t in range(20, len(ids)):
        out.append(np.array(m(mx.array([[ids[t]]]), cache=cache)[0].astype(mx.float32)))
    inc = np.concatenate(out, 0)
    assert (full.argmax(-1) == inc.argmax(-1)).all()
    assert np.abs(full - inc).max() < 1e-3


def test_sliding_window_convention():
    """The serialized window is inclusive of the query token (span = w + 1).

    Forced small so the window actually binds on a short sequence.
    """
    ids = _ids()
    a = _torch(torch.float32, sliding_window=8)(torch.tensor(ids)[None])[0]
    a = a.detach().float().numpy()
    b = _mlx(mx.float32, mx.float32, {"sliding_window": 8})(mx.array([ids]))[0]
    b = np.array(b.astype(mx.float32))
    assert (a.argmax(-1) == b.argmax(-1)).all()
    assert np.abs(a - b).max() < 1e-2


def test_rotation_with_small_window():
    """Cache rotation (RotatingKVCache) must not change the result."""
    ids = _ids()
    m = _mlx(mx.float32, mx.float32, {"sliding_window": 8})
    full = np.array(m(mx.array([ids]))[0].astype(mx.float32))
    cache = m.make_cache()
    out = [np.array(m(mx.array([ids[:12]]), cache=cache)[0].astype(mx.float32))]
    for t in range(12, len(ids)):
        out.append(np.array(m(mx.array([[ids[t]]]), cache=cache)[0].astype(mx.float32)))
    inc = np.concatenate(out, 0)
    assert (full.argmax(-1) == inc.argmax(-1)).all()
