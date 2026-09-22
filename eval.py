#!/usr/bin/env python3
"""Sanity-check a checkpoint: next-token loss and top-1 on held-out prose.

    python eval.py --model model-base     # expect ~3.2 nats / ~41%
    python eval.py --model model          # Violetto is a maths model; prose is worse
"""
import argparse
import math

import mlx.core as mx
import numpy as np
from mlx_lm import load

TEXT = (
    "The history of the printing press is one of the clearest examples of how a single "
    "technology can reshape an entire civilisation. Before Johannes Gutenberg began "
    "experimenting with movable type in Mainz around 1440, books in Europe were copied "
    "by hand, usually by monks working in scriptoria. A single Bible could take a scribe "
    "more than a year to produce, and the cost of such a volume was equivalent to a "
    "labourer's wages for several years. Literacy was therefore confined to a narrow "
    "class of clergy, nobles and merchants."
)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="model")
a = ap.parse_args()

model, tokenizer = load(a.model)
ids = tokenizer.encode(TEXT)
logits = np.array(model(mx.array([ids]))[0].astype(mx.float32))[:-1].astype(np.float64)
target = np.array(ids[1:])

logits -= logits.max(-1, keepdims=True)
nll = -(logits[np.arange(len(target)), target] - np.log(np.exp(logits).sum(-1))).mean()
top1 = (logits.argmax(-1) == target).mean()

print(f"{a.model}: {len(ids)} tokens")
print(f"  loss  = {nll:.3f} nats/token  (uniform baseline "
      f"{math.log(model.args.vocab_size):.2f})")
print(f"  top-1 = {top1:.1%}")
