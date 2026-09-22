#!/usr/bin/env python3
"""Chat with a Limite model on Apple Silicon.

    python run.py "What is the remainder when 7^100 is divided by 13?"
    python run.py --model model-base --raw "The capital of France is"
"""
import argparse
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="*", help="the question (or completion prefix with --raw)")
    ap.add_argument("--model", default="model", help="model directory")
    ap.add_argument("--raw", action="store_true", help="plain completion, no chat template")
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="Violetto reasons inside <think> before answering; give it room")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    text = " ".join(args.prompt).strip()
    if not text:
        ap.error("no prompt given")
    if args.seed is not None:
        mx.random.seed(args.seed)

    model, tokenizer = load(args.model)
    if args.raw:
        prompt = tokenizer.encode(text)
    else:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": args.system},
             {"role": "user", "content": text}],
            add_generation_prompt=True,
        )

    sampler = make_sampler(temp=args.temp, top_p=args.top_p)
    n = 0
    for resp in stream_generate(model, tokenizer, prompt=prompt,
                                max_tokens=args.max_tokens, sampler=sampler):
        print(resp.text, end="", flush=True)
        n += 1
    print()
    print(f"\n[{n} tokens, {resp.generation_tps:.1f} tok/s, "
          f"peak {resp.peak_memory:.2f} GB]", file=sys.stderr)


if __name__ == "__main__":
    main()
