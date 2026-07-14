"""Text generation with Ouro LoopLM on MLX.

Usage:
    python -m ouro_mlx.generate models/ouro-2.6b-thinking-8bit "prompt" \
        [--steps N] [--max-tokens N] [--temp T] [--raw] [--greedy]
"""
from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from .model import load_model

hf_logging.set_verbosity_error()


def sample(logits: mx.array, temp: float, top_p: float) -> mx.array:
    if temp <= 0:
        return mx.argmax(logits, axis=-1)
    logits = logits / temp
    if top_p < 1.0:
        probs = mx.softmax(logits, axis=-1)
        order = mx.argsort(-probs, axis=-1)
        sorted_probs = mx.take_along_axis(probs, order, axis=-1)
        cum = mx.cumsum(sorted_probs, axis=-1)
        cut = cum - sorted_probs > top_p  # mask the tail outside the nucleus
        sorted_probs = mx.where(cut, mx.zeros_like(sorted_probs), sorted_probs)
        masked = mx.zeros_like(probs)
        masked = mx.put_along_axis(masked, order, sorted_probs, axis=-1)
        return mx.random.categorical(mx.log(masked + 1e-12), axis=-1)
    return mx.random.categorical(logits, axis=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir")
    p.add_argument("prompt")
    p.add_argument("--steps", type=int, default=None,
                   help="number of loop passes (default from config = 4)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--raw", action="store_true",
                   help="no chat template (raw text continuation)")
    args = p.parse_args()

    t0 = time.monotonic()
    model = load_model(args.model_dir, ut_steps=args.steps)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    print(f"load: {time.monotonic() - t0:.1f}s | ut_steps={model.ut_steps}",
          file=sys.stderr)
    if model.ut_steps > model.cfg.total_ut_steps:
        print(f"⚠ model was trained with total_ut_steps={model.cfg.total_ut_steps}; "
              f"running {model.ut_steps} loop steps is out-of-distribution and "
              f"degrades output quality", file=sys.stderr)

    if args.raw:
        ids = tok(args.prompt, return_tensors="np")["input_ids"][0].tolist()
    else:
        ids = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                      add_generation_prompt=True,
                                      tokenize=True, return_dict=False)
    tokens = mx.array([ids])
    caches = model.make_cache()

    t1 = time.monotonic()
    logits = model(tokens, caches)[:, -1, :]
    mx.eval(logits)
    prefill_s = time.monotonic() - t1

    temp = 0.0 if args.greedy else args.temp
    eos_ids = {tok.eos_token_id}
    out_ids: list[int] = []
    printed = ""
    t2 = time.monotonic()
    for _ in range(args.max_tokens):
        next_tok = sample(logits, temp, args.top_p)
        mx.eval(next_tok)
        tid = int(next_tok.item())
        if tid in eos_ids:
            break
        out_ids.append(tid)
        # per-token decode breaks multi-byte UTF-8 (e.g. Cyrillic) — decode the
        # accumulated ids and print only the completed suffix
        text = tok.decode(out_ids)
        if not text.endswith("�"):
            print(text[len(printed):], end="", flush=True)
            printed = text
        logits = model(next_tok.reshape(1, 1), caches)[:, -1, :]
    dt = time.monotonic() - t2
    print()
    print(f"--- prefill {len(ids)} tok in {prefill_s:.1f}s "
          f"({len(ids) / prefill_s:.0f} tok/s) | "
          f"generation {len(out_ids)} tok in {dt:.1f}s "
          f"({len(out_ids) / max(dt, 1e-6):.1f} tok/s) ---", file=sys.stderr)


if __name__ == "__main__":
    main()
