"""Reference HF Transformers runner for Ouro-2.6B-Thinking (parity testing).

Used to verify that the MLX port produces token-identical greedy output.
Compare:
    python reference/run_transformers.py "The capital of France is" --greedy --raw
    python -m ouro_mlx.generate models/ouro-bf16 "The capital of France is" --greedy --raw

SETUP (the reference implementation is version-sensitive):
  1. pip install "transformers==4.54.1" torch accelerate
     (>=4.56 breaks the custom KV cache; <=4.53 misses required imports)
  2. Even on 4.54.1 the model's custom code crashes with
     "property 'key_cache' of 'UniversalTransformerCache' object has no setter".
     Patch the cached modeling file
     (~/.cache/huggingface/modules/transformers_modules/ByteDance/Ouro-2.6B-Thinking/*/modeling_ouro.py)
     by adding two class attributes right after
     `class UniversalTransformerCache(Cache):`:
         key_cache = None
         value_cache = None
     Re-downloading the model resets the patch.
"""
import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "ByteDance/Ouro-2.6B-Thinking"
TRAINED_UT_STEPS = 4  # loop depth the model was trained with


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("--steps", type=int, default=None,
                   help="total_ut_steps: number of loop passes (config default = 4)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--raw", action="store_true",
                   help="no chat template (raw text continuation)")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", file=sys.stderr)

    t0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    kwargs = {"total_ut_steps": args.steps} if args.steps else {}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, **kwargs
    ).to(device)
    print(f"load: {time.monotonic() - t0:.0f}s | "
          f"ut_steps={getattr(model.config, 'total_ut_steps', '?')}", file=sys.stderr)
    if args.steps and args.steps > TRAINED_UT_STEPS:
        print(f"⚠ model was trained with total_ut_steps={TRAINED_UT_STEPS}; "
              f"running {args.steps} loop steps is out-of-distribution and "
              f"degrades output quality", file=sys.stderr)

    if args.raw:
        enc = tok(args.prompt, return_tensors="pt")
    else:
        enc = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                      add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    t1 = time.monotonic()
    with torch.no_grad():
        out = model.generate(input_ids, attention_mask=attention_mask,
                             max_new_tokens=args.max_tokens,
                             do_sample=not args.greedy, temperature=0.7, top_p=0.95,
                             pad_token_id=tok.eos_token_id)
    gen = out[0][input_ids.shape[1]:]
    dt = time.monotonic() - t1
    print(tok.decode(gen, skip_special_tokens=True))
    print(f"\n--- {len(gen)} tokens in {dt:.0f}s ({len(gen) / dt:.1f} tok/s) ---",
          file=sys.stderr)


if __name__ == "__main__":
    main()
