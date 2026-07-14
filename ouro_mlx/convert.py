"""Convert Ouro weights from a HF snapshot to MLX format (optional quantization).

Usage:
    python -m ouro_mlx.convert --hf-path <HF snapshot> -o models/ouro-2.6b-thinking-8bit -q 8
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .model import OuroConfig, OuroForCausalLM

DEFAULT_HF_GLOB = (
    "~/.cache/huggingface/hub/models--ByteDance--Ouro-2.6B-Thinking/snapshots/*"
)


def find_hf_snapshot(path: str | None) -> Path:
    if path:
        return Path(path).expanduser()
    hits = sorted(glob.glob(str(Path(DEFAULT_HF_GLOB).expanduser())))
    if not hits:
        raise SystemExit("Snapshot not found. Download the model or pass --hf-path")
    return Path(hits[-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-path", default=None,
                   help="path to the HF snapshot (default: huggingface cache)")
    p.add_argument("-o", "--out", required=True, help="output directory")
    p.add_argument("-q", "--quantize", type=int, choices=[4, 8], default=None,
                   help="weight bits (omit for bf16)")
    p.add_argument("--group-size", type=int, default=64)
    args = p.parse_args()

    src = find_hf_snapshot(args.hf_path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = OuroConfig.from_json(src / "config.json")
    model = OuroForCausalLM(cfg)

    print(f"loading weights from {src} …")
    weights = mx.load(str(src / "model.safetensors"))
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())

    if args.quantize:
        print(f"quantizing to {args.quantize} bits (group_size={args.group_size}) …")
        nn.quantize(model, group_size=args.group_size, bits=args.quantize,
                    class_predicate=lambda p_, m: isinstance(m, (nn.Linear, nn.Embedding))
                    and "early_exit_gate" not in p_)

    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out / "weights.safetensors"), flat)

    raw_cfg = json.loads((src / "config.json").read_text())
    raw_cfg.pop("auto_map", None)  # otherwise AutoTokenizer asks for trust_remote_code
    if args.quantize:
        raw_cfg["quantization"] = {"bits": args.quantize, "group_size": args.group_size}
    (out / "config.json").write_text(json.dumps(raw_cfg, indent=2))

    for name in ("tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "vocab.json", "merges.txt"):
        if (src / name).exists():
            shutil.copy(src / name, out / name)

    size = sum(f.stat().st_size for f in out.glob("*")) / 2**30
    print(f"✓ {out} ({size:.1f} GB)")


if __name__ == "__main__":
    main()
