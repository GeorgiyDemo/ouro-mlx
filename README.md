# ouro-mlx

**[Ouro LoopLM](https://huggingface.co/ByteDance/Ouro-2.6B-Thinking) inference on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).**

Ouro is ByteDance's *looped language model*: the same 48-layer stack is applied
`total_ut_steps` (default 4) times per token, performing iterative "latent
reasoning". A 2.6B model punches near 8B-class weight — but neither
llama.cpp nor Ollama support the looped architecture
([ollama#14252](https://github.com/ollama/ollama/issues/14252)), and the
reference HF Transformers implementation is slow on Macs.

This is a faithful, dependency-light MLX port.

## Verified parity

Greedy decoding in bf16 produces **token-identical output** to the reference
`transformers` implementation (checked on ByteDance/Ouro-2.6B-Thinking).
8-bit quantization also matched the reference token-for-token on our probes;
4-bit shows visible quality degradation — small models quantize poorly, use
with care.

## Benchmarks (M5 Pro, 24 GB, ut_steps=4, 150 new tokens)

| Variant | Size | Prefill | Generation | vs HF Transformers (MPS) |
|---|---|---|---|---|
| HF Transformers bf16 | 5.2 GB | — | 7.2 tok/s | 1× |
| MLX bf16 | 5.0 GB | 66 tok/s | 11.5 tok/s | 1.6× |
| MLX 8-bit | 2.6 GB | 166 tok/s | **19.4 tok/s** | 2.7× |
| MLX 4-bit | 1.4 GB | 175 tok/s | 30.2 tok/s | 4.2× |

Fun fact: at `--steps 1` the model generates fluent gibberish at 75 tok/s —
the loop *is* the intelligence.

## Quickstart

```bash
git clone https://github.com/GeorgiyDemo/ouro-mlx && cd ouro-mlx
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python mlx transformers jinja2

# one-time: download reference weights + convert (8-bit recommended)
.venv/bin/huggingface-cli download ByteDance/Ouro-2.6B-Thinking
.venv/bin/python -m ouro_mlx.convert -o models/ouro-8bit -q 8

# generate
.venv/bin/python -m ouro_mlx.generate models/ouro-8bit "Explain deadlocks briefly" \
    --max-tokens 300 --temp 0.7
```

Options: `--steps N` (loop depth; fewer = faster & dumber), `--greedy`,
`--raw` (no chat template), `--top-p`.

## Implementation notes

- Sandwich-norm decoder blocks (`res + norm2(attn(norm1(x)))`), full MHA
  (16 heads, head_dim 128), RoPE θ=1e6, SwiGLU, RMSNorm.
- KV cache keeps a slot per **(loop step, layer)** pair — 192 slots for the
  2.6B; growing-buffer implementation in the style of mlx-lm.
- With the model's default `early_exit_threshold=1.0`, logits always come from
  the final loop step, so the exit gates are loaded but not evaluated.
- Weight names match the HF checkpoint exactly — conversion is remap-free.

## Credits & license

Model by [ByteDance](https://huggingface.co/ByteDance/Ouro-2.6B-Thinking)
(Apache-2.0) — paper: [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741).
This port: Apache-2.0.
