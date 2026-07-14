"""Ouro LoopLM (ByteDance) for Apple MLX.

Looped decoder-only transformer: the same stack of layers is applied
``total_ut_steps`` times per token ("latent reasoning"). Each (step, layer)
pair keeps its own KV cache entry, so the cache holds
``num_hidden_layers * total_ut_steps`` slots.

Reference: https://huggingface.co/ByteDance/Ouro-2.6B-Thinking
Paper: "Scaling Latent Reasoning via Looped Language Models" (arXiv:2510.25741)
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


@dataclasses.dataclass
class OuroConfig:
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    vocab_size: int = 49152
    total_ut_steps: int = 4
    tie_word_embeddings: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> "OuroConfig":
        raw = json.loads(Path(path).read_text())
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in fields})


class KVCache:
    """Growing K/V buffer (256-token increments), mlx-lm style."""

    step = 256

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        prev = self.offset
        new = keys.shape[2]
        if self.keys is None or prev + new > self.keys.shape[2]:
            grow = ((new + self.step - 1) // self.step) * self.step
            b, h, _, d = keys.shape
            new_k = mx.zeros((b, h, grow, d), keys.dtype)
            new_v = mx.zeros((b, h, grow, d), values.dtype)
            if self.keys is None:
                self.keys, self.values = new_k, new_v
            else:
                self.keys = mx.concatenate([self.keys[..., :prev, :], new_k], axis=2)
                self.values = mx.concatenate([self.values[..., :prev, :], new_v], axis=2)
        self.keys[..., prev:prev + new, :] = keys
        self.values[..., prev:prev + new, :] = values
        self.offset += new
        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]


class OuroAttention(nn.Module):
    def __init__(self, cfg: OuroConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.scale = cfg.head_dim ** -0.5
        dim = cfg.hidden_size
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, cfg.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, cfg.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=cfg.rope_theta)

    def __call__(self, x: mx.array, mask, cache: KVCache | None) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update(k, v)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class OuroMLP(nn.Module):
    def __init__(self, cfg: OuroConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class OuroDecoderLayer(nn.Module):
    """Sandwich-norm block: res + norm2(attn(norm1(h))); res + norm4(mlp(norm3(h)))."""

    def __init__(self, cfg: OuroConfig):
        super().__init__()
        self.self_attn = OuroAttention(cfg)
        self.mlp = OuroMLP(cfg)
        eps = cfg.rms_norm_eps
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=eps)
        self.input_layernorm_2 = nn.RMSNorm(cfg.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=eps)
        self.post_attention_layernorm_2 = nn.RMSNorm(cfg.hidden_size, eps=eps)

    def __call__(self, x: mx.array, mask, cache: KVCache | None) -> mx.array:
        h = self.input_layernorm_2(self.self_attn(self.input_layernorm(x), mask, cache))
        x = x + h
        h = self.post_attention_layernorm_2(self.mlp(self.post_attention_layernorm(x)))
        return x + h


class OuroModel(nn.Module):
    def __init__(self, cfg: OuroConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [OuroDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.early_exit_gate = nn.Linear(cfg.hidden_size, 1, bias=True)


class OuroForCausalLM(nn.Module):
    def __init__(self, cfg: OuroConfig, ut_steps: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.ut_steps = ut_steps or cfg.total_ut_steps
        self.model = OuroModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        # a dedicated KV slot per (loop step, layer) pair
        return [KVCache() for _ in range(self.ut_steps * self.cfg.num_hidden_layers)]

    def __call__(self, tokens: mx.array, caches: list[KVCache]) -> mx.array:
        h = self.model.embed_tokens(tokens)
        L = tokens.shape[1]
        mask = "causal" if L > 1 else None
        n_layers = self.cfg.num_hidden_layers
        for step in range(self.ut_steps):
            for i, layer in enumerate(self.model.layers):
                h = layer(h, mask, caches[step * n_layers + i])
            # each step's output is normed and becomes the next step's input
            h = self.model.norm(h)
        # early_exit_threshold=1.0 in the model config => logits from the last step
        return self.lm_head(h)


def load_model(model_dir: str | Path, ut_steps: int | None = None) -> OuroForCausalLM:
    """Load a converted model directory (config.json + weights.safetensors)."""
    model_dir = Path(model_dir)
    cfg = OuroConfig.from_json(model_dir / "config.json")
    model = OuroForCausalLM(cfg, ut_steps=ut_steps)
    quant = json.loads((model_dir / "config.json").read_text()).get("quantization")
    if quant:
        nn.quantize(model, group_size=quant["group_size"], bits=quant["bits"],
                    class_predicate=lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
                    and "early_exit_gate" not in p)
    model.load_weights(str(model_dir / "weights.safetensors"), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model
