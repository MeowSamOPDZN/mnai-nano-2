"""MNAI Nano 2, the 0.9B model. Own shape, trained from scratch.

The shape is ours: 1536 hidden, 24 layers, 16 query heads with only
4 key/value heads, a 6400-wide SwiGLU feed-forward, RoPE positions,
RMSNorm, no biases, tied embeddings, 32768-token vocab, 4096 context.
It adds up to 899,753,472 parameters, so 0.90B.

Why this shape and not a copy of something else: the GQA ratio keeps
attention cheap, the wide SwiGLU carries most of the capacity, and
tying the output head to the input embeddings saves about 50M params
for free. It is built on the standard Llama block layout on purpose,
so llama.cpp can turn the finished checkpoint into a GGUF with its
normal converter, no custom converter needed.
"""
import json
import os

import torch
from transformers import LlamaConfig, LlamaForCausalLM

# the shape, all in one place so train.py and generate.py agree
VOCAB_SIZE = 32768
HIDDEN_SIZE = 1536
NUM_LAYERS = 24
NUM_HEADS = 16
NUM_KV_HEADS = 4
INTERMEDIATE_SIZE = 6400
MAX_CTX = 4096
ROPE_THETA = 500000.0
RMS_NORM_EPS = 1e-5


def build_config():
    return LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=MAX_CTX,
        rope_theta=ROPE_THETA,
        rms_norm_eps=RMS_NORM_EPS,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


def build_model(dtype=torch.bfloat16):
    cfg = build_config()
    model = LlamaForCausalLM(cfg)
    if dtype is not None:
        model = model.to(dtype)
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def describe(model):
    total, _ = count_params(model)
    lines = [
        f"vocab {VOCAB_SIZE}, hidden {HIDDEN_SIZE}, layers {NUM_LAYERS}",
        f"heads {NUM_HEADS}, kv heads {NUM_KV_HEADS}, ffn {INTERMEDIATE_SIZE}",
        f"context {MAX_CTX}, rope theta {int(ROPE_THETA)}",
        f"params {total:,} ({total / 1e9:.3f}B), tied embeddings: yes",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    model = build_model(dtype=torch.float32)
    print(describe(model))
    print()
    # tiny forward pass, just to prove the shape runs
    x = torch.randint(0, VOCAB_SIZE, (1, 16))
    out = model(x)
    print("logits shape:", tuple(out.logits.shape))
    print("expected last dim:", VOCAB_SIZE)
    # save the config next to the code so Colab and HF agree
    cfg = build_config()
    cfg.save_pretrained(os.path.join(here, "hf_config_only"))
    print("wrote hf_config_only/config.json (config only, no weights)")
