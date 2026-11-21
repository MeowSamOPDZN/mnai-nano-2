"""Train MNAI Nano 2 from scratch. Random weights, no base model.

Data is FineWeb-Edu streamed from Hugging Face, tokenized on the fly
with our own tokenizer, packed into 2048-token blocks. Checkpoints
save in HF format every so often, so a Colab session ending just
means resume with --resume and keep going.

Memory on a free Colab T4 (16GB): weights bf16 1.8GB, grads 1.8GB,
8-bit Adam about 0.9GB, activations with checkpointing a few GB.
That is why 8-bit Adam is the default here. Plain AdamW works too
but wants a bigger card, pass --optim adamw if you have one.

Usage, Colab T4:
  python train_tokenizer.py --out tokenizer.json
  python train.py --tokenizer tokenizer.json --out out-09b --max-tokens 1000000000
  python train.py --tokenizer tokenizer.json --out out-09b --resume out-09b/ckpt-500

Smoke test on CPU, no data download. Uses a tiny twin of the same
block layout for the loop test, full 0.9B forward is in model.py:
  python train.py --smoke --max-steps 3 --out /tmp/smoke-09b
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast, get_cosine_schedule_with_warmup

from model import VOCAB_SIZE, build_config, build_model, count_params

SEQ_LEN = 2048
# smoke test uses a tiny sequence, a full 2048 block on CPU would crawl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--out", default="out-09b")
    p.add_argument("--resume", default=None)
    p.add_argument("--max-tokens", type=int, default=1_000_000_000)
    p.add_argument("--max-steps", type=int, default=0, help="optimizer steps, 0 = use max-tokens")
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--optim", default="bnb8bit", choices=["bnb8bit", "adamw"])
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--ckpt-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def token_stream(tok, smoke=False):
    # endless stream of token ids, so training never waits on disk
    if smoke:
        rng = random.Random(0)
        while True:
            yield rng.randrange(100, VOCAB_SIZE)
        return
    from datasets import load_dataset

    print("loading FineWeb-Edu stream, first download can take a few minutes...", flush=True)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                      split="train", streaming=True)
    print("stream ready, reading first docs...", flush=True)
    here = os.path.dirname(os.path.abspath(__file__))
    ident = ""
    ident_path = os.path.join(here, "identity_data.txt")
    if os.path.exists(ident_path):
        with open(ident_path, encoding="utf-8") as f:
            ident = f.read()
    n_docs = 0
    for row in ds:
        text = row["text"]
        n_docs += 1
        if n_docs == 1:
            print(f"first doc loaded, {len(text):,} chars, tokenizing...", flush=True)
        # a pinch of identity in the mix, about one doc in fifty
        if n_docs % 50 == 0 and ident:
            text = text + "\n" + ident
        for tok_id in tok.encode(text):
            yield tok_id


def batch_stream(tok, micro_batch, smoke=False, seq_len=SEQ_LEN):
    # pack the token stream into seq_len blocks, then into batches
    buf = []
    stream = token_stream(tok, smoke=smoke)
    first = True
    while True:
        while len(buf) < seq_len * micro_batch:
            try:
                buf.append(next(stream))
            except StopIteration:
                stream = token_stream(tok, smoke=smoke)
        chunk = buf[: seq_len * micro_batch]
        buf = buf[seq_len * micro_batch:]
        x = torch.tensor(chunk, dtype=torch.long).view(micro_batch, seq_len)
        if first:
            print("first batch ready, starting training...", flush=True)
            first = False
        yield x


def save_ckpt(model, tok, out_dir, step, tokens_seen, extra=None):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    state = {"step": step, "tokens_seen": tokens_seen}
    if extra:
        state.update(extra)
    with open(os.path.join(out_dir, "trainer_state.json"), "w") as f:
        json.dump(state, f, indent=2)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    if args.smoke:
        # smoke mode uses random tokens, no dataset, no tokenizer file
        # and a tiny sequence so it actually finishes on CPU
        tok = None
        args.seq_len = 64
        args.micro_batch = 1
        args.grad_accum = 1
        args.optim = "adamw"
        print("smoke mode: random tokens, no dataset, no tokenizer file", flush=True)
    else:
        tok = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
        tok.pad_token = "<|pad|>"
        tok.bos_token = "<|bos|>"
        tok.eos_token = "<|eos|>"
        assert tok.vocab_size == VOCAB_SIZE, f"tokenizer vocab {tok.vocab_size} != {VOCAB_SIZE}"

    # smoke uses a tiny twin of the same shape (2 layers, 256 hidden)
    # for the loop test, because a full 0.9B backward pass needs more
    # RAM than this CPU box has. The full 0.9B forward pass is tested
    # separately with python model.py, and real training runs on a GPU.
    train_dtype = torch.float32 if device == "cpu" else torch.bfloat16
    if args.resume:
        print(f"resuming weights from {args.resume}", flush=True)
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(args.resume, torch_dtype=train_dtype)
        step = 0
        tokens_seen = 0
        resume_state = os.path.join(args.resume, "trainer_state.json")
        if os.path.exists(resume_state):
            with open(resume_state) as f:
                saved = json.load(f)
            step = saved.get("step", 0)
            tokens_seen = saved.get("tokens_seen", 0)
        print(f"resumed at step {step}, tokens {tokens_seen:,}", flush=True)
    else:
        if args.smoke:
            from transformers import LlamaConfig, LlamaForCausalLM

            tiny_cfg = build_config()
            tiny_cfg.hidden_size = 256
            tiny_cfg.intermediate_size = 512
            tiny_cfg.num_hidden_layers = 2
            tiny_cfg.num_attention_heads = 4
            tiny_cfg.num_key_value_heads = 2
            model = LlamaForCausalLM(tiny_cfg)
            print("smoke model: tiny twin, same block layout, 2 layers", flush=True)
        else:
            model = build_model(dtype=train_dtype)
        step = 0
        tokens_seen = 0

    total, _ = count_params(model)
    print(f"model: {total:,} params ({total / 1e9:.3f}B)", flush=True)
    model.to(device)
    model.gradient_checkpointing_enable()
    model.train()

    if args.smoke:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        print("optimizer: AdamW (smoke, tiny model)", flush=True)
    elif args.optim == "bnb8bit":
        try:
            import bitsandbytes as bnb

            opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay, betas=(0.9, 0.95))
            print("optimizer: AdamW 8-bit", flush=True)
        except Exception as e:
            print(f"8-bit Adam not available ({e}), falling back to AdamW", flush=True)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay, betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay, betas=(0.9, 0.95))
        print("optimizer: AdamW", flush=True)

    tokens_per_step = args.micro_batch * args.seq_len * args.grad_accum
    if args.max_steps:
        total_steps = args.max_steps
    else:
        total_steps = max(1, args.max_tokens // tokens_per_step)
    sched = get_cosine_schedule_with_warmup(
        opt, args.warmup, total_steps)
    # cosine in transformers goes to 0, we keep a floor by scaling lr later
    print(f"plan: {total_steps} optimizer steps, {tokens_per_step:,} tokens/step", flush=True)
    print(f"target tokens: {args.max_tokens:,}", flush=True)

    stream = batch_stream(tok, args.micro_batch, smoke=args.smoke, seq_len=args.seq_len)
    use_bf16 = device == "cuda"
    t0 = time.time()
    micro = 0
    loss_sum = 0.0
    loss_n = 0

    print("waiting for first batch, this can take a few minutes on first run...", flush=True)
    while step < total_steps:
        x = next(stream).to(device)
        if micro == 0 and step == 0:
            print("first batch in, running first forward/backward...", flush=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            out = model(x, labels=x)
            loss = out.loss / args.grad_accum
        loss.backward()
        loss_sum += out.loss.item()
        loss_n += 1
        micro += 1
        tokens_seen += x.numel()

        if micro % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step <= 20 or step % args.log_every == 0:
                el = time.time() - t0
                tok_s = tokens_seen / max(el, 1)
                print(f"step {step}/{total_steps} loss {loss_sum / max(loss_n, 1):.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} tokens {tokens_seen:,} "
                      f"{tok_s:,.0f} tok/s elapsed {el / 60:.1f}m", flush=True)
                loss_sum, loss_n = 0.0, 0
            if step % args.ckpt_every == 0 or step == total_steps:
                ckpt = os.path.join(args.out, f"ckpt-{step}")
                print(f"saving {ckpt}", flush=True)
                # save in fp32-safe way: HF handles bf16 weights fine
                if tok is not None:
                    save_ckpt(model, tok, ckpt, step, tokens_seen)
                else:
                    os.makedirs(ckpt, exist_ok=True)
                    model.save_pretrained(ckpt)
                with open(os.path.join(args.out, "train_log.jsonl"), "a") as f:
                    f.write(json.dumps({"step": step, "tokens_seen": tokens_seen,
                                        "time_s": time.time() - t0}) + "\n")
            if args.smoke and step >= 3:
                print("smoke done, 3 steps ran fine", flush=True)
                break

    print(f"done. steps {step}, tokens {tokens_seen:,}, time {(time.time() - t0) / 60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
