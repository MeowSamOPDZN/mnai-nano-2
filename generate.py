"""Chat with a trained Nano 2 checkpoint. Loads HF weights, samples text.

Usage:
  python generate.py --model out-09b/ckpt-500 --prompt "MNAI was created by"
  python generate.py --model out-09b/ckpt-500 --prompt "The sun rises in" --max-new 100 --temp 0.8
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="MNAI was created by")
    p.add_argument("--max-new", type=int, default=120)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=args.max_new,
                             do_sample=True, temperature=args.temp,
                             top_k=args.top_k, pad_token_id=tok.pad_token_id)
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
