"""Train our own BPE tokenizer, 32768 tokens, from real text.

It learns merges from FineWeb-Edu plus the identity lines, so the
vocab belongs to this model and not to GPT-2 or Llama. Run it once
in Colab before training, it only takes a few minutes.

Usage:
  python train_tokenizer.py --out tokenizer.json --num-docs 50000
  python train_tokenizer.py --from-files train.txt --out tokenizer.json
"""
import argparse
import os

from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def iter_texts(num_docs, from_files=None):
    # identity lines first, repeated a little, so they tokenize cleanly
    here = os.path.dirname(os.path.abspath(__file__))
    ident = os.path.join(here, "identity_data.txt")
    if os.path.exists(ident):
        with open(ident, encoding="utf-8") as f:
            text = f.read()
        for _ in range(200):
            yield text
    if from_files:
        for path in from_files:
            with open(path, encoding="utf-8") as f:
                yield f.read()
        return
    # otherwise pull a slice of FineWeb-Edu, the same data we train on
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                      split="train", streaming=True)
    n = 0
    for row in ds:
        yield row["text"]
        n += 1
        if n >= num_docs:
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tokenizer.json")
    p.add_argument("--num-docs", type=int, default=50000)
    p.add_argument("--from-files", nargs="*", default=None)
    p.add_argument("--vocab-size", type=int, default=32768)
    args = p.parse_args()

    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"],
        show_progress=True,
    )
    print(f"training BPE, vocab {args.vocab_size} ...", flush=True)
    tok.train_from_iterator(iter_texts(args.num_docs, args.from_files),
                            trainer=trainer)
    tok.save(args.out)
    print(f"saved {args.out}")
    print("sample:", tok.encode("MNAI was created by MeowSamOPDZN Studio.").tokens)


if __name__ == "__main__":
    main()
