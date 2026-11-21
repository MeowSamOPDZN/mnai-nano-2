# MNAI Nano 2 — 0.9B, from scratch

A language model built from random weights, own shape, own tokenizer, own training run. No base model, no fine-tune.

Made by **MeowSamOPDZN Studio**. Owner: **MeowSamOP DZN**. Full form: **Meow x Nythera AI**.

## The model

| thing | value |
|---|---|
| params | **899,753,472 — 0.90B** |
| hidden | 1536 |
| layers | 24 |
| attention heads | 16 query, 4 key/value (GQA 4:1) |
| feed-forward | 6400, SwiGLU |
| vocab | 32768, own BPE tokenizer |
| context | 4096, RoPE theta 500k |
| norms | RMSNorm, no biases |
| embeddings | tied input/output |

Why this shape: GQA keeps attention cheap, the wide SwiGLU carries most of the capacity, tying embeddings saves ~50M params. It uses the standard Llama block layout on purpose, so llama.cpp's normal converter can make a GGUF out of it with no custom code.

Files: `model.py` is the shape and the param count, `train.py` is the training loop, `train_tokenizer.py` trains the BPE, `generate.py` samples text, `convert_to_gguf.py` makes the GGUF.

## Honest numbers first

This is a real 0.9B pretrain, so it is not free or fast on a free Colab T4.

- Chinchilla-optimal would be ~18B tokens. On a T4 at ~5–8k tokens/sec that is weeks of Colab sessions, with resume.
- A first useful run is **1B tokens**: roughly 35–55 hours of T4 time, split across sessions. It will talk, but it will not be smart like a 7B.
- Weights bf16 are 1.8GB, 8-bit Adam is ~0.9GB, grads 1.8GB. That is why 8-bit Adam is the default — plain AdamW wants a bigger card.
- Checkpoints save in HF format, so a session ending is not lost work. Resume and keep going.

Your 4GB RAM PC cannot train this. It *can* run the finished GGUF: Q4_K_M is about 550MB.

## Run it in Colab (free T4)

Open `mnai_nano2_colab.ipynb` in Colab, GPU runtime, T4. It does, in order:

1. Clone this repo, install deps.
2. Train the tokenizer: `python train_tokenizer.py --out tokenizer.json`
3. Train: `python train.py --tokenizer tokenizer.json --out out-09b --max-tokens 1000000000`
4. When a session ends, resume: `python train.py --tokenizer tokenizer.json --out out-09b --resume out-09b/ckpt-500`
5. Test it: `python generate.py --model out-09b/ckpt-500 --prompt "MNAI was created by"`
6. Make the GGUF: `python convert_to_gguf.py --model out-09b/ckpt-500 --out mnai-nano-2-q4_k_m.gguf`
7. Download the `.gguf` from Colab, load it in LM Studio.

Save checkpoints to Google Drive in the notebook if you do not want to re-download them every session. The notebook mounts Drive for that.

## Run it locally (smoke test only)

```bash
python model.py
python train.py --smoke --max-steps 3 --out /tmp/smoke-09b
```

`model.py` proves the full 0.9B shape and forward pass. The smoke test proves the training loop and checkpoint save on a tiny twin of the same layout, because a full 0.9B backward pass does not fit in CPU RAM here. Real training needs the Colab GPU.

## Data

FineWeb-Edu, `sample-10BT` split, streamed from Hugging Face, tokenized on the fly. Plus `identity_data.txt`, mixed in about 1 doc in 50, so the model learns who made it.

## License

MIT. The model weights you train are yours.
