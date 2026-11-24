"""Turn a trained HF checkpoint into a GGUF, the LM Studio / llama.cpp way.

It clones llama.cpp if needed, runs the official converter, then
quantizes to Q4_K_M, the usual sweet spot: a 0.9B model becomes
about 550MB and runs on a 4GB RAM laptop.

Usage, in Colab or locally:
  python convert_to_gguf.py --model out-09b/ckpt-500 --out mnai-nano-2-q4_k_m.gguf
  python convert_to_gguf.py --model out-09b/ckpt-500 --out mnai-nano-2-f16.gguf --quantize f16
"""
import argparse
import os
import subprocess
import sys


def run(cmd, cwd=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def installed_version(pkg):
    # fresh interpreter each call, so a pip install in this process
    # is visible right away instead of reading stale metadata
    out = subprocess.run(
        [sys.executable, "-c",
         f"from importlib.metadata import version; print(version({pkg!r}))"],
        capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="mnai-nano-2-q4_k_m.gguf")
    p.add_argument("--quantize", default="q4_k_m")
    p.add_argument("--llama-cpp", default="llama.cpp")
    args = p.parse_args()

    if not os.path.isdir(args.llama_cpp):
        run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", args.llama_cpp])
    # llama.cpp's own requirements pin transformers==4.57.6, and installing
    # them would silently downgrade the transformers this run started with.
    # That breaks checkpoints saved by transformers 5: their
    # tokenizer_config.json says "TokenizersBackend", a class 4.x has never
    # heard of, so the converter dies with "Tokenizer class TokenizersBackend
    # does not exist". Remember the version and put it back afterwards.
    tf_before = installed_version("transformers")
    run([sys.executable, "-m", "pip", "install", "-q", "-r",
         os.path.join(args.llama_cpp, "requirements.txt")])
    tf_after = installed_version("transformers")
    if tf_before and tf_after != tf_before:
        print(f"llama.cpp requirements moved transformers "
              f"{tf_before} -> {tf_after}, restoring {tf_before}", flush=True)
        run([sys.executable, "-m", "pip", "install", "-q",
             f"transformers=={tf_before}"])

    f16 = args.out + ".f16.tmp.gguf" if args.quantize.lower() != "f16" else args.out
    run([sys.executable, os.path.join(args.llama_cpp, "convert_hf_to_gguf.py"),
         args.model, "--outfile", f16, "--outtype", "f16"])

    if args.quantize.lower() != "f16":
        quant = os.path.join(args.llama_cpp, "llama-quantize")
        if not os.path.exists(quant):
            # -j 2, not unbounded -j: the unity build files are huge and
            # one cc1plus can eat several GB, so a free -j OOM-kills the
            # build on a small box (proved on a 6GB VM, cc1plus killed)
            run(["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"],
                cwd=args.llama_cpp)
            run(["cmake", "--build", "build", "-j", "2", "--target", "llama-quantize"],
                cwd=args.llama_cpp)
            quant = os.path.join(args.llama_cpp, "build", "bin", "llama-quantize")
        run([quant, f16, args.out, args.quantize.upper()])
        os.remove(f16)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"done: {args.out} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
