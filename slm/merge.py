"""Merge the LoRA adapter into the base weights (bf16) so vLLM can serve one plain model.

    python -m slm.merge --model Qwen/Qwen2.5-1.5B-Instruct --adapter out/qwen1.5b-cord/adapter --out out/qwen1.5b-cord/merged
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    merged.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.adapter).save_pretrained(a.out)
    print("merged into", a.out)


if __name__ == "__main__":
    main()
