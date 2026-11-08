"""QLoRA fine-tune of a small instruct model on CORD receipts (OCR text -> JSON).

    python -m slm.train --model Qwen/Qwen2.5-1.5B-Instruct --out out/qwen1.5b-cord

Only the assistant's JSON is trained on; the system prompt and OCR text are masked out.
Memory is kept low for a shared 16 GB card: 4-bit base weights, gradient checkpointing,
batch size 1 with accumulation, and sequences capped at --max-len tokens.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

from .prepare import SYSTEM

ROOT = Path(__file__).resolve().parents[1]


def messages(row: dict, with_answer: bool = True) -> list[dict]:
    m = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["ocr"]}]
    if with_answer:
        m.append({"role": "assistant", "content": json.dumps(row["target"], ensure_ascii=False)})
    return m


def encode(tok, row: dict, max_len: int) -> dict | None:
    prompt = tok.apply_chat_template(messages(row, False), tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(messages(row, True), tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    if len(f_ids) > max_len:
        return None                           # skip rather than train on a cut-off answer
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return {"input_ids": f_ids, "labels": labels}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default=str(ROOT / "out" / "qwen1.5b-cord"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1280)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(l) for l in open(ROOT / "data" / "train.jsonl", encoding="utf-8")]
    data = [e for e in (encode(tok, r, a.max_len) for r in rows) if e]
    print(f"{len(data)} of {len(rows)} train examples fit in {a.max_len} tokens", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        a.model, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True),
        dtype=torch.bfloat16, device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = math.ceil(len(data) * a.epochs / a.accum)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)
    model.train()
    log = []
    start = time.time()
    step = 0
    for epoch in range(a.epochs):
        random.shuffle(data)
        running, n = 0.0, 0
        for i, ex in enumerate(data, 1):
            ids = torch.tensor([ex["input_ids"]], device=0)
            labels = torch.tensor([ex["labels"]], device=0)
            loss = model(input_ids=ids, labels=labels).loss / a.accum
            loss.backward()
            running += loss.item() * a.accum
            n += 1
            if i % a.accum == 0 or i == len(data):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    rec = {"epoch": epoch + 1, "step": step, "loss": round(running / n, 4),
                           "minutes": round((time.time() - start) / 60, 1),
                           "gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
                    log.append(rec)
                    print(json.dumps(rec), flush=True)
                    running, n = 0.0, 0
    out = Path(a.out)
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    (out / "train_log.json").write_text(json.dumps({"args": vars(a), "examples": len(data), "log": log,
                                                    "minutes": round((time.time() - start) / 60, 1)}, indent=2))
    print("saved", out / "adapter")


if __name__ == "__main__":
    main()
