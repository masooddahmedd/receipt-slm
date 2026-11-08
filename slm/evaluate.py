"""Score a model on the CORD test split with the receipt-cascade rules.

    python -m slm.evaluate --base-url http://127.0.0.1:8100/v1 --model cord --name qwen1.5b-ft

Calls any OpenAI-compatible endpoint (vLLM here), temperature 0, one request per receipt with
--concurrency in flight. Writes per-receipt predictions and a summary with accuracy per field,
JSON validity, latency and tokens.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from .cord import SCORED_FIELDS, canonical, is_correct
from .prepare import SYSTEM

ROOT = Path(__file__).resolve().parents[1]


def parse(text: str) -> dict | None:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None


def truth_of(row: dict) -> dict:
    t = dict(row["truth"])
    if "line_items" in t:
        t["line_items"] = [tuple(x) for x in t["line_items"]]
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--model", default="cord")
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()

    client = OpenAI(base_url=a.base_url, api_key="none", timeout=300)
    rows = [json.loads(l) for l in open(ROOT / "data" / f"{a.split}.jsonl", encoding="utf-8")]

    def run(row):
        t = time.perf_counter()
        r = client.chat.completions.create(
            model=a.model, temperature=0, max_tokens=900,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["ocr"]}])
        return row, r.choices[0].message.content or "", time.perf_counter() - t, r.usage

    start = time.perf_counter()
    with ThreadPoolExecutor(a.concurrency) as pool:
        results = list(pool.map(run, rows))
    wall = time.perf_counter() - start

    per_field = {f: [0, 0] for f in SCORED_FIELDS}
    valid, latencies, out_tokens, records = 0, [], [], []
    for row, text, secs, usage in results:
        pred = parse(text)
        valid += pred is not None
        latencies.append(secs)
        out_tokens.append(usage.completion_tokens if usage else 0)
        truth = truth_of(row)
        rec = {"id": row["id"], "raw": text, "fields": {}}
        for f in SCORED_FIELDS:
            if f not in truth:
                continue
            ok = pred is not None and is_correct(f, canonical(pred, f), truth)
            per_field[f][0] += ok
            per_field[f][1] += 1
            rec["fields"][f] = ok
        records.append(rec)

    correct = sum(c for c, _ in per_field.values())
    total = sum(t for _, t in per_field.values())
    summary = {
        "name": a.name, "model": a.model, "split": a.split, "receipts": len(rows), "fields": total,
        "accuracy": round(correct / total, 4),
        "by_field": {f: round(c / t, 4) for f, (c, t) in per_field.items() if t},
        "valid_json": f"{valid}/{len(rows)}",
        "latency_p50_s": round(statistics.median(latencies), 2),
        "latency_p90_s": round(sorted(latencies)[int(0.9 * (len(latencies) - 1))], 2),
        "receipts_per_minute": round(len(rows) / wall * 60, 1),
        "concurrency": a.concurrency,
        "output_tokens_mean": round(statistics.mean(out_tokens)),
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"{a.name}_{a.split}.json").write_text(json.dumps(summary, indent=2))
    with open(out / f"{a.name}_{a.split}_predictions.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
