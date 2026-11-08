"""Build train/dev/test JSONL: row-joined OCR text in, labelled fields as JSON out.

    python -m slm.prepare --cord data/cord --cascade-cache ../selfProjects/01-receipt-cascade/cache/ocr_boxes

Dev and test reuse the OCR boxes the receipt-cascade run cached, so test inputs are byte-for-byte
the text gpt-4o-mini saw. Train receipts are OCR'd here with the same RapidOCR setup.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .cord import load_parquets, ocr_boxes, parse_truth, rows_to_text, target_json

ROOT = Path(__file__).resolve().parents[1]

SYSTEM = (
    "You extract data from receipt OCR text. The text may contain recognition errors; boxes on the "
    "same printed row are joined on one line. Copy amounts exactly as printed (keep dots and commas). "
    "line_items has one entry per purchased product with its quantity and line price. subtotal, tax "
    "and total are null if the receipt does not print them. Reply with JSON only: "
    '{"line_items": [{"name": str, "quantity": int, "price": str}], "subtotal": str|null, '
    '"tax": str|null, "total": str|null}'
)


def build(receipts, cache: Path, out: Path) -> int:
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for r in receipts:
            text = rows_to_text(ocr_boxes(r.image_bytes, cache, r.sha))
            f.write(json.dumps({
                "id": r.id, "sha": r.sha, "ocr": text,
                "target": target_json(r.gt_parse), "truth": parse_truth(r.gt_parse),
            }) + "\n")
            n += 1
            if n % 100 == 0:
                print(out.name, n, flush=True)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cord", default=str(ROOT / "data" / "cord"), help="folder with the CORD v2 parquet files")
    ap.add_argument("--cascade-cache", default="", help="receipt-cascade cache/ocr_boxes to reuse")
    a = ap.parse_args()
    cord, cache = Path(a.cord), ROOT / "data" / "ocr_boxes"
    cache.mkdir(parents=True, exist_ok=True)
    if a.cascade_cache:
        for p in Path(a.cascade_cache).glob("*.json"):
            if not (cache / p.name).exists():
                shutil.copy(p, cache / p.name)
    for split, pattern in (("dev", "validation-*.parquet"), ("test", "test-*.parquet"), ("train", "train-*.parquet")):
        files = list(cord.glob(pattern))
        if not files:
            print("missing", split)
            continue
        n = build(load_parquets(files, split), cache, ROOT / "data" / f"{split}.jsonl")
        print(split, n)


if __name__ == "__main__":
    main()
