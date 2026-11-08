"""CORD v2 loading, OCR text, ground truth and scoring.

Copied from my receipt-cascade project (github.com/masooddahmedd/receipt-cascade) so the
fine-tuned model is scored with exactly the same rules, on the same OCR text, as the
gpt-4o-mini and gpt-4o numbers it is compared against.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

NAME_SIMILARITY = 0.8
SCORED_FIELDS = ["line_items", "subtotal", "tax", "total"]


def to_amount(value) -> int | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    return int(digits) if digits else None


def name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def as_text(value) -> str:
    if isinstance(value, list):
        return " ".join(as_text(v) for v in value)
    return "" if value is None else str(value)


def flatten_menu(menu) -> list[dict]:
    if isinstance(menu, dict):
        menu = [menu]
    items = []
    for m in menu:
        items.append(m)
        items.extend(flatten_menu(m["sub"]) if "sub" in m else [])
    return items


def parse_truth(gt_parse: dict) -> dict:
    """Comparable ground truth: amounts as ints, item names as lowercase alphanumerics."""
    truth = {}
    items = [m for m in flatten_menu(gt_parse.get("menu", []))
             if as_text(m.get("nm")) and to_amount(as_text(m.get("price"))) is not None]
    if items:
        truth["line_items"] = [(name_key(as_text(m["nm"])), to_amount(as_text(m.get("cnt"))) or 1,
                                to_amount(as_text(m["price"]))) for m in items]
    sub = gt_parse.get("sub_total") or {}
    if isinstance(sub, dict):
        for field, key in [("subtotal", "subtotal_price"), ("tax", "tax_price")]:
            if to_amount(as_text(sub.get(key))) is not None:
                truth[field] = to_amount(as_text(sub[key]))
    total = gt_parse.get("total") or {}
    if isinstance(total, dict) and to_amount(as_text(total.get("total_price"))) is not None:
        truth["total"] = to_amount(as_text(total["total_price"]))
    return truth


def target_json(gt_parse: dict) -> dict:
    """What the model should output: the labelled fields, amounts as printed on the receipt."""
    items = [m for m in flatten_menu(gt_parse.get("menu", []))
             if as_text(m.get("nm")) and to_amount(as_text(m.get("price"))) is not None]
    sub = gt_parse.get("sub_total") if isinstance(gt_parse.get("sub_total"), dict) else {}
    total = gt_parse.get("total") if isinstance(gt_parse.get("total"), dict) else {}

    def printed(d, key):
        v = as_text(d.get(key)).strip()
        return v if to_amount(v) is not None else None

    return {
        "line_items": [{"name": as_text(m["nm"]).strip(), "quantity": to_amount(as_text(m.get("cnt"))) or 1,
                        "price": as_text(m["price"]).strip()} for m in items],
        "subtotal": printed(sub, "subtotal_price"),
        "tax": printed(sub, "tax_price"),
        "total": printed(total, "total_price"),
    }


def canonical(pred: dict, field: str):
    if field == "line_items":
        out = []
        for i in pred.get("line_items") or []:
            try:
                out.append((name_key(str(i.get("name", ""))), int(i.get("quantity") or 1), to_amount(i.get("price"))))
            except (TypeError, ValueError, AttributeError):
                continue
        return tuple(sorted(out))
    return to_amount(pred.get(field))


def items_match(pred, truth) -> bool:
    if len(pred) != len(truth):
        return False
    left = list(truth)
    for name, qty, price in pred:
        hit = next((t for t in left if t[1] == qty and t[2] == price
                    and SequenceMatcher(None, name, t[0]).ratio() >= NAME_SIMILARITY), None)
        if hit is None:
            return False
        left.remove(hit)
    return True


def is_correct(field: str, value, truth: dict) -> bool:
    if field == "line_items":
        return items_match(value, truth["line_items"])
    return value == truth[field]


def rows_to_text(boxes: list) -> str:
    """OCR boxes on the same visual row joined into one line (the cascade's 'layout' text)."""
    items = []
    for quad, text in boxes:
        ys = [p[1] for p in quad]
        xs = [p[0] for p in quad]
        items.append((sum(ys) / 4, max(ys) - min(ys), min(xs), text))
    if not items:
        return ""
    items.sort()
    typical_height = float(np.median([h for _, h, _, _ in items]))
    rows, current, row_y = [], [], None
    for y, _, x, text in items:
        if row_y is not None and abs(y - row_y) > 0.6 * typical_height:
            rows.append(current)
            current = []
        current.append((x, text))
        row_y = y if len(current) == 1 else row_y
    rows.append(current)
    return "\n".join("    ".join(t for _, t in sorted(r)) for r in rows)


_engine = None


def ocr_boxes(image_bytes: bytes, cache_dir: Path, sha: str) -> list:
    global _engine
    path = cache_dir / f"{sha}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    from PIL import Image
    result, _ = _engine(np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB")))
    boxes = [[b, t] for b, t, _ in (result or [])]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(boxes), encoding="utf-8")
    return boxes


@dataclass
class Receipt:
    id: str
    sha: str
    image_bytes: bytes
    gt_parse: dict


def load_parquets(files: list[Path], split: str) -> list[Receipt]:
    import pandas as pd
    out = []
    i = 0
    for f in sorted(files):
        for _, row in pd.read_parquet(f).iterrows():
            raw = row["image"]["bytes"]
            out.append(Receipt(f"{split}-{i:03d}", hashlib.sha256(raw).hexdigest(), raw,
                               json.loads(row["ground_truth"])["gt_parse"]))
            i += 1
    return out
