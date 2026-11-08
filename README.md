# A 1.5B model fine-tuned for receipt extraction, compared with gpt-4o-mini and gpt-4o

Qwen2.5-1.5B-Instruct, fine-tuned with QLoRA on 800 CORD v2 receipts, reads receipt OCR text and returns line items, subtotal, tax and total as JSON. It is scored on the same 100 test receipts, the same OCR text and the same rules as the API models in my [receipt-cascade](https://github.com/masooddahmedd/receipt-cascade) project, so the numbers are directly comparable. It is served with vLLM on a single consumer GPU.

## Results

CORD v2 test split: 100 receipts, 300 scored fields. The gpt-4o-mini and gpt-4o rows come from the receipt-cascade run on the same receipts, fields and scorer.

| Model | Input | Accuracy | Line items (99) | Subtotal (65) | Tax (41) | Total (95) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-1.5B-Instruct, no fine-tune | OCR text | 40.7% | 15.2% | 70.8% | 43.9% | 45.3% |
| gpt-4o-mini (3 runs, majority) | OCR text | 72.7% | 50.5% | 95.4% | 58.5% | 86.3% |
| **Qwen2.5-1.5B, fine-tuned** | OCR text | **80.7%** | **61.6%** | 92.3% | **90.2%** | **88.4%** |
| gpt-4o | receipt image | 89.7% | 86.9% | 93.8% | 75.6% | 95.8% |

Bootstrap 95% interval for the fine-tuned model, resampling receipts: 75.3% to 85.8%.

What this says:

- **Fine-tuning took the 1.5B model from 41% to 81%.** That is about 8 points above gpt-4o-mini reading the same OCR text, and it costs one cheap model call instead of three.
- **gpt-4o reading the image is still about 9 points better overall.** The gap is almost all line items (87% vs 62%). The OCR text loses which price belongs to which item; the image keeps that layout. Totals and tax come through OCR fine.
- **Tax is where training helped most** (44% to 90%, above gpt-4o's 76%). CORD receipts print tax in a few house styles that the model learned from 800 examples.
- **JSON was valid on 100/100 receipts** after fine-tuning (99/100 before).

**Serving.** vLLM, bf16, on one RTX 4070 Ti SUPER sharing the card with other services, 45% of its memory:

| | Fine-tuned | Base |
| --- | --- | --- |
| Latency per receipt, one request at a time (p50 / p90) | 0.72 s / 1.27 s | 0.85 s / 1.66 s |
| Throughput with 8 requests in flight | 512 receipts/min | 414 receipts/min |
| Output tokens per receipt | 102 | 130 |

At 512 receipts a minute, 1,000 receipts take about 2 GPU-minutes. For comparison, the receipt-cascade API costs per 1,000 receipts were $0.31 for gpt-4o-mini (3 runs) and $3.34 for gpt-4o.

**Training.**
- 6.8 minutes on the same card, 4.2 GB peak GPU memory.
- 18.5M trainable LoRA parameters (1.2% of the model).
- Loss fell from 0.18 to 0.035 over 3 epochs.
- Log: `results/train_log.json`.

## Limits

- 100 test receipts is small; see the interval above.
- CORD is Indonesian restaurant receipts. A model trained on 800 of them will not transfer to invoices or other countries' receipts without new data.
- The fine-tuned model reads OCR text, so it can't fix line items the OCR splits apart. Feeding it OCR boxes with coordinates, or a small vision model, is the next thing to try.
- No hyperparameter search: one run, settings fixed before looking at test results.


## Data

- **CORD v2.** Indonesian restaurant and shop receipts (NAVER Clova, CC BY 4.0), using the 800 train, 100 dev and 100 test splits.
- **Input.** RapidOCR text with boxes on the same printed row joined into one line. Test inputs are the exact OCR files the receipt-cascade run cached, so the fine-tuned model sees the same text gpt-4o-mini saw.
- **Target.** The labelled fields as printed on the receipt. Amounts keep their dots and commas ("45.000"), because models like to "fix" Indonesian amounts into 45.0.
- **Scoring,** copied from receipt-cascade:
  - Amounts are compared as integers.
  - A line item list is correct only if every item matches one-to-one, with the same quantity and price and a name similarity of at least 0.8.
  - The test split has 300 scored fields.
- **Length.** Prompt plus answer is 303 tokens at the median and 1,150 at most, so nothing is truncated.

## Training

QLoRA:
- base weights in 4-bit NF4
- LoRA rank 16 on all attention and MLP projections
- 3 epochs, learning rate 2e-4, cosine schedule
- batch size 1 with 8-step accumulation

Loss is computed on the JSON answer only; the system prompt and OCR text are masked. The adapter is merged into bf16 weights so vLLM serves one plain model.

## Run it

```bash
python -m slm.prepare --cascade-cache ../receipt-cascade/cache/ocr_boxes   # needs the CORD parquet files in data/cord
python -m slm.train --model Qwen/Qwen2.5-1.5B-Instruct --out out/qwen1.5b-cord
python -m slm.merge --model Qwen/Qwen2.5-1.5B-Instruct --adapter out/qwen1.5b-cord/adapter --out out/qwen1.5b-cord/merged
scripts/eval_served.sh out/qwen1.5b-cord/merged:qwen1.5b-ft Qwen/Qwen2.5-1.5B-Instruct:qwen1.5b-base
```

`data/train.jsonl`, `dev.jsonl` and `test.jsonl` are committed, so the training and eval steps don't need the images or the OCR.

Tested with torch 2.13 (CUDA 13), transformers 5.17, peft 0.20, bitsandbytes 0.50 and vLLM 0.28.
