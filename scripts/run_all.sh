#!/usr/bin/env bash
# Whole pipeline on one GPU box: baseline, fine-tune, merge, serve, evaluate.
#   PY=/path/to/venv/bin/python scripts/run_all.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
export VLLM=${VLLM:-$(dirname "$PY")/vllm} GPU_UTIL=${GPU_UTIL:-0.45} PYTHONUNBUFFERED=1
BASE=Qwen/Qwen2.5-1.5B-Instruct
OUT=out/qwen1.5b-cord

serve_and_eval() {   # $1 model path/id, $2 result name
  scripts/serve.sh "$1" cord 8100 > "out/vllm_$2.log" 2>&1 &
  local pid=$!
  until curl -s http://127.0.0.1:8100/v1/models >/dev/null; do
    kill -0 $pid 2>/dev/null || { echo "vLLM failed, see out/vllm_$2.log"; return 1; }
    sleep 5
  done
  $PY -m slm.evaluate --name "$2" --concurrency 8
  $PY -m slm.evaluate --name "$2_single" --concurrency 1
  kill $pid; wait $pid 2>/dev/null
}

mkdir -p out
echo "== baseline $(date +%T)";  serve_and_eval "$BASE" qwen1.5b-base
echo "== train $(date +%T)";     $PY -m slm.train --model "$BASE" --out "$OUT"
echo "== merge $(date +%T)";     $PY -m slm.merge --model "$BASE" --adapter "$OUT/adapter" --out "$OUT/merged"
echo "== fine-tuned $(date +%T)"; serve_and_eval "$OUT/merged" qwen1.5b-ft
echo "== done $(date +%T)"
