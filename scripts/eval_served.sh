#!/usr/bin/env bash
# Serve each model with vLLM and score it on the test split: batched (8 in flight) and one at a time.
#   PY=/path/to/python scripts/eval_served.sh Qwen/Qwen2.5-1.5B-Instruct:qwen1.5b-base out/qwen1.5b-cord/merged:qwen1.5b-ft
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
export VLLM=${VLLM:-$(dirname "$PY")/vllm} PYTHONUNBUFFERED=1
mkdir -p out
for pair in "$@"; do
  model=${pair%%:*}; name=${pair##*:}
  scripts/serve.sh "$model" cord 8100 > "out/vllm_$name.log" 2>&1 &
  pid=$!
  until curl -s http://127.0.0.1:8100/v1/models >/dev/null; do
    kill -0 $pid 2>/dev/null || { echo "vLLM failed for $name, see out/vllm_$name.log"; continue 2; }
    sleep 5
  done
  $PY -m slm.evaluate --name "$name" --concurrency 8
  $PY -m slm.evaluate --name "${name}_single" --concurrency 1
  kill $pid; wait $pid 2>/dev/null
done
echo EVALDONE
