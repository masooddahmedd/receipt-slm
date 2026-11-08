#!/usr/bin/env bash
# Serve a model with vLLM's OpenAI-compatible server, sized for a GPU shared with other services.
#   scripts/serve.sh out/qwen1.5b-cord/merged cord 8100
set -euo pipefail
MODEL=${1:?model path or HF id}
NAME=${2:-cord}
PORT=${3:-8100}
VLLM=${VLLM:-.venv/bin/vllm}
if [ -x "$VLLM" ]; then
  CMD=("$VLLM")
else
  # some venvs ship the vllm package without the console script
  CMD=("$(dirname "$VLLM")/python" -m vllm.entrypoints.cli.main)
fi
# FlashInfer's sampler compiles a CUDA kernel on first use (needs ninja and a matching nvcc);
# PyTorch's sampler gives the same greedy outputs without the build step.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
exec "${CMD[@]}" serve "$MODEL" --served-model-name "$NAME" --host 127.0.0.1 --port "$PORT" \
  --max-model-len 2048 --gpu-memory-utilization "${GPU_UTIL:-0.45}" --max-num-seqs 16 \
  --dtype bfloat16 --enforce-eager
