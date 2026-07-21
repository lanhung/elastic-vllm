#!/usr/bin/env bash
# Terminal 1.  Assumes: source /root/autodl-tmp/env.sh
set -e
MODEL=${VLLM_MODEL:-/root/autodl-tmp/models/Qwen2.5-14B-Instruct}
PORT=${VLLM_PORT:-8000}
# 48 GB vGPU: 14B bf16 is ~28 GB of weights, leaving ~14 GB for KV at 0.88.
# vLLM prints "GPU KV cache size: N tokens" at startup -- record that number,
# it is the paper's kv_capacity parameter, measured rather than assumed.
exec vllm serve "$MODEL" \
  --served-model-name agent-model \
  --port "$PORT" \
  --enable-prefix-caching \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.88
