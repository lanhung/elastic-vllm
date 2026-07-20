#!/usr/bin/env bash
# One-time setup on AutoDL.  Run preflight.sh FIRST.
set -e
D=/root/autodl-tmp                     # 30 GB system disk cannot hold a 14B model
mkdir -p $D/models $D/hf
export HF_ENDPOINT=https://hf-mirror.com          # HF is blocked on AutoDL
export HF_HOME=$D/hf
export HF_HUB_DISABLE_XET=1

# vLLM pins its own torch.  Fighting the system torch 2.12 / CUDA 13 is a
# waste of an afternoon, so give it a venv and let it choose.
if [ ! -d $D/venv ]; then
  python3 -m venv $D/venv
fi
source $D/venv/bin/activate
pip install -q -U pip
pip install -q vllm requests pandas numpy matplotlib huggingface_hub

python3 - <<'PY'
import torch
print("venv torch", torch.__version__, "cuda", torch.version.cuda,
      "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device", p.name, round(p.total_memory/2**30,1), "GB")
PY

MODEL=${VLLM_MODEL:-Qwen/Qwen2.5-14B-Instruct}
echo ">>> downloading $MODEL to $D/models"
hf download "$MODEL" --local-dir $D/models/$(basename $MODEL)

cat > $D/env.sh <<ENV
source $D/venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$D/hf
export HF_HUB_DISABLE_XET=1
export VLLM_MODEL=$D/models/$(basename $MODEL)
ENV
echo
echo "ready.  in every new shell:  source $D/env.sh"
echo "then:   bash serve.sh   (terminal 1)"
