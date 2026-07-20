#!/usr/bin/env bash
# preflight.sh -- run this FIRST, before installing anything.
# Two minutes.  Answers the only three questions that can waste your money.
set +e

echo "==============================================================="
echo " PREFLIGHT"
echo "==============================================================="

# ---------------------------------------------------------------- 1
echo
echo "[1] Is this a real GPU or a virtualised slice, and do we get"
echo "    utilisation readings at all?"
echo
nvidia-smi --query-gpu=name,memory.total,utilization.gpu,driver_version \
           --format=csv 2>&1 | sed 's/^/    /'
echo
if nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
     | grep -qE '^[0-9]+$'; then
  echo "    OK   utilization.gpu is readable"
  UTIL_OK=1
else
  echo "    WARN utilization.gpu is NOT readable on this vGPU."
  echo "         V2 must fall back to vLLM's own metrics (it can; see below)."
  UTIL_OK=0
fi

# is the physical GPU shared?  if other processes we don't own appear,
# any utilisation reading includes their work and is contaminated.
echo
echo "    processes currently on this GPU:"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>&1 | sed 's/^/      /'
echo "    (if you see PIDs that are not yours, utilisation is shared and"
echo "     nvidia-smi readings will be contaminated by another tenant)"

# ---------------------------------------------------------------- 2
echo
echo "[2] Does the CUDA / torch / vLLM stack actually line up?"
echo
python3 - <<'PY' 2>&1 | sed 's/^/    /'
import torch, sys
print("python      ", sys.version.split()[0])
print("torch       ", torch.__version__)
print("torch cuda  ", torch.version.cuda)
print("cuda avail  ", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device      ", p.name)
    print("vram GB     ", round(p.total_memory/2**30, 1))
    print("capability  ", f"{p.major}.{p.minor}")
PY
echo
nvcc --version 2>/dev/null | tail -2 | sed 's/^/    /' || echo "    (no nvcc, fine)"
echo
echo "    NOTE: vLLM wheels are built against specific torch+CUDA pairs."
echo "    torch 2.12 / CUDA 13 is very new.  If 'pip install vllm' pulls a"
echo "    wheel that then fails to import, the fix is to pin an older torch"
echo "    in a venv rather than to fight the system one:"
echo "      python3 -m venv ~/autodl-tmp/venv && source ~/autodl-tmp/venv/bin/activate"
echo "      pip install vllm            # let it choose its own torch"

# ---------------------------------------------------------------- 3
echo
echo "[3] Disk.  Weights are the big item; put them on the data disk."
echo
df -h / /root/autodl-tmp 2>/dev/null | sed 's/^/    /'
echo
echo "    A 14B model in bf16 is ~28 GB.  The 30 GB system disk will NOT"
echo "    hold it.  Everything must go to /root/autodl-tmp."

# ---------------------------------------------------------------- 4
echo
echo "==============================================================="
echo " VERDICT"
echo "==============================================================="
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$VRAM" ] && [ "$VRAM" -ge 40000 ]; then
  echo "  VRAM ${VRAM} MiB  -> run Qwen2.5-14B-Instruct (better than 7B:"
  echo "                       closer to what agentic serving actually uses)"
elif [ -n "$VRAM" ] && [ "$VRAM" -ge 20000 ]; then
  echo "  VRAM ${VRAM} MiB  -> run Qwen2.5-7B-Instruct"
else
  echo "  VRAM ${VRAM} MiB  -> too small, use a smaller model"
fi
if [ "$UTIL_OK" = "1" ]; then
  echo "  GPU util readable -> V2 can use nvidia-smi AND vLLM metrics"
else
  echo "  GPU util MISSING  -> V2 uses vLLM metrics only.  This is fine, and"
  echo "                       arguably better: vllm:num_requests_running is"
  echo "                       what KEDA actually reads, and it is not"
  echo "                       contaminated by other tenants on the card."
fi
echo
echo "  Next:  bash setup_autodl.sh"
echo "==============================================================="
