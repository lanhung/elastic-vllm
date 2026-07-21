# elastic-vllm hardware validation

Single-GPU measurements for the modelling assumptions used by the elastic
vLLM/ParkAware study. The tested server reports an RTX 4090 with 49,140 MiB,
vLLM 0.25.1, PyTorch 2.11.0+cu130, and a local Qwen2.5-14B-Instruct model.

## Reproduce

```bash
bash preflight.sh
bash setup_autodl.sh
source /root/autodl-tmp/env.sh
bash serve.sh
python3 vllm_validate.py --all --out results_vllm
python3 vllm_validate.py --p1 --out results_p1
python3 vllm_validate.py --p2 --out results_p2
```

Set `VLLM_PORT` and the matching `VLLM_BASE` when port 8000 is occupied, for
example `VLLM_PORT=8002 bash serve.sh` and
`VLLM_BASE=http://127.0.0.1:8002 python3 vllm_validate.py --p1 ...`.

`--all` retains the original V1-V4 suite. P1 and P2 are explicit because they
exercise cache survival and pressure-aware admission separately.

## Results

See [`results/README.md`](results/README.md) for run provenance and the
canonical dataset. `python3 analyze_results.py` regenerates the canonical
summary and the fitted `k_half` value from raw CSVs.

The E1–E10 study, corrected simulator, P1/P2 integration, regenerated
figures, and revised paper are under [`paper_artifact/`](paper_artifact/).
The hardware measurements falsified the original assumption that parked
programs reserve KV in native vLLM. The current artifact reports both the
corrected negative ParkAware result and the pressure-aware follow-up: baseline
prefix loss is dominated by eviction rather than scale-in. Protected admission
preserves a hardware cache hit without dropping work; in simulation it trades
bounded queueing and workload-dependent GPU cost for avoided recomputation.
The uploaded pre-calibration version remains under
`paper_artifact/legacy_precalibration/` for provenance.
