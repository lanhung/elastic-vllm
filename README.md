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
```

Set `VLLM_PORT` and the matching `VLLM_BASE` when port 8000 is occupied, for
example `VLLM_PORT=8002 bash serve.sh` and
`VLLM_BASE=http://127.0.0.1:8002 python3 vllm_validate.py --p1 ...`.

`--all` retains the original V1-V4 suite. P1 is explicit because its default
survival grid takes substantially longer.

## Results

See [`results/README.md`](results/README.md) for run provenance and the
canonical dataset. `python3 analyze_results.py` regenerates the canonical
summary and the fitted `k_half` value from raw CSVs.

The uploaded E1–E7 study, corrected simulator, P1 integration, regenerated
figures, and revised paper are under [`paper_artifact/`](paper_artifact/).
The hardware measurements falsified the original assumption that parked
programs reserve KV in native vLLM; the current artifact and paper report the
corrected negative policy result. The uploaded pre-calibration version remains
under `paper_artifact/legacy_precalibration/` for provenance.
