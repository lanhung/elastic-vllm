# Your Neighbours, Not Your Autoscaler — artifact

This artifact combines a real vLLM hardware validation with a calibrated,
trace-driven study of autoscaling under agentic LLM workloads. The hardware
experiment falsified the project's original assumption: after a request
returns, native vLLM does **not** reserve KV blocks for the parked program.
Prefix-cache content may survive opportunistically, but it is reclaimable and
invisible to GPU utilization, queue depth, and exported active-KV utilization.

The simulator and paper in this directory use that corrected mechanism. The
original, pre-measurement paper and outputs are retained under
`legacy_precalibration/` for auditability; they are not current results.

## Reproduce the CPU experiments

```bash
python3 -m pip install numpy pandas matplotlib
make data       # no-op when the vendored Azure traces are already present
make sim        # deterministic E1–E10, about 30 minutes on one CPU core
make sim JOBS=8 # parallelize independent E8/E9 workload points
make paper      # figures plus paper/parked.pdf (requires pdflatex)
```

Seed `20260720` is used throughout. Simulation arrivals, context tokens, and
generated tokens come from the public Azure LLM inference traces. Program
composition (`T`) and tool delay (`tau`) are constructed and swept because the
public traces do not contain those fields.

## Hardware evidence

`vllm_measured/raw/` contains the measurements from vLLM 0.25.1 serving
Qwen2.5-14B-Instruct on one reported RTX 4090 with 49,140 MiB:

- V1: a 32k-token cache miss adds 9.03 s relative to a hit; the recompute
  penalty is approximately linear (`R²=0.989`, 0.286 s per 1k tokens).
- V2: across 114 samples in which all eight sessions were inside tool calls,
  GPU utilization, running requests, waiting requests, and active-KV
  utilization all read zero.
- V3: the measured batch curve fits `per_seq_rate = R/(k+k_half)` with
  `k_half=4.606` (`R²=0.99995`), not the original assumed value 16.
- V4: process cold start with a warm host page cache is 38.155 ± 0.010 s.
- P1: a parked 16k prefix survives minimum park targets of 0–32 s when pressure is at most
  eight concurrent 4k contexts, is about 64.5% retained at 16 neighbors, and
  is fully evicted at 24. Pressure, not elapsed time in this range, dominates.
- P2: admitting eight neighbors before the target probe and deferring the
  remaining sixteen preserves 100% of the prefix at 0.124 s TTFT; admitting
  all 24 first yields 0% survival and 3.735 s TTFT. Both complete all work.

`vllm_validation/` contains the scripts that produced these data. They are not
required for the CPU simulation.

`vllm_measured/summary.json` also records the explicit V1/V3 calibration
derivation. It yields 19,606.97 token-equivalents/s and decode weight 94.282
relative to prefill; these are the defaults used by `src/sim.py`.

## Corrected result and pressure-aware follow-up

The corrected simulation does **not** validate the original ParkAware claim.
Across the six agentic coding-trace points (`T>1`, `tau>0`), mean SLO
attainment is 0.936 for HPA, 0.874 for KV-util, 0.902 for ParkAware, and 0.986
for pressure-aware admission. The pressure policy uses 2.357× the cheapest
static GPU budget on average, so it is not a universal cost winner.

Across four baseline controllers and `T={2,4,8,16}`, pressure eviction causes
96.53% of destroyed prefix tokens; scale-in causes 3.47%. E8 contains all 72
planned rows, including both RL variants and Predictive at `T={8,64}`. At
`T=8`, 8× load, pressure admission attains 1.000 SLO versus HPA's 0.997 and
uses 45.6% less GPU. At `T=64`, 8×, it attains 0.991 versus 0.979 and saves
326M recomputed tokens, but uses 8.2% more GPU. E9 and E10 report admission
threshold sensitivity and the explicit queue–recompute–GPU exchange. See
`RESULTS_ANALYSIS.md` for the complete interpretation.

## Layout

- `src/`: workload builder, calibrated simulator, policies, E1–E10 drivers,
  and figure generator.
- `tests/`: regression tests for invisible parked cache, partial eviction,
  pressure-safe placement, load amplification, per-replica batching, and
  end-of-trace draining.
- `results/`: regenerated E1–E10 outputs.
- `vllm_measured/`: canonical hardware measurements and summary.
- `paper/`: corrected paper source and compiled PDF.
- `legacy_precalibration/`: immutable record of the uploaded original claim.
