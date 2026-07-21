# Final results analysis

This report describes the canonical queue-inclusive results generated on
2026-07-21. The simulator is calibrated to the Qwen2.5-14B/vLLM 0.25.1
hardware run and enforces that server's 32,768-token maximum sequence length.
All reported turn slowdowns include dispatch queueing and service time.

## What changed during final reproduction

- Parked prefix content is reclaimable and invisible to active-KV metrics; it
  is not modeled as reserved memory.
- A completed request caches prompt and generated tokens. Tool output becomes
  new prefill input at the next request and is not mislabeled as recomputation.
- Constructed 64-turn trajectories are capped at 32,768 prompt-plus-decode
  tokens, matching the validated server rather than creating unservable
  133k-token requests.
- Unfinished work is an SLO failure, program goodput is reported separately,
  and E8/E9 checkpoints can run in parallel without changing deterministic
  results.

## Hardware evidence

- V1: losing a 32k prefix adds 9.028 seconds; recompute penalty is approximately
  linear at 0.286 seconds per 1k tokens (`R²=0.989`).
- V2: in 114 all-parked samples, GPU, running, waiting, and active-KV signals
  are all zero.
- V3/V4: `k_half=4.606`; process cold start with warm host page cache is
  `38.155 ± 0.010 s`.
- P1: a parked 16k prefix is intact through eight concurrent 4k neighbors,
  64.5% retained at 16, and effectively gone at 24. Minimum parking targets
  from 0 to 32 seconds do not independently explain survival.
- P2: submitting all 24 neighbors before the probe gives 0% survival and
  3.735-second TTFT. Submitting eight, probing, then draining the remaining 16
  gives 100% survival and 0.124-second TTFT. Both modes complete every neighbor
  (`goodput=1`).

## Corrected baseline result (E3–E4)

Across the six agentic coding-trace points (`T>1`, `tau>0`):

| Policy | Mean SLO | GPU/static | SLO/GPU |
|---|---:|---:|---:|
| Pressure-aware | 0.986 | 2.357 | 0.418 |
| HPA | 0.936 | 2.105 | 0.445 |
| ParkAware | 0.902 | 1.220 | 0.739 |
| KV-util | 0.874 | 1.206 | 0.725 |
| RL Q-learning | 0.791 | 1.078 | 0.734 |
| RL + parked | 0.774 | 0.998 | 0.775 |
| Predictive | 0.678 | 0.806 | 0.841 |
| KEDA | 0.552 | 0.993 | 0.556 |

No dynamic policy dominates reliability and cost. More importantly, across
HPA, KEDA, KV-util, and ParkAware at `T={2,4,8,16}`, pressure eviction causes
96.53% of destroyed prefix tokens and scale-in causes only 3.47%.
Pressure-aware admission reduces simulated pressure-evicted tokens to zero.

## Full E8 load and trajectory result

E8 contains 72 rows: two trajectory lengths, four load multipliers, and nine
policies per point (Static, HPA, KEDA, KV-util, Predictive, two RL variants,
ParkAware, and Pressure-aware). RL is trained on a disjoint trace prefix at
every point.

| T | Load | Pressure SLO | HPA SLO | Pressure GPU/static | HPA GPU/static | Recompute saved vs HPA | Pressure goodput |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1× | 0.975 | 0.997 | 2.352 | 2.415 | 1.21M | 1.0 |
| 8 | 2× | 0.980 | 0.998 | 2.055 | 2.361 | 3.36M | 1.0 |
| 8 | 4× | 0.999 | 1.000 | 1.389 | 2.743 | 3.93M | 1.0 |
| 8 | 8× | 1.000 | 0.997 | 1.461 | 2.687 | 16.65M | 1.0 |
| 64 | 1× | 0.997 | 1.000 | 1.626 | 1.761 | 41.12M | 1.0 |
| 64 | 2× | 0.998 | 0.996 | 1.550 | 1.497 | 91.24M | 1.0 |
| 64 | 4× | 0.998 | 0.969 | 1.484 | 1.422 | 149.35M | 1.0 |
| 64 | 8× | 0.991 | 0.979 | 1.335 | 1.234 | 326.28M | 1.0 |

At short trajectories and high load, avoiding recomputation improves SLO and
cuts GPU time. At long trajectories, it still improves SLO and removes far
more recomputation, but preserving many prefixes requires 3.5–8.2% more GPU at
2–8× load. This is the central queue/recompute/GPU tradeoff.

## E9 admission sensitivity

E9 contains all 28 planned rows. The P1-derived admission batch of eight has
zero pressure evictions, `goodput=1`, and SLO between 0.991 and 1.000 across
the four stress points. It is robust but not universally optimal: at `T=64`,
8× load, batch four reaches 0.999 SLO versus batch eight's 0.991. At `T=8`,
8× load, batches 6, 8, and 10 all reach 1.000. The measured threshold is a
safe operating point, not a tuned constant for every workload.

## E10 queue–recompute–GPU accounting

Pressure-aware admission replaces hidden recomputation with explicit queueing.
At `T=8`, 1–2× load, it adds 3.37k–5.63k aggregate queue-seconds and loses
1.8–2.1 SLO points versus HPA. At 4–8×, the extra queue falls to 70.5–0
seconds, it saves 3.93M–16.65M recomputed tokens, and uses 45.6–49.4% less GPU.
At `T=64`, it adds 0.28k–4.55k queue-seconds and saves 41M–326M recomputed
tokens; SLO improves at 2–8×, while GPU rises at the three highest loads.

## Interpretation and limits

The evidence supports a bounded claim: concurrent memory pressure, not elapsed
parking time or autoscaler scale-in, is the dominant threat in this measured
configuration. Pressure-aware admission directly controls that threat and is
validated both by P2 hardware isolation and by queue-inclusive simulation. It
is not a universally cheapest controller. Results rely on one model, one GPU,
one vLLM version, constructed program identities/tool delays, bounded context,
and synthetic load amplification. E2–E10 remain simulations rather than a
multi-replica production replay.

## Generated artifacts

- `results/e1_trace_stats.json` and `results/e2_*.csv`: workload and metric
  characterization.
- `results/e3_*.csv` through `results/e7_*.csv`: corrected queue-inclusive
  baseline, mechanism, cold-start, workload, and sensitivity studies.
- `results/e8_high_load.csv`: 72 complete load/trajectory/policy rows.
- `results/e9_admission_sensitivity.csv`: 28 threshold rows.
- `results/e10_queue_tradeoff.csv`: 64 HPA-relative accounting rows.
- `vllm_measured/raw/p2_*.csv`: protected/uncontrolled hardware trials.
- `figs/f1_*.{pdf,png}` through `figs/f11_*.{pdf,png}` and
  `paper/parked.pdf`: regenerated publication outputs.
