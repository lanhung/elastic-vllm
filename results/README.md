# Result provenance

The former duplicate top-level result directories are preserved as two raw
runs under `runs/`. No raw CSV or log was deleted.

## Canonical run

`runs/20260720_1704` is canonical because V1-V3 and V4 were rerun after the
compatibility fixes, and its V4 measurements use the local model path with
reliable process-group cleanup.

Canonical files:

- V1: `runs/20260720_1704/results_vllm/v1_prefix_cache.csv`
- V2 samples: `runs/20260720_1704/results_vllm/v2_parking_samples.csv`
- V2 phases: `runs/20260720_1704/results_vllm/v2_parking_phases.csv`
- V3: `runs/20260720_1704/results_vllm/v3_batch_curve.csv`
- V4: `runs/20260720_1704/results_vllm_run_20260720_1709/v4_cold_start.csv`
- Server log: `runs/20260720_1704/serve.log`

`canonical_summary.json` is generated from those files by
`python3 analyze_results.py`. Important canonical values are:

- V1 32k recompute penalty: 9.028 s; linear slope 0.2859 s/1k tokens,
  R²=0.9894.
- V2 all eight sessions parked: 114 samples, GPU=0%, running=0,
  waiting=0, reported KV usage=0%.
- V3 fit `per_seq_tok_s = R/(k+k_half)`: R=156.19,
  `k_half=4.606`, R²=0.99995.
- V4 process cold start with a warm host page cache: 38.155 ± 0.010 s.
- Joint V1/V3 service calibration: 19,606.97 token-equivalents/s and decode
  weight 94.282 relative to prefill (the derivation is machine-readable in
  `canonical_summary.json`).

The earlier `runs/20260720_1604` data remains available for provenance but
must not be mixed into canonical aggregate statistics.

## P1 cache-survival run

`runs/20260720_p1` contains all 45 trials measuring whether a parked 16k-token
prefix survives minimum total park targets of 0, 8, or 32 seconds under 0, 4,
8, 16, or 24 concurrent 4k-token neighbors. Neighbors run first and the script
then waits out any remaining target time; pressure itself therefore overruns
the 0/8-second targets at 16 and 24 neighbors. The canonical aggregate is
`results_p1_20260720/p1_cache_survival_summary.csv`:

- 0–8 neighbors: essentially complete retention at every delay;
- 16 neighbors: mean retained-prefix score 0.645 (partial eviction);
- 24 neighbors: mean retained-prefix score below 0.012 (full eviction).

The service used port 8002 so the unrelated wave-height service on port 8000
was not touched. The exact validation script and server log are stored with
the run.
