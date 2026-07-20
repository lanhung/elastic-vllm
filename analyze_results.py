#!/usr/bin/env python3
"""Produce one auditable summary from the canonical hardware run."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

from vllm_validate import fit_batch_model


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def mean(rows, key):
    return statistics.mean(float(r[key]) for r in rows)


def analyse_v1(path: Path) -> dict:
    rows = read_csv(path)
    lengths = sorted({int(r["ctx_tokens"]) for r in rows})
    grouped = []
    for length in lengths:
        rs = [r for r in rows if int(r["ctx_tokens"]) == length]
        grouped.append(dict(
            ctx_tokens=length,
            mean_miss_ttft_s=mean(rs, "ttft_miss_s"),
            mean_hit_ttft_s=mean(rs, "ttft_hit_s"),
            mean_recompute_penalty_s=mean(rs, "recompute_penalty_s"),
            mean_speedup=mean(rs, "speedup")))

    x = np.asarray([r["ctx_tokens"] / 1000 for r in grouped], dtype=float)
    y = np.asarray([r["mean_recompute_penalty_s"] for r in grouped])
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    r2 = 1.0 - np.square(y - pred).sum() / np.square(y - y.mean()).sum()
    return dict(by_context=grouped, linear_fit=dict(
        slope_s_per_1k_tokens=float(slope), intercept_s=float(intercept),
        r2=float(r2)))


def analyse_v2(samples_path: Path, phases_path: Path) -> dict:
    samples = read_csv(samples_path)
    phases = read_csv(phases_path)
    session_count = len({int(r["sid"]) for r in phases})

    all_parked, any_compute = [], []
    for sample in samples:
        t = float(sample["t"])
        parked = sum(r["phase"] == "park" and
                     float(r["t0"]) <= t <= float(r["t1"])
                     for r in phases)
        computing = sum(r["phase"] == "compute" and
                        float(r["t0"]) <= t <= float(r["t1"])
                        for r in phases)
        if parked == session_count:
            all_parked.append(sample)
        if computing > 0:
            any_compute.append(sample)

    def signals(rows):
        return dict(
            samples=len(rows), mean_gpu_pct=mean(rows, "gpu"),
            mean_running=mean(rows, "running"),
            mean_waiting=mean(rows, "waiting"),
            mean_kv_pct=100 * mean(rows, "kv"))

    return dict(session_count=session_count,
                all_sessions_parked=signals(all_parked),
                any_session_computing=signals(any_compute))


def analyse_v3(path: Path) -> dict:
    rows = read_csv(path)
    for r in rows:
        r["k"] = int(r["k"])
        r["per_seq_tok_s"] = float(r["agg_tok_s"]) / r["k"]
    fit = fit_batch_model([r["k"] for r in rows],
                          [r["per_seq_tok_s"] for r in rows])
    grouped = []
    for k in sorted({r["k"] for r in rows}):
        rs = [r for r in rows if r["k"] == k]
        grouped.append(dict(k=k, mean_wall_s=mean(rs, "wall_s"),
                            mean_per_seq_tok_s=mean(rs, "per_seq_tok_s"),
                            mean_aggregate_tok_s=mean(rs, "agg_tok_s")))
    return dict(by_concurrency=grouped, fit=fit,
                model="per_seq_tok_s = R / (k + k_half)")


def analyse_v4(path: Path) -> dict:
    values = [float(r["cold_start_s"]) for r in read_csv(path)]
    return dict(values_s=values, mean_s=statistics.mean(values),
                population_stdev_s=statistics.pstdev(values),
                scope="process cold start with warm host page cache")


def derive_sim_calibration(v1: dict, v3: dict,
                           v3_ctx_tokens: int = 2000,
                           v3_gen_tokens: int = 64) -> dict:
    """Derive the simulator's work units from the measured service curves.

    With one active sequence, a prefill token takes
    ``(1 + k_half) / service_rate`` seconds.  V1 measures that slope.  V3's
    fitted numerator then fixes the relative decode-token weight for its
    nominal 2k-prefix/64-token workload.
    """
    slope_s_per_token = v1["linear_fit"]["slope_s_per_1k_tokens"] / 1000.0
    k_half = v3["fit"]["k_half"]
    fitted_rate = v3["fit"]["R"]
    service_rate = (1.0 + k_half) / slope_s_per_token
    decode_weight = (
        v3_gen_tokens * service_rate / fitted_rate - v3_ctx_tokens
    ) / v3_gen_tokens
    return dict(
        service_rate_token_equiv_s=service_rate,
        prefill_weight=1.0,
        decode_weight=decode_weight,
        v3_nominal_context_tokens=v3_ctx_tokens,
        v3_generated_tokens=v3_gen_tokens,
        derivation=(
            "service_rate=(1+k_half)/v1_slope_per_token; "
            "decode_weight=(gen*service_rate/R-context)/gen"
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path,
                    default=Path("results/runs/20260720_1704"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/canonical_summary.json"))
    ap.add_argument("--artifact-out", type=Path,
                    default=Path("paper_artifact/vllm_measured/summary.json"))
    args = ap.parse_args()

    v1 = analyse_v1(args.run / "results_vllm/v1_prefix_cache.csv")
    v3 = analyse_v3(args.run / "results_vllm/v3_batch_curve.csv")
    summary = dict(
        canonical_run=args.run.name,
        v1=v1,
        v2=analyse_v2(args.run / "results_vllm/v2_parking_samples.csv",
                      args.run / "results_vllm/v2_parking_phases.csv"),
        v3=v3,
        v4=analyse_v4(args.run /
                      "results_vllm_run_20260720_1709/v4_cold_start.csv"),
        sim_calibration=derive_sim_calibration(v1, v3))
    for path in (args.out, args.artifact_out):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"-> {args.out}\n-> {args.artifact_out}")


if __name__ == "__main__":
    main()
