"""E8: stress the policies with phase-shifted trace superposition.

The Azure coding trace is replayed at 1x, 2x, 4x, and 8x load for both T=8 and
T=64 by superposing deterministically phase-shifted copies. This is a synthetic
stress test, not a claim that Azure observed these rates. It addresses the
low-load regime of the base experiment, evaluates pressure-aware admission at
long trajectory length, and includes every E3 dynamic baseline.
"""
from __future__ import annotations

from dataclasses import asdict
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from policies import (HpaGpuUtil, KedaQueue, KvUtil, ParkAware,
                      PressureAwareAdmission, Static)
from rl_policy import PredictScaler
from run_experiments import cheapest_static, simulate, train_rl
from sim import HW
from workload import load_azure

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
T_VALUES, TAU = (8, 64), 8.0


def run_point(turns: int, load: int) -> list[dict]:
    """Run one independent workload point (safe to execute in a worker)."""
    df = load_azure("code")
    n_star, static_result = cheapest_static(
        df, turns, TAU, hi=64, load_multiplier=load)
    policies = [
        Static(n_star), HpaGpuUtil(), KedaQueue(), KvUtil(),
        PredictScaler(horizon_s=HW().cold_start_s,
                      max_batch=HW().max_batch),
        train_rl(df, turns, TAU, see_parked=False, n0=n_star,
                 load_multiplier=load),
        train_rl(df, turns, TAU, see_parked=True, n0=n_star,
                 load_multiplier=load),
        ParkAware(), PressureAwareAdmission(),
    ]
    rows = []
    for policy in policies:
        result, _ = simulate(
            df, turns, TAU, policy, init_replicas=n_star,
            load_multiplier=load)
        row = asdict(result)
        row.update(
            load_multiplier=load,
            static_ref_n=n_star,
            static_ref_gpu_s=static_result.gpu_seconds,
            gpu_vs_static=(result.gpu_seconds /
                           max(1e-9, static_result.gpu_seconds)),
            slo_per_gpu_ratio=(result.slo_attain /
                               max(1e-9, result.gpu_seconds /
                                   static_result.gpu_seconds)),
        )
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1,
                    help="independent workload points to run concurrently")
    args = ap.parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    partial = RES / "e8_high_load.partial.csv"
    rows = (pd.read_csv(partial).to_dict("records")
            if partial.exists() else [])
    complete = set()
    if rows:
        checkpoint = pd.DataFrame(rows)
        counts = checkpoint.groupby(
            ["turns_per_program", "load_multiplier"]).size()
        complete = {tuple(map(int, key)) for key, n in counts.items()
                    if n == 9}
        print(f"resuming {len(complete)} completed workload points")
    points = []
    for turns in T_VALUES:
        for load in (1, 2, 4, 8):
            if (turns, load) in complete:
                print(f"  T={turns} load={load}x checkpointed")
            else:
                points.append((turns, load))
    workers = min(args.jobs, len(points), os.cpu_count() or 1)
    if workers <= 1:
        completed = ((point, run_point(*point)) for point in points)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        futures = {pool.submit(run_point, *point): point for point in points}
        completed = ((futures[f], f.result()) for f in as_completed(futures))
    for (turns, load), point_rows in completed:
        rows.extend(point_rows)
        pd.DataFrame(rows).to_csv(partial, index=False)
        print(f"  T={turns} load={load}x static*={point_rows[0]['static_ref_n']}")
    if workers > 1:
        pool.shutdown()

    out = pd.DataFrame(rows).sort_values(
        ["turns_per_program", "load_multiplier", "policy"])
    path = RES / "e8_high_load.csv"
    out.to_csv(path, index=False)
    partial.unlink(missing_ok=True)
    print(f"-> {path} ({len(out)} rows)")
    print(out[["turns_per_program", "load_multiplier", "policy",
               "static_ref_n", "slo_attain", "gpu_vs_static",
               "p99_slowdown", "p99_queue_s", "recomputed_tokens",
               "goodput"]].to_string(index=False))


if __name__ == "__main__":
    main()
