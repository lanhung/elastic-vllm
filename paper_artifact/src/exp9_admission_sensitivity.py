"""E9: sensitivity of the P1-derived per-replica admission guard.

The default of eight is the largest neighbour count directly observed to
retain a 16k prefix in every P1 hardware trial. This sweep changes that guard
without changing the exact capacity-protection check, at high synthetic load
and both short and long constructed trajectories.
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
from policies import PressureAwareAdmission
from run_experiments import cheapest_static, simulate
from sim import HW
from workload import load_azure

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TAU = 8.0
BATCHES = (4, 6, 8, 10, 12, 14, 16)
TARGET_TOKENS = 16_000
NEIGHBOUR_TOKENS = 4_000


def run_point(turns: int, load: int) -> list[dict]:
    df = load_azure("code")
    capacity_bound = ((HW().kv_capacity - TARGET_TOKENS) //
                      NEIGHBOUR_TOKENS)
    n_star, static = cheapest_static(
        df, turns, TAU, hi=64, load_multiplier=load)
    rows = []
    for batch in BATCHES:
        result, _ = simulate(
            df, turns, TAU,
            PressureAwareAdmission(admission_batch=batch),
            init_replicas=n_star, load_multiplier=load)
        row = asdict(result)
        row.update(
            admission_batch=batch,
            load_multiplier=load,
            static_ref_n=n_star,
            static_ref_gpu_s=static.gpu_seconds,
            gpu_vs_static=(result.gpu_seconds /
                           max(1e-9, static.gpu_seconds)),
            p1_safe_observed=8,
            capacity_only_bound=capacity_bound,
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
    partial = RES / "e9_admission_sensitivity.partial.csv"
    rows = (pd.read_csv(partial).to_dict("records")
            if partial.exists() else [])
    complete = set()
    if rows:
        checkpoint = pd.DataFrame(rows)
        counts = checkpoint.groupby(
            ["turns_per_program", "load_multiplier"]).size()
        complete = {tuple(map(int, key)) for key, n in counts.items()
                    if n == len(BATCHES)}
        print(f"resuming {len(complete)} completed workload points")
    capacity_bound = ((HW().kv_capacity - TARGET_TOKENS) //
                      NEIGHBOUR_TOKENS)
    points = []
    for turns in (8, 64):
        for load in (4, 8):
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
        ["turns_per_program", "load_multiplier", "admission_batch"])
    path = RES / "e9_admission_sensitivity.csv"
    out.to_csv(path, index=False)
    partial.unlink(missing_ok=True)
    print(f"-> {path} ({len(out)} rows)")
    print(out[["turns_per_program", "load_multiplier", "admission_batch",
               "slo_attain", "p99_slowdown", "p99_queue_s",
               "gpu_vs_static", "recomputed_tokens", "kills_evict",
               "goodput"]].to_string(index=False))
    print(f"capacity-only neighbour bound={capacity_bound}; "
          "P1 all-hit observed bound=8")


if __name__ == "__main__":
    main()
