"""E8: stress the policies with phase-shifted trace superposition.

The Azure coding trace is replayed at 1x, 2x, 4x, and 8x load by superposing
deterministically phase-shifted copies. This is a synthetic stress test, not a
claim that Azure observed these rates. It addresses the low-load regime of the
base experiment and directly evaluates pressure-aware admission.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from policies import (HpaGpuUtil, KedaQueue, KvUtil, ParkAware,
                      PressureAwareAdmission, Static)
from run_experiments import cheapest_static, simulate
from workload import load_azure

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
T, TAU = 8, 8.0


def main():
    df = load_azure("code")
    rows = []
    for load in (1, 2, 4, 8):
        n_star, static_result = cheapest_static(
            df, T, TAU, hi=64, load_multiplier=load)
        policies = [
            Static(n_star), HpaGpuUtil(), KedaQueue(), KvUtil(),
            ParkAware(), PressureAwareAdmission(),
        ]
        for policy in policies:
            result, _ = simulate(
                df, T, TAU, policy, init_replicas=n_star,
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
        print(f"  load={load}x static*={n_star}")

    out = pd.DataFrame(rows)
    path = RES / "e8_high_load.csv"
    out.to_csv(path, index=False)
    print(f"-> {path} ({len(out)} rows)")
    print(out[["load_multiplier", "policy", "static_ref_n", "slo_attain",
               "gpu_vs_static", "slo_per_gpu_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
