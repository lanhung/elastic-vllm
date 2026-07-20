"""
exp7_sensitivity.py -- the sensitivity study behind Appendix A.

Three parameters a reviewer will question:
  A  KV capacity per replica, i.e. model size
  B  maximum batch
  C  ParkAware's target threshold theta

Run:  python3 exp7_sensitivity.py
Writes results/e7_sensitivity.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from workload import load_azure, build_programs
from sim import Cluster, HW, SLO, recommended_drain_s
from policies import HpaGpuUtil, KedaQueue, KvUtil, ParkAware

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SEED = 20260720
HORIZON = 2400.0
WARMUP = 300.0
T, TAU, N0 = 8, 8.0, 5


def run(df, hw: HW, policy) -> object:
    progs = build_programs(df, turns_per_program=T, think_mean_s=TAU,
                           seed=SEED, horizon_s=HORIZON)
    return Cluster(progs, hw, SLO(), policy, dt=0.5,
                   init_replicas=N0, max_replicas=64,
                   warmup_s=WARMUP).run(
                       HORIZON, drain_s=recommended_drain_s(progs, hw))


def main():
    df = load_azure("code")
    rows = []

    # --- A: KV per replica (a proxy for model size) --------------------
    for kv, lab in [(31_000, "31k"), (62_000, "62k"),
                    (75_216, "measured 14B"), (150_000, "150k"),
                    (500_000, "500k")]:
        hw = HW(kv_capacity=kv)
        for mk in (lambda: HpaGpuUtil(), lambda: KedaQueue(), lambda: KvUtil(),
                   lambda: ParkAware(max_batch=hw.max_batch)):
            r = run(df, hw, mk())
            rows.append(dict(sweep="kv_capacity", value=kv, label=lab,
                             policy=r.policy, slo=r.slo_attain,
                             goodput=r.goodput, gpu_s=r.gpu_seconds))
        print(f"  A kv={kv:,}")

    # --- B: maximum batch ---------------------------------------------
    for b in (8, 16, 32, 64):
        hw = HW(max_batch=b)
        for mk in (lambda: HpaGpuUtil(), lambda: KedaQueue(), lambda: KvUtil(),
                   lambda b=b: ParkAware(max_batch=b)):
            r = run(df, hw, mk())
            rows.append(dict(sweep="max_batch", value=b, label=str(b),
                             policy=r.policy, slo=r.slo_attain,
                             goodput=r.goodput, gpu_s=r.gpu_seconds))
        print(f"  B batch={b}")

    # --- C: ParkAware target ------------------------------------------
    for th in (0.5, 0.6, 0.7, 0.8, 0.9):
        r = run(df, HW(), ParkAware(target=th, max_batch=HW().max_batch))
        rows.append(dict(sweep="target", value=th, label=f"{th}",
                         policy="park-aware", slo=r.slo_attain,
                         goodput=r.goodput, gpu_s=r.gpu_seconds))
    print("  C target sweep")

    d = pd.DataFrame(rows)
    d.to_csv(RES / "e7_sensitivity.csv", index=False)
    print(f"\nwrote {RES/'e7_sensitivity.csv'} ({len(d)} rows)\n")
    for s in d.sweep.unique():
        print(f"=== {s} (SLO attainment) ===")
        print(d[d.sweep == s].pivot_table(index="label", columns="policy",
                                          values="slo").round(3).to_string())
        print()


if __name__ == "__main__":
    main()
