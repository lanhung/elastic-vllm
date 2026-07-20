"""
run_experiments.py -- the experiment driver.

Fairness rule
-------------
Comparing autoscalers on cost alone is meaningless (always scale to 1) and
on SLO alone is meaningless (always scale to max).  So for every workload
point we first find, by search, the cheapest STATIC cluster that meets the
SLO.  That static configuration is the reference.  An autoscaler is only
interesting if it holds the SLO while spending fewer GPU-seconds than the
static reference, because a static cluster is what you get if you never
autoscale at all.

Everything written to results/ is produced by running the simulator.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from workload import load_azure, build_programs, trace_stats
from sim import Cluster, HW, SLO, recommended_drain_s
from policies import HpaGpuUtil, KedaQueue, KvUtil, ParkAware, Static
from rl_policy import QLearnScaler, PredictScaler

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

HORIZON = 2400.0          # 40 min of the trace
TRAIN_HORIZON = 1200.0    # disjoint prefix used only to train the RL agent
RL_EPISODES = 25
SLO_TARGET = 0.95         # 95% of turns must meet the TTFT SLO
SEED = 20260720


WARMUP = 300.0

def simulate(df, T, tau, policy, hw=None, init_replicas=4, horizon=HORIZON):
    progs = build_programs(df, turns_per_program=T, think_mean_s=tau,
                           seed=SEED, horizon_s=horizon)
    c = Cluster(progs, hw or HW(), SLO(), policy, dt=0.5,
                init_replicas=init_replicas, max_replicas=64,
                warmup_s=WARMUP)
    r = c.run(horizon, drain_s=recommended_drain_s(progs, hw or HW()))
    r.turns_per_program = T
    r.think_mean_s = tau
    return r, c


def train_rl(df, T, tau, see_parked=False, hw=None, n0=5):
    """Train a Q-learner on a disjoint prefix.  Never scored on this data."""
    q = QLearnScaler(see_parked=see_parked, slo_weight=12.0, eps=0.25)
    for ep in range(RL_EPISODES):
        ps = build_programs(df, turns_per_program=T, think_mean_s=tau,
                            seed=SEED + 1000 + ep, horizon_s=TRAIN_HORIZON)
        q.reset_episode()
        Cluster(ps, hw or HW(), SLO(), q, dt=0.5,
                init_replicas=n0, max_replicas=64, warmup_s=100.0).run(TRAIN_HORIZON)
    q.training = False
    q.reset_episode()
    q.pending_violations = 0
    q.pending_turns = 0
    return q


def cheapest_static(df, T, tau, hw=None, lo=1, hi=48):
    """Smallest static N with SLO attainment >= SLO_TARGET."""
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        r, _ = simulate(df, T, tau, Static(mid), hw, init_replicas=mid)
        if r.slo_attain >= SLO_TARGET:
            best = (mid, r)
            hi = mid - 1
        else:
            lo = mid + 1
    if best is None:
        r, _ = simulate(df, T, tau, Static(48), hw, init_replicas=48)
        best = (48, r)
    return best


# ----------------------------------------------------------------------
def exp1_characterisation():
    """E1: what the real traces look like.  Pure measurement, no simulation."""
    out = {k: trace_stats(load_azure(k)) for k in ("code", "conv")}
    (RES / "e1_trace_stats.json").write_text(json.dumps(out, indent=2))
    print("E1 trace characterisation:")
    for k, v in out.items():
        print(f"  {k:5s} n={v['n_requests']:,} rate={v['rate_rps']:.2f}/s "
              f"ctx/gen={v['ctx_gen_ratio']:.1f}x IoD={v['index_of_dispersion']:.1f} "
              f"peak/mean={v['peak_to_mean']:.2f}")
    return out


def exp2_metric_divergence():
    """E2: do native metrics expose parked application state?

    We run a fixed cluster (no autoscaling, so the policy cannot confound
    the measurement) and record compute, active-KV, and reclaimable cache.
    """
    df = load_azure("code")
    rows = []
    for T, tau in [(1, 0.0), (4, 4.0), (8, 8.0), (16, 8.0)]:
        r, c = simulate(df, T, tau, Static(8), init_replicas=8)
        tl = pd.DataFrame(c.timeline)
        if len(tl) == 0:
            continue
        tl = tl[(tl.t > WARMUP) & (tl.t <= HORIZON)]
        cache_util = tl.cache_resident / tl.kv_capacity
        parked_cache_util = tl.parked_cached / tl.kv_capacity
        invisible = ((tl.parked > 0) & (tl.running == 0) &
                     (tl.gpu_util == 0) & (tl.kv_util == 0))
        rows.append(dict(T=T, tau=tau,
                         gpu_util_mean=float(tl.gpu_util.mean()),
                         kv_util_mean=float(tl.kv_util.mean()),
                         cache_resident_util_mean=float(cache_util.mean()),
                         parked_cache_util_mean=float(parked_cache_util.mean()),
                         invisible_parked_frac=float(invisible.mean()),
                         corr=float(tl.gpu_util.corr(tl.kv_util)),
                         parked_frac=float((tl.parked /
                                            (tl.parked + tl.running).clip(lower=1)).mean())))
        tl.to_csv(RES / f"e2_timeline_T{T}.csv", index=False)
    d = pd.DataFrame(rows)
    d.to_csv(RES / "e2_metric_divergence.csv", index=False)
    print("\nE2 metric divergence (fixed 8-replica cluster):")
    print(d.to_string(index=False))
    return d


def exp3_policy_sweep():
    """E3: the main result.  Sweep T and tau, compare policies."""
    df = load_azure("code")
    Ts = [1, 4, 16, 64]   # real SWE agents average ~51 turns
    taus = [0.0, 8.0, 30.0]
    rows = []
    for T in Ts:
        for tau in taus:
            if T == 1 and tau > 0:
                continue                          # single-turn has no think time
            t0 = time.time()
            n_star, r_star = cheapest_static(df, T, tau)
            ref_gpu = r_star.gpu_seconds
            policies = [
                Static(n_star),
                HpaGpuUtil(target=0.70),
                KedaQueue(target_per_replica=5.0),
                KvUtil(target=0.80),
                PredictScaler(horizon_s=HW().cold_start_s, max_batch=HW().max_batch),
                train_rl(df, T, tau, see_parked=False, n0=n_star),
                train_rl(df, T, tau, see_parked=True,  n0=n_star),
                ParkAware(target=0.70, max_batch=HW().max_batch),
            ]
            for pol in policies:
                r, _ = simulate(df, T, tau, pol, init_replicas=n_star)
                d = asdict(r)
                d.update(static_ref_n=n_star, goodput=r.goodput,
                         static_ref_gpu_s=ref_gpu,
                         gpu_vs_static=r.gpu_seconds / max(1e-9, ref_gpu),
                         meets_slo=bool(r.slo_attain >= SLO_TARGET))
                rows.append(d)
            print(f"  T={T:2d} tau={tau:4.1f} static*={n_star:2d} "
                  f"({time.time()-t0:.0f}s)")
    d = pd.DataFrame(rows)
    d.to_csv(RES / "e3_policy_sweep.csv", index=False)
    print(f"\nE3 wrote {len(d)} rows")
    return d


def exp4_scale_in_damage():
    """E4: attribute KV loss to eviction vs scale-in, and price the damage."""
    df = load_azure("code")
    rows = []
    for T in [1, 2, 4, 8, 16]:
        tau = 8.0 if T > 1 else 0.0
        n_star, _ = cheapest_static(df, T, tau)
        for pol_f in [lambda: HpaGpuUtil(), lambda: KedaQueue(),
                      lambda: KvUtil(), lambda: ParkAware(max_batch=HW().max_batch)]:
            r, _ = simulate(df, T, tau, pol_f(), init_replicas=n_star)
            rows.append(dict(T=T, tau=tau, policy=r.policy,
                             kills_evict=r.kills_evict,
                             kills_scalein=r.kills_scalein,
                             evicted_tokens=r.evicted_tokens,
                             scalein_tokens=r.scalein_tokens,
                             scale_ins=r.scale_ins,
                             recomputed_tokens=r.recomputed_tokens,
                             useful_prefill=r.useful_prefill_tokens,
                             waste_ratio=r.waste_ratio,
                             slo=r.slo_attain,
                             gpu_s=r.gpu_seconds))
    d = pd.DataFrame(rows)
    d.to_csv(RES / "e4_scale_in_damage.csv", index=False)
    print("\nE4 scale-in damage:")
    print(d.to_string(index=False))
    return d


def exp5_coldstart_sensitivity():
    """E5: how much of the problem is model-load cold start?"""
    df = load_azure("code")
    rows = []
    for cs in [0.0, 30.0, 38.155, 90.0, 180.0, 300.0]:
        hw = HW(cold_start_s=cs)
        for pol_f in [lambda: HpaGpuUtil(), lambda: KedaQueue(),
                      lambda: PredictScaler(horizon_s=cs, max_batch=hw.max_batch),
                      lambda: ParkAware(max_batch=hw.max_batch)]:
            r, _ = simulate(df, 8, 8.0, pol_f(), hw=hw)
            rows.append(dict(cold_start_s=cs, policy=r.policy,
                             slo=r.slo_attain, gpu_s=r.gpu_seconds,
                             p99_prog=r.p99_program_s, goodput=r.goodput,
                             waste=r.waste_ratio))
    d = pd.DataFrame(rows)
    d.to_csv(RES / "e5_coldstart.csv", index=False)
    print("\nE5 cold-start sensitivity:")
    print(d.to_string(index=False))
    return d


def exp6_workload_contrast():
    """E6: does the effect hold on the conversation trace too?"""
    rows = []
    for kind in ("code", "conv"):
        df = load_azure(kind)
        for T, tau in [(1, 0.0), (8, 8.0)]:
            for pol_f in [lambda: HpaGpuUtil(), lambda: KedaQueue(),
                          lambda: KvUtil(),
                          lambda: ParkAware(max_batch=HW().max_batch)]:
                r, _ = simulate(df, T, tau, pol_f())
                rows.append(dict(trace=kind, T=T, tau=tau, policy=r.policy,
                                 slo=r.slo_attain, goodput=r.goodput,
                                 gpu_s=r.gpu_seconds,
                                 waste=r.waste_ratio,
                                 kills_scalein=r.kills_scalein))
    d = pd.DataFrame(rows)
    d.to_csv(RES / "e6_workload_contrast.csv", index=False)
    print("\nE6 code vs conversation:")
    print(d.to_string(index=False))
    return d


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if which in ("all", "1"): exp1_characterisation()
    if which in ("all", "2"): exp2_metric_divergence()
    if which in ("all", "3"): exp3_policy_sweep()
    if which in ("all", "4"): exp4_scale_in_damage()
    if which in ("all", "5"): exp5_coldstart_sensitivity()
    if which in ("all", "6"): exp6_workload_contrast()
    print(f"\ntotal {time.time()-t0:.0f}s")
