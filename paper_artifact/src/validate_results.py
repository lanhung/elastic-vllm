"""Validate that the canonical E1--E10 result bundle is complete and coherent."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def load(name: str, rows: int) -> pd.DataFrame:
    path = RES / name
    if not path.exists():
        raise AssertionError(f"missing {path}")
    frame = pd.read_csv(path)
    if len(frame) != rows:
        raise AssertionError(f"{name}: expected {rows} rows, got {len(frame)}")
    return frame


def main() -> None:
    trace = json.loads((RES / "e1_trace_stats.json").read_text())
    if set(trace) != {"code", "conv"}:
        raise AssertionError("E1 must contain code and conversation traces")
    load("e2_metric_divergence.csv", 4)
    e3 = load("e3_policy_sweep.csv", 90)
    e4 = load("e4_scale_in_damage.csv", 25)
    load("e5_coldstart.csv", 30)
    load("e6_workload_contrast.csv", 20)
    load("e7_sensitivity.csv", 100)
    e8 = load("e8_high_load.csv", 72)
    e9 = load("e9_admission_sensitivity.csv", 28)
    e10 = load("e10_queue_tradeoff.csv", 64)

    if not (e8.groupby(["turns_per_program", "load_multiplier"]).size() == 9).all():
        raise AssertionError("E8 must have nine policies at every workload point")
    if not (e9.groupby(["turns_per_program", "load_multiplier"]).size() == 7).all():
        raise AssertionError("E9 must have seven admission batches per point")
    if any(RES.glob("*.partial.csv")):
        raise AssertionError("partial result checkpoints remain")

    pressure = e8[e8.policy == "pressure-aware"]
    if len(pressure) != 8 or not (pressure.goodput == 1.0).all():
        raise AssertionError("pressure-aware E8 must complete every program")
    if not (e9.kills_evict == 0).all():
        raise AssertionError("E9 pressure-aware points must have zero eviction")
    tradeoff = e10[e10.policy == "pressure-aware"]
    if not (tradeoff.recompute_saved_vs_hpa > 0).all():
        raise AssertionError("pressure admission must save recomputation vs HPA")

    baseline = e4[(e4["T"] > 1) & (e4.policy != "pressure-aware")]
    evict_share = (baseline.evicted_tokens.sum() /
                   (baseline.evicted_tokens.sum() +
                    baseline.scalein_tokens.sum()))
    agentic = e3[(e3.turns_per_program > 1) & (e3.think_mean_s > 0)]
    mean_slo = agentic.groupby("policy").slo_attain.mean()
    print("canonical E1-E10 bundle: PASS")
    print("rows: E3=90 E4=25 E5=30 E6=20 E7=100 E8=72 E9=28 E10=64")
    print(f"baseline pressure-eviction share={evict_share:.4%}")
    print(f"mean agentic SLO: HPA={mean_slo['hpa-gpu']:.4f} "
          f"ParkAware={mean_slo['park-aware']:.4f} "
          f"PressureAware={mean_slo['pressure-aware']:.4f}")
    print("pressure-aware E8 goodput=1 at all 8 points; E9 evictions=0")


if __name__ == "__main__":
    main()
