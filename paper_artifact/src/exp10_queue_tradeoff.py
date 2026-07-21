"""E10: make the queue/recomputation/GPU tradeoff in E8 explicit."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def main():
    source = pd.read_csv(RES / "e8_high_load.csv")
    source = source[~source.policy.str.startswith("static")].copy()
    rows = []
    for (turns, load), group in source.groupby(
            ["turns_per_program", "load_multiplier"]):
        hpa = group[group.policy == "hpa-gpu"].iloc[0]
        for _, result in group.iterrows():
            row = result.to_dict()
            row.update(
                queue_delta_s_vs_hpa=(result.total_queue_s -
                                      hpa.total_queue_s),
                recompute_saved_vs_hpa=(hpa.recomputed_tokens -
                                        result.recomputed_tokens),
                gpu_saved_pct_vs_hpa=(1.0 - result.gpu_seconds /
                                      max(1e-9, hpa.gpu_seconds)),
                slo_delta_vs_hpa=result.slo_attain - hpa.slo_attain,
                completed_delta_vs_hpa=result.n_completed - hpa.n_completed,
            )
            rows.append(row)

    out = pd.DataFrame(rows)
    path = RES / "e10_queue_tradeoff.csv"
    out.to_csv(path, index=False)
    print(f"-> {path} ({len(out)} rows)")
    pressure = out[out.policy == "pressure-aware"]
    print(pressure[["turns_per_program", "load_multiplier", "slo_attain",
                    "p99_slowdown", "mean_queue_s", "p99_queue_s",
                    "queue_delta_s_vs_hpa", "recompute_saved_vs_hpa",
                    "gpu_saved_pct_vs_hpa", "goodput"]].to_string(index=False))


if __name__ == "__main__":
    main()
