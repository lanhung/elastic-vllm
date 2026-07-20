"""make_figs.py -- all figures.  Reads only results/*.csv; invents nothing."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figs"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "lines.linewidth": 1.5,
})
C = {"hpa-gpu": "#D55E00", "keda-queue": "#0072B2", "kv-util": "#009E73",
     "predictive": "#E69F00", "rl-qlearn": "#8C8C8C",
     "rl-qlearn+parked": "#56B4E9", "park-aware": "#CC79A7", "static": "#999999"}
LBL = {"hpa-gpu": "HPA (GPU util)", "keda-queue": "KEDA (queue)",
       "kv-util": "KV-util (llm-d)", "predictive": "Predictive (Holt)",
       "rl-qlearn": "RL (Q-learning)", "rl-qlearn+parked": "RL + parked signal",
       "park-aware": "ParkAware (candidate)"}
POLS = ["hpa-gpu","keda-queue","kv-util","predictive","rl-qlearn","park-aware"]


def save(fig, n):
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"{n}.{e}")
    plt.close(fig); print("  ", n)


# --- F1: the two traces are not the same kind of workload ----------------
def f1():
    st = json.loads((RES / "e1_trace_stats.json").read_text())
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.0))
    keys = [("ctx_gen_ratio", "context / generated"),
            ("index_of_dispersion", "index of dispersion"),
            ("peak_to_mean", "peak / mean arrivals")]
    for a, (k, lab) in zip(ax, keys):
        vals = [st["conv"][k], st["code"][k]]
        b = a.bar(["chat", "coding"], vals, color=["#0072B2", "#D55E00"], width=.55)
        a.set_ylabel(lab)
        a.set_ylim(0, max(vals) * 1.32)
        for r, v in zip(b, vals):
            a.text(r.get_x() + r.get_width() / 2, v * 1.04, f"{v:.1f}",
                   ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout(); save(fig, "f1_trace_contrast")


# --- F2: how much of the cluster is parked -------------------------------
def f2():
    d = pd.read_csv(RES / "e2_metric_divergence.csv")
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.plot(d["T"], d.parked_frac * 100, "o-", color=C["park-aware"],
            label="programs parked")
    ax.plot(d["T"], d.invisible_parked_frac * 100, "s--", color=C["kv-util"],
            label="samples: parked, all native signals zero")
    ax.set_xlabel("turns per agent program $T$")
    ax.set_ylabel("% of programs parked\nin a tool call")
    ax.set_xscale("log", base=2); ax.set_xticks(d["T"])
    ax.set_xticklabels([str(int(x)) for x in d["T"]])
    ax.set_ylim(-3, 100); ax.legend(fontsize=7)
    fig.tight_layout(); save(fig, "f2_parked_fraction")


# --- F3: main result -----------------------------------------------------
def f3():
    d = pd.read_csv(RES / "e3_policy_sweep.csv")
    a = d[~d.policy.str.startswith("static")].copy()
    taus = sorted(a.think_mean_s.unique())
    fig, AX = plt.subplots(2, len(taus), figsize=(7.2, 3.6), sharey="row")
    for ax, tau in zip(AX[0], taus):
        s = a[a.think_mean_s == tau]
        for pol in POLS:
            q = s[s.policy == pol].sort_values("turns_per_program")
            if len(q) == 0: continue
            ax.plot(q.turns_per_program, q.slo_attain, "o-",
                    color=C[pol], label=LBL[pol], markersize=3.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1,4,16,64]); ax.set_xticklabels(["1","4","16","64"])
        ax.set_title(rf"tool time $\tau$={tau:g}s"); ax.set_ylim(0, 1.05)
    for ax, tau in zip(AX[1], taus):
        s2 = a[a.think_mean_s == tau]
        for pol in POLS:
            q = s2[s2.policy == pol].sort_values("turns_per_program")
            if len(q) == 0: continue
            ax.plot(q.turns_per_program, q.gpu_vs_static, "o-", color=C[pol],
                    markersize=3.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1,4,16,64]); ax.set_xticklabels(["1","4","16","64"])
        ax.set_xlabel("turns $T$"); ax.axhline(1, color="#777777", lw=.8, ls=":")
    AX[0][0].set_ylabel("SLO attainment")
    AX[1][0].set_ylabel("GPU-seconds / cheapest\nstatic SLO configuration")
    handles, labels = AX[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7,
               bbox_to_anchor=(.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, .91)); save(fig, "f3_main_slo")


# --- F4: where the KV goes ----------------------------------------------
def f4():
    d = pd.read_csv(RES / "e4_scale_in_damage.csv")
    d = d[d["T"] > 1]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.2))
    pols = ["hpa-gpu", "keda-queue", "kv-util", "park-aware"]
    w, x = 0.2, np.arange(len(sorted(d["T"].unique())))
    Ts = sorted(d["T"].unique())
    for i, pol in enumerate(pols):
        v = [d[(d["T"] == t) & (d.policy == pol)].recomputed_tokens.values[0] / 1e6
             for t in Ts]
        ax[0].bar(x + (i - 1.5) * w, v, w, color=C[pol], label=LBL[pol])
    ax[0].set_xticks(x); ax[0].set_xticklabels([str(t) for t in Ts])
    ax[0].set_xlabel("turns $T$"); ax[0].set_ylabel("recomputed prefix\n(million tokens)")
    ax[0].legend(fontsize=7)
    for i, pol in enumerate(pols):
        v = [d[(d["T"] == t) & (d.policy == pol)].slo.values[0] for t in Ts]
        ax[1].plot(Ts, v, "o-", color=C[pol], label=LBL[pol], markersize=3.5)
    ax[1].set_xscale("log", base=2); ax[1].set_xticks(Ts)
    ax[1].set_xticklabels([str(t) for t in Ts])
    ax[1].set_xlabel("turns $T$"); ax[1].set_ylabel("SLO attainment"); ax[1].set_ylim(0, 1.05)
    fig.tight_layout(); save(fig, "f4_scalein_damage")


# --- F5: cold start ------------------------------------------------------
def f5():
    d = pd.read_csv(RES / "e5_coldstart.csv")
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.2))
    for pol in d.policy.unique():
        q = d[d.policy == pol].sort_values("cold_start_s")
        ax[0].plot(q.cold_start_s, q.slo, "o-", color=C[pol], label=LBL[pol], markersize=3.5)
        ax[1].plot(q.cold_start_s, q.gpu_s / 1000, "o-", color=C[pol], markersize=3.5)
    ax[0].set_xlabel("replica cold start (s)"); ax[0].set_ylabel("SLO attainment")
    ax[0].set_ylim(0, 1.05); ax[0].legend(fontsize=7, loc="lower left")
    ax[1].set_xlabel("replica cold start (s)"); ax[1].set_ylabel("GPU-seconds (thousands)")
    fig.tight_layout(); save(fig, "f5_coldstart")


# --- F6: holds on both traces -------------------------------------------
def f6():
    d = pd.read_csv(RES / "e6_workload_contrast.csv")
    fig, ax = plt.subplots(figsize=(4.4, 2.3))
    pols = ["hpa-gpu", "keda-queue", "kv-util", "park-aware"]
    groups = [("conv", 1), ("conv", 8), ("code", 1), ("code", 8)]
    labs = ["chat\n$T$=1", "chat\n$T$=8", "coding\n$T$=1", "coding\n$T$=8"]
    x = np.arange(len(groups)); w = 0.2
    for i, pol in enumerate(pols):
        v = [d[(d.trace == g) & (d["T"] == t) & (d.policy == pol)].slo.values[0]
             for g, t in groups]
        ax.bar(x + (i - 1.5) * w, v, w, color=C[pol], label=LBL[pol])
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_ylabel("SLO attainment"); ax.set_ylim(0, 1.28)
    ax.legend(ncol=2, fontsize=7, loc="upper center")
    fig.tight_layout(); save(fig, "f6_both_traces")


# --- F7: sensitivity (Appendix A) ---------------------------------------
def f7():
    path = RES / "e7_sensitivity.csv"
    if not path.exists():
        print("   f7 skipped (run exp7_sensitivity.py first)"); return
    d = pd.read_csv(path)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.1))
    keys = ["hpa-gpu", "keda-queue", "kv-util", "park-aware"]

    a = d[d.sweep == "kv_capacity"].sort_values("value")
    for p_ in keys:
        q = a[a.policy == p_]
        if len(q): ax[0].plot(q.value/1000, q.slo, "o-", color=C[p_],
                              label=LBL[p_], markersize=3.5)
    ax[0].set_xscale("log"); ax[0].set_xlabel("KV per replica (k tokens)")
    ax[0].set_ylabel("SLO attainment"); ax[0].set_ylim(0, 1.02)
    ax[0].legend(ncol=2, fontsize=6.5)

    b = d[d.sweep == "max_batch"].sort_values("value")
    for p_ in keys:
        q = b[b.policy == p_]
        if len(q): ax[1].plot(q.value, q.slo, "o-", color=C[p_], markersize=3.5)
    ax[1].set_xscale("log", base=2); ax[1].set_xlabel("max batch")
    ax[1].set_ylim(0, 1.02)

    c = d[d.sweep == "target"].sort_values("value")
    ax[2].plot(c.value, c.slo, "o-", color=C["park-aware"], markersize=4)
    ax[2].set_xlabel(r"ParkAware target $\theta$"); ax[2].set_ylim(0, 1.02)
    fig.tight_layout(); save(fig, "f7_sensitivity")


# --- F8: measurements that calibrate and falsify the model --------------
def f8():
    raw = ROOT / "vllm_measured" / "raw"
    v1 = pd.read_csv(raw / "v1_prefix_cache.csv")
    v2 = pd.read_csv(raw / "v2_parking_samples.csv")
    p1 = pd.read_csv(raw / "p1_cache_survival_summary.csv")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.05))

    # One representative parking cycle: compute-side and active-KV signals
    # all drop together, including vLLM's exported KV usage.
    q = v2[v2.t <= 16]
    ax[0].plot(q.t, q.gpu, color=C["hpa-gpu"], label="GPU util")
    ax[0].plot(q.t, q.running / 8 * 100, color=C["keda-queue"], label="running / 8")
    ax[0].plot(q.t, q.kv * 100, color=C["kv-util"], label="active KV")
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("native signal (%)")
    ax[0].set_title("(a) parked is invisible"); ax[0].legend(fontsize=6.5)

    g = v1.groupby("ctx_tokens", as_index=False).mean(numeric_only=True)
    ax[1].plot(g.ctx_tokens / 1000, g.ttft_miss_s, "o-", color=C["hpa-gpu"],
               label="cache miss")
    ax[1].plot(g.ctx_tokens / 1000, g.ttft_hit_s, "s-", color=C["kv-util"],
               label="cache hit")
    ax[1].set_yscale("log"); ax[1].set_xlabel("context (k tokens)")
    ax[1].set_ylabel("TTFT (s)"); ax[1].set_title("(b) state has value")
    ax[1].legend(fontsize=6.5)

    for tau, q in p1.groupby("tau_target_s"):
        ax[2].plot(q.neighbours, q.mean_survival_score, "o-",
                   label=rf"$\tau_{{\min}}$={tau:g}s")
    ax[2].set_xlabel("concurrent 4k-token neighbors")
    ax[2].set_ylabel("retained-prefix fraction")
    ax[2].set_ylim(-.03, 1.05); ax[2].set_title("(c) pressure evicts state")
    ax[2].legend(fontsize=6.5)
    fig.tight_layout(); save(fig, "f8_vllm_measured")


if __name__ == "__main__":
    print("figures:")
    f1(); f2(); f3(); f4(); f5(); f6(); f7(); f8()
