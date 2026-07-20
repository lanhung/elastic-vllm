"""
workload.py -- build agentic workloads from the real Azure LLM inference trace.

PROVENANCE
----------
Base data: Azure LLM Inference Trace 2023 (CC-BY), from
https://github.com/Azure/AzurePublicDataset
Released with: Patel et al., "Splitwise", ISCA 2024.
Fields: TIMESTAMP, ContextTokens, GeneratedTokens.

WHAT IS REAL AND WHAT IS CONSTRUCTED
------------------------------------
REAL (measured from the trace, never invented):
  * arrival timestamps  -> the arrival process, including its burstiness
  * ContextTokens       -> prefill work per LLM call
  * GeneratedTokens     -> decode work per LLM call

CONSTRUCTED (a parameter we sweep, NOT a claim about Azure):
  * which calls belong to the same agent program
  * tool think-time between turns

The Azure trace predates agentic serving and carries no program identifiers,
so no public trace can supply the second group.  We therefore treat program
structure as an explicit, swept parameter: turns-per-program T and tool
think-time tau.  Conclusions are reported over the whole sweep, so they do
not depend on any single choice.  T=1 exactly recovers the original
request-level (chat) workload, which is our control.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


# ----------------------------------------------------------------------
# Turn: one LLM call inside an agent program
# ----------------------------------------------------------------------
@dataclass
class Turn:
    prefill_tokens: int      # FULL context at this turn (accumulates)
    decode_tokens: int       # tokens to generate (real, from trace)
    think_s: float           # tool execution time AFTER this turn (constructed)
    tool_out_tokens: int = 0 # tokens the tool appends to the context


@dataclass
class Program:
    pid: int
    arrival_s: float
    turns: list[Turn]
    # runtime state, filled by the simulator
    turn_idx: int = 0
    replica: int | None = None
    cached_tokens: int = 0          # KV currently resident for this program
    cache_last_used_s: float = 0.0  # LRU timestamp for reclaimable prefix KV
    ctx_tokens: int = 0             # accumulated context length
    recomputed_tokens: int = 0      # waste caused by eviction / scale-in
    start_s: float | None = None
    finish_s: float | None = None
    killed: int = 0                 # times its KV was destroyed

    @property
    def n_turns(self) -> int:
        return len(self.turns)


# ----------------------------------------------------------------------
# Trace loading
# ----------------------------------------------------------------------
def load_azure(kind: str = "code") -> pd.DataFrame:
    """kind in {'code','conv'}.  Returns df with rel_s, ctx, gen."""
    f = DATA / f"AzureLLMInferenceTrace_{kind}.csv"
    df = pd.read_csv(f)
    t = pd.to_datetime(df["TIMESTAMP"])
    df = df.assign(rel_s=(t - t.min()).dt.total_seconds(),
                   ctx=df["ContextTokens"].astype(int),
                   gen=df["GeneratedTokens"].astype(int))
    return df[["rel_s", "ctx", "gen"]].sort_values("rel_s").reset_index(drop=True)


def trace_stats(df: pd.DataFrame, bin_s: float = 10.0) -> dict:
    """Measured properties of the arrival process.  All real."""
    span = float(df.rel_s.max() - df.rel_s.min())
    bins = np.arange(0, span + bin_s, bin_s)
    counts, _ = np.histogram(df.rel_s, bins=bins)
    ia = np.diff(df.rel_s.values)
    ia = ia[ia > 0]
    return {
        "n_requests": int(len(df)),
        "span_s": span,
        "rate_rps": len(df) / span,
        "ctx_mean": float(df.ctx.mean()),
        "ctx_p50": float(df.ctx.median()),
        "ctx_p99": float(df.ctx.quantile(0.99)),
        "gen_mean": float(df.gen.mean()),
        "gen_p50": float(df.gen.median()),
        "ctx_gen_ratio": float(df.ctx.mean() / df.gen.mean()),
        "index_of_dispersion": float(counts.var() / counts.mean()),
        "peak_to_mean": float(counts.max() / counts.mean()),
        "ia_cv": float(ia.std() / ia.mean()),
    }


# ----------------------------------------------------------------------
# Building agent programs on top of the real calls
# ----------------------------------------------------------------------
def build_programs(df: pd.DataFrame,
                   turns_per_program: int,
                   think_mean_s: float,
                   think_cv: float = 1.0,
                   seed: int = 20260720,
                   horizon_s: float | None = None,
                   tool_out_frac: float = 0.15,
                   load_multiplier: int = 1) -> list[Program]:
    """
    Compose real LLM calls into agent programs.

    A program takes `turns_per_program` consecutive calls from the trace.
    Its arrival time is the arrival time of its first call, so the
    program-level arrival process inherits the real burstiness.

    Think time is drawn log-normal with the requested mean and CV; tool
    latency in published agent measurements is strongly right-skewed, and
    log-normal is the standard fit.  With turns_per_program=1 and
    think_mean_s=0 this returns exactly the original trace.

    ``load_multiplier`` superposes deterministic phase-shifted copies of the
    trace-derived programs. It is a synthetic stress parameter, not an
    additional measured trace. Phase shifts avoid synchronised duplicate
    bursts while preserving each copy's within-trace structure.
    """
    rng = np.random.default_rng(seed)
    if horizon_s is not None:
        df = df[df.rel_s <= horizon_s]

    ctx = df.ctx.values
    gen = df.gen.values
    arr = df.rel_s.values
    n = len(df) // turns_per_program

    if think_mean_s > 0:
        sigma = np.sqrt(np.log(1 + think_cv ** 2))
        mu = np.log(think_mean_s) - sigma ** 2 / 2
        thinks = rng.lognormal(mu, sigma, size=n * turns_per_program)
    else:
        thinks = np.zeros(n * turns_per_program)

    programs = []
    for i in range(n):
        s = i * turns_per_program
        e = s + turns_per_program
        turns = []
        # Context ACCUMULATES across turns: the agent re-sends the whole
        # transcript every time.  ctx_0 comes from the trace; thereafter
        # ctx grows by what the model generated plus what the tool returned.
        c = int(ctx[s])
        for k, j in enumerate(range(s, e)):
            last = (k == turns_per_program - 1)
            tool_out = 0 if last else int(
                rng.lognormal(np.log(max(1.0, tool_out_frac * ctx[s])), 0.8))
            turns.append(Turn(prefill_tokens=c,
                              decode_tokens=int(gen[j]),
                              think_s=float(thinks[j]) if not last else 0.0,
                              tool_out_tokens=tool_out))
            c = c + int(gen[j]) + tool_out
        programs.append(Program(pid=i, arrival_s=float(arr[s]), turns=turns))
    if load_multiplier < 1:
        raise ValueError("load_multiplier must be >= 1")
    if load_multiplier == 1 or not programs:
        return programs

    period = float(horizon_s or max(p.arrival_s for p in programs) or 1.0)
    expanded = []
    for copy_idx in range(load_multiplier):
        phase = copy_idx * period / load_multiplier
        for p in programs:
            turns = [Turn(prefill_tokens=t.prefill_tokens,
                          decode_tokens=t.decode_tokens,
                          think_s=t.think_s,
                          tool_out_tokens=t.tool_out_tokens)
                     for t in p.turns]
            expanded.append(Program(
                pid=0,
                arrival_s=(p.arrival_s + phase) % period,
                turns=turns))
    expanded.sort(key=lambda p: p.arrival_s)
    for pid, p in enumerate(expanded):
        p.pid = pid
    return expanded


if __name__ == "__main__":
    import json
    out = {}
    for kind in ("code", "conv"):
        df = load_azure(kind)
        out[kind] = trace_stats(df)
        print(f"=== {kind} ===")
        for k, v in out[kind].items():
            print(f"  {k:22s} {v:,.4g}" if isinstance(v, float) else f"  {k:22s} {v:,}")
    (DATA.parent / "results").mkdir(exist_ok=True)
    (DATA.parent / "results" / "trace_stats.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/trace_stats.json")
