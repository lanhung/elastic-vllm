#!/usr/bin/env python3
"""
vllm_validate.py -- turn the paper's four modelling assumptions into
measurements.  Runs against ONE vLLM instance on ONE GPU.

No Kubernetes, no root, no MIG.  An AutoDL container is enough.

WHAT THIS MEASURES AND WHY IT MATTERS
-------------------------------------
V1  prefix-cache hit vs miss
    -> the real price of destroying a program's KV cache.
       This is Eq. (4) in the paper, currently an assumption.

V2  the parking phenomenon
    -> GPU utilisation while a program holds KV but runs no kernel.
       This is the paper's central claim, currently only simulated.

V3  batch scaling curve
    -> fits R/(k + k_half); the simulator's service model, currently
       taken from the literature.

V4  cold start
    -> seconds from `vllm serve` to first served token.  The single
       most leveraged parameter in the paper (Sec. 5.5).

P1  prefix-cache survival
    -> whether a parked program retains a cheap re-entry path after time
       passes and concurrent neighbours consume the cache.

P2  pressure-aware admission
    -> compare 24 neighbours admitted before a parked target returns against
       admitting the P1-derived safe eight and deferring the remaining 16.

USAGE
-----
  # terminal 1
  bash serve.sh
  # terminal 2
  python3 vllm_validate.py --all --out results_vllm

Each experiment writes a CSV.  Feed them to make_vllm_figs.py.
"""
from __future__ import annotations

import argparse, json, os, random, signal, subprocess, time
from pathlib import Path

import numpy as np
import requests

BASE = os.environ.get("VLLM_BASE", "http://127.0.0.1:8000")
# serve.sh registers the model under a stable alias so the API name does
# not change when you swap checkpoints.
MODEL = os.environ.get("VLLM_API_MODEL", "agent-model")


# ----------------------------------------------------------------- utils
_UTIL_OK = None

def gpu_util() -> float:
    """SM utilisation, what HPA/DCGM reads.

    On a vGPU this may be unavailable, or may report the *physical* card
    and therefore include another tenant's work.  Either way we return NaN
    rather than a number we cannot defend, and V2 falls back to vLLM's own
    metrics -- which is arguably the better measurement anyway, since two
    of the three autoscalers we model read exactly those.
    """
    global _UTIL_OK
    if _UTIL_OK is False:
        return float("nan")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        v = float(out[0])
        _UTIL_OK = True
        return v
    except Exception:
        _UTIL_OK = False
        return float("nan")


def running_waiting() -> tuple[float, float]:
    """vllm:num_requests_running / _waiting -- what KEDA reads.
    Tenant-isolated: unaffected by anyone else on the physical card."""
    m = vllm_metrics()
    return (m.get("vllm:num_requests_running", float("nan")),
            m.get("vllm:num_requests_waiting", float("nan")))


def vllm_metrics() -> dict:
    """Scrape vLLM's Prometheus endpoint.  These are the same metric names
    the paper's simulator emits, so the substitution is mechanical."""
    try:
        txt = requests.get(f"{BASE}/metrics", timeout=5).text
    except Exception:
        return {}
    m = {}
    for line in txt.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        try:
            name, val = line.rsplit(" ", 1)
            key = name.split("{")[0]
            m[key] = float(val)
        except ValueError:
            continue
    return m


def kv_usage() -> float:
    m = vllm_metrics()
    for k in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc",
              "vllm:gpu_cache_usage"):
        if k in m:
            return m[k]
    return float("nan")


def make_prompt(n_tokens: int, tag: str) -> str:
    """~n_tokens of unique-but-repeatable text.  `tag` changes the prefix
    so we can force a cache miss on demand."""
    # Qwen tokenizes " x" as one token.  The unique session tag keeps
    # experiments isolated without multiplying the intended context length.
    body = " x" * n_tokens
    return f"Session {tag}. Context follows.\n{body}\nSummarise in one word."


def ttft(prompt: str, max_tokens: int = 1) -> float:
    """Time to first token, measured on the streaming API."""
    t0 = time.perf_counter()
    r = requests.post(f"{BASE}/v1/completions",
                      json={"model": MODEL, "prompt": prompt,
                            "max_tokens": max_tokens, "stream": True,
                            "temperature": 0.0},
                      stream=True, timeout=600)
    r.raise_for_status()
    for line in r.iter_lines():
        if line and line.strip() != b"data: [DONE]":
            return time.perf_counter() - t0
    return float("nan")


# ----------------------------------------------------------------- V1
def v1_prefix_cache(out: Path, lengths=(1000, 2000, 4000, 8000, 16000, 32000),
                    reps: int = 3):
    """The price of losing a KV cache.

    For each context length: send a prompt cold (miss), then send the SAME
    prompt again (hit).  The difference is exactly what a program pays when
    a scale-in destroys its cache.
    """
    rows = []
    for L in lengths:
        for rep in range(reps):
            tag = f"v1r{rep}L{L}"
            p = make_prompt(L, tag)
            miss = ttft(p)                     # first time: nothing cached
            time.sleep(0.3)
            hit = ttft(p)                      # second time: prefix cached
            rows.append(dict(ctx_tokens=L, rep=rep,
                             ttft_miss_s=miss, ttft_hit_s=hit,
                             recompute_penalty_s=miss - hit,
                             speedup=miss / hit if hit > 0 else float("nan")))
            print(f"  V1 L={L:6d} rep={rep} miss={miss:6.3f}s "
                  f"hit={hit:6.3f}s penalty={miss-hit:6.3f}s")
    _save(rows, out / "v1_prefix_cache.csv")


# ----------------------------------------------------------------- V2
def v2_parking(out: Path, n_sessions: int = 8, turns: int = 6,
               think_s: float = 8.0, ctx0: int = 2000, grow: int = 800):
    """The parking phenomenon, on real hardware.

    Runs `n_sessions` agent-like programs.  Each: prompt, generate, then
    sleep for `think_s` (a tool call), then prompt again with a longer
    transcript.  Throughout, we sample GPU utilisation and vLLM's KV cache
    usage every 250 ms.

    The claim to falsify: during think time, GPU util goes to ~0 while KV
    usage stays high.  If that is not what happens, the paper is wrong.
    """
    import threading
    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            run_, wait_ = running_waiting()
            samples.append(dict(t=time.perf_counter(),
                                gpu=gpu_util(),        # HPA / DCGM
                                running=run_,          # KEDA
                                waiting=wait_,         # KEDA
                                kv=kv_usage()))        # llm-d WVA
            time.sleep(0.25)

    th = threading.Thread(target=sampler, daemon=True); th.start()
    t_start = time.perf_counter()
    phases = []

    def run_session(sid: int):
        ctx = ctx0
        for k in range(turns):
            p = make_prompt(ctx, f"v2s{sid}")
            a = time.perf_counter()
            r = requests.post(f"{BASE}/v1/completions",
                              json={"model": MODEL, "prompt": p,
                                    "max_tokens": 32, "temperature": 0.0},
                              timeout=600)
            r.raise_for_status()
            b = time.perf_counter()
            phases.append(dict(sid=sid, turn=k, phase="compute",
                               t0=a - t_start, t1=b - t_start))
            if k < turns - 1:
                time.sleep(think_s)            # the tool call
                phases.append(dict(sid=sid, turn=k, phase="park",
                                   t0=b - t_start,
                                   t1=time.perf_counter() - t_start))
            ctx += grow

    ths = [threading.Thread(target=run_session, args=(i,)) for i in range(n_sessions)]
    for t in ths: t.start()
    for t in ths: t.join()
    stop.set(); th.join(timeout=2)

    for s in samples:
        s["t"] -= t_start

    # Headline: compare moments when every session is parked with moments
    # where at least one session is computing. The old implementation used a
    # union of park intervals ("any parked") and mislabeled it "all parked".
    import pandas as pd
    sm = pd.DataFrame(samples); ph = pd.DataFrame(phases)
    parked_count = np.zeros(len(sm), dtype=int)
    computing_count = np.zeros(len(sm), dtype=int)
    for _, r in ph.iterrows():
        mask = ((sm.t >= r.t0) & (sm.t <= r.t1)).values
        if r.phase == "park":
            parked_count += mask
        elif r.phase == "compute":
            computing_count += mask
    sm["parked_sessions"] = parked_count
    sm["computing_sessions"] = computing_count
    all_parked = parked_count == n_sessions
    any_compute = computing_count > 0

    _save(sm.to_dict("records"), out / "v2_parking_samples.csv")
    _save(phases, out / "v2_parking_phases.csv")

    def fmt(mask, lab):
        g, r, w, k = (sm.gpu[mask].mean(), sm.running[mask].mean(),
                      sm.waiting[mask].mean(), sm.kv[mask].mean())
        gs = "n/a (vGPU)" if np.isnan(g) else f"{g:5.1f}%"
        print(f"  {lab:20s} n={mask.sum():4d}  GPU={gs:>11s}  "
              f"running={r:5.2f}  waiting={w:5.2f}  KV={k*100:5.1f}%")

    print("\n  --- the three signals the autoscalers read ---")
    fmt(all_parked, "all sessions parked")
    fmt(any_compute, "any session compute")
    print("\n  The claim: during tool calls the two compute-side signals")
    print("  (GPU util, running/waiting) go to ~0 while KV stays high.")
    print("  If that is not what the numbers above show, the paper is wrong.")


# ----------------------------------------------------------------- V3
def fit_batch_model(ks, per_seq_tok_s) -> dict:
    """Least-squares fit of per-sequence throughput R/(k + k_half)."""
    xk = np.asarray(ks, dtype=float)
    y = np.asarray(per_seq_tok_s, dtype=float)

    def solve(k_half):
        basis = 1.0 / (xk + k_half)
        rate = float(np.dot(basis, y) / np.dot(basis, basis))
        residual = y - rate * basis
        return float(np.dot(residual, residual)), rate

    lo, hi = 0.0, max(1.0, float(xk.max()) * 20.0)
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    c, d = hi - (hi - lo) / phi, lo + (hi - lo) / phi
    for _ in range(160):
        if solve(c)[0] < solve(d)[0]:
            hi, d = d, c
            c = hi - (hi - lo) / phi
        else:
            lo, c = c, d
            d = lo + (hi - lo) / phi
    k_half = (lo + hi) / 2.0
    sse, rate = solve(k_half)
    total = float(np.dot(y - y.mean(), y - y.mean()))
    r2 = 1.0 - sse / total if total > 0 else float("nan")
    return dict(R=rate, k_half=k_half, r2=r2)


def v3_batch_curve(out: Path, ks=(1, 2, 4, 8, 16, 32), ctx: int = 2000,
                   gen: int = 64, reps: int = 3):
    """Per-request latency against concurrency; fits R/(k + k_half)."""
    from concurrent.futures import ThreadPoolExecutor
    rows = []
    for k in ks:
        for rep in range(reps):
            prompts = [make_prompt(ctx, f"v3k{k}r{rep}s{i}") for i in range(k)]
            t0 = time.perf_counter()
            def complete(p):
                r = requests.post(
                    f"{BASE}/v1/completions",
                    json={"model": MODEL, "prompt": p, "max_tokens": gen,
                          "temperature": 0.0}, timeout=600)
                r.raise_for_status()
                return r
            with ThreadPoolExecutor(max_workers=k) as ex:
                list(ex.map(complete, prompts))
            dt = time.perf_counter() - t0
            rows.append(dict(k=k, rep=rep, wall_s=dt,
                             per_req_s=dt, per_seq_tok_s=gen / dt,
                             agg_tok_s=k * gen / dt))
            print(f"  V3 k={k:3d} rep={rep} wall={dt:6.2f}s "
                  f"agg={k*gen/dt:8.1f} tok/s")
    _save(rows, out / "v3_batch_curve.csv")
    fit = fit_batch_model([r["k"] for r in rows],
                          [r["per_seq_tok_s"] for r in rows])
    _save([fit], out / "v3_batch_fit.csv")
    print(f"  fit per_seq=R/(k+k_half): R={fit['R']:.2f}, "
          f"k_half={fit['k_half']:.3f}, R^2={fit['r2']:.5f}")


# ----------------------------------------------------------------- V4
def v4_cold_start(out: Path, model: str | None = None, port: int = 8001,
                  reps: int = 3):
    """Seconds from process launch to first served token.

    Run this with NO vLLM already on the GPU.  It starts and kills its own.
    """
    model = model or os.environ.get("VLLM_MODEL", MODEL)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rep in range(reps):
        subprocess.run("pkill -f '[v]llm serve' || true", shell=True)
        time.sleep(8)
        t0 = time.perf_counter()
        server_log = open(out / f"v4_server_rep{rep}.log", "w")
        proc = subprocess.Popen(
            ["vllm", "serve", model, "--served-model-name", MODEL,
             "--port", str(port), "--enable-prefix-caching",
             "--max-model-len", "32768",
             "--gpu-memory-utilization", "0.88"],
            stdout=server_log, stderr=subprocess.STDOUT,
            start_new_session=True)
        ready = None
        while time.perf_counter() - t0 < 900:
            try:
                r = requests.post(f"http://127.0.0.1:{port}/v1/completions",
                                  json={"model": MODEL, "prompt": "hi",
                                        "max_tokens": 1}, timeout=5)
                if r.status_code == 200:
                    ready = time.perf_counter() - t0
                    break
            except Exception:
                pass
            time.sleep(1.0)
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)
        server_log.close()
        rows.append(dict(rep=rep, model=model, cold_start_s=ready))
        shown = "timeout" if ready is None else f"{ready:.1f}s"
        print(f"  V4 rep={rep} cold_start={shown}")
        time.sleep(5)
    _save(rows, out / "v4_cold_start.csv")


# ----------------------------------------------------------------- P1
def p1_cache_survival(out: Path, taus=(0.0, 8.0, 32.0),
                      neighbours=(0, 4, 8, 16, 24), target_ctx: int = 16000,
                      neighbour_ctx: int = 4000, reps: int = 3,
                      seed: int = 7):
    """Measure how long an opportunistic prefix-cache entry survives.

    Each trial measures a target prompt cold and immediately warm, making
    those timings its own miss/hit controls. It then parks that target while
    ``neighbours`` unique prompts run concurrently. The final target probe is
    classified against the midpoint of its controls and also reported as a
    continuous survival score: 1 is hit-like, 0 is miss-like.

    ``tau`` is the minimum wall-clock park duration including neighbour work.
    If pressure itself takes longer, ``park_actual_s`` records the overrun.
    Trials are shuffled deterministically so time-related drift is not
    confounded with increasing pressure.
    """
    from concurrent.futures import ThreadPoolExecutor

    def complete(prompt: str):
        r = requests.post(
            f"{BASE}/v1/completions",
            json={"model": MODEL, "prompt": prompt, "max_tokens": 1,
                  "temperature": 0.0}, timeout=600)
        r.raise_for_status()
        return r

    configs = [(float(tau), int(n), rep)
               for tau in taus for n in neighbours for rep in range(reps)]
    random.Random(seed).shuffle(configs)
    rows = []

    for trial, (tau, n, rep) in enumerate(configs):
        tag = f"p1t{trial}tau{tau:g}n{n}r{rep}"
        target = make_prompt(target_ctx, tag)

        miss = ttft(target)
        time.sleep(0.2)
        hit = ttft(target)
        kv_after_hit = kv_usage()

        pressure_prompts = [
            make_prompt(neighbour_ctx, f"{tag}peer{i}") for i in range(n)]
        park_start = time.perf_counter()
        pressure_start = time.perf_counter()
        if pressure_prompts:
            with ThreadPoolExecutor(max_workers=n) as ex:
                list(ex.map(complete, pressure_prompts))
        pressure_s = time.perf_counter() - pressure_start

        remaining = tau - (time.perf_counter() - park_start)
        if remaining > 0:
            time.sleep(remaining)
        park_actual_s = time.perf_counter() - park_start

        kv_before_probe = kv_usage()
        running, waiting = running_waiting()
        probe = ttft(target)

        span = miss - hit
        score = ((miss - probe) / span) if span > 0 else float("nan")
        if not np.isnan(score):
            score = float(np.clip(score, 0.0, 1.0))
        retained = bool(score >= 0.5) if not np.isnan(score) else False
        rows.append(dict(
            trial=trial, tau_target_s=tau, neighbours=n, rep=rep,
            target_ctx_tokens=target_ctx, neighbour_ctx_tokens=neighbour_ctx,
            miss_ttft_s=miss, hit_ttft_s=hit, probe_ttft_s=probe,
            pressure_s=pressure_s, park_actual_s=park_actual_s,
            survival_score=score, retained=retained,
            kv_after_hit=kv_after_hit, kv_before_probe=kv_before_probe,
            running_before_probe=running, waiting_before_probe=waiting))
        state = "HIT" if retained else "MISS"
        print(f"  P1 tau={tau:4.0f}s N={n:2d} rep={rep} "
              f"base={miss:5.2f}/{hit:5.2f}s probe={probe:5.2f}s "
              f"score={score:4.2f} {state} park={park_actual_s:5.1f}s")

    path = out / "p1_cache_survival.csv"
    _save(rows, path)

    import pandas as pd
    df = pd.DataFrame(rows)
    summary = (df.groupby(["tau_target_s", "neighbours"], as_index=False)
                 .agg(survival_probability=("retained", "mean"),
                      mean_survival_score=("survival_score", "mean"),
                      mean_probe_ttft_s=("probe_ttft_s", "mean"),
                      mean_park_actual_s=("park_actual_s", "mean")))
    summary.to_csv(out / "p1_cache_survival_summary.csv", index=False)
    print("\n  --- prefix-cache survival probability ---")
    print(summary.to_string(index=False))
    print(f"  -> {out / 'p1_cache_survival_summary.csv'}")


# ----------------------------------------------------------------- P2
def p2_pressure_admission(out: Path, neighbours: int = 24,
                          admission_limit: int = 8,
                          target_ctx: int = 16000,
                          neighbour_ctx: int = 4000,
                          reps: int = 3, seed: int = 19):
    """Validate the pressure-aware admission mechanism on one GPU.

    ``uncontrolled`` submits every neighbour before the target returns.
    ``protected`` submits only ``admission_limit`` neighbours, probes the
    returning target, then drains every deferred neighbour. Thus both modes
    complete the same work; protected admission exchanges cache-destructive
    work before the return for a client-side queue, matching the controller's
    reject-and-scale/queue action rather than merely lowering goodput.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not 0 < admission_limit <= neighbours:
        raise ValueError("admission_limit must be in [1, neighbours]")

    def complete(prompt: str):
        r = requests.post(
            f"{BASE}/v1/completions",
            json={"model": MODEL, "prompt": prompt, "max_tokens": 1,
                  "temperature": 0.0}, timeout=600)
        r.raise_for_status()
        return True

    configs = [(mode, rep) for mode in ("uncontrolled", "protected")
               for rep in range(reps)]
    random.Random(seed).shuffle(configs)
    rows = []
    for trial, (mode, rep) in enumerate(configs):
        tag = f"p2t{trial}{mode}r{rep}"
        target = make_prompt(target_ctx, tag)
        prompts = [make_prompt(neighbour_ctx, f"{tag}peer{i}")
                   for i in range(neighbours)]

        miss = ttft(target)
        time.sleep(0.2)
        hit = ttft(target)

        admitted = neighbours if mode == "uncontrolled" else admission_limit
        before = prompts[:admitted]
        deferred = prompts[admitted:]
        pressure_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, admitted)) as ex:
            completed_before = sum(ex.map(complete, before))
        pressure_s = time.perf_counter() - pressure_start

        probe = ttft(target)
        drain_start = time.perf_counter()
        completed_after = 0
        if deferred:
            with ThreadPoolExecutor(max_workers=admission_limit) as ex:
                completed_after = sum(ex.map(complete, deferred))
        deferred_drain_s = time.perf_counter() - drain_start

        span = miss - hit
        score = ((miss - probe) / span) if span > 0 else float("nan")
        if not np.isnan(score):
            score = float(np.clip(score, 0.0, 1.0))
        retained = bool(score >= 0.5) if not np.isnan(score) else False
        completed = completed_before + completed_after
        rows.append(dict(
            trial=trial, mode=mode, rep=rep,
            total_neighbours=neighbours,
            admitted_before_probe=admitted,
            deferred_until_after_probe=len(deferred),
            admission_limit=admission_limit,
            target_ctx_tokens=target_ctx,
            neighbour_ctx_tokens=neighbour_ctx,
            miss_ttft_s=miss, hit_ttft_s=hit, probe_ttft_s=probe,
            survival_score=score, retained=retained,
            pressure_before_probe_s=pressure_s,
            deferred_drain_s=deferred_drain_s,
            neighbours_completed=completed,
            goodput=completed / neighbours))
        state = "HIT" if retained else "MISS"
        print(f"  P2 {mode:12s} rep={rep} admitted={admitted:2d} "
              f"deferred={len(deferred):2d} probe={probe:5.2f}s "
              f"score={score:4.2f} {state} completed={completed}/{neighbours}")

    path = out / "p2_pressure_admission.csv"
    _save(rows, path)
    import pandas as pd
    df = pd.DataFrame(rows)
    summary = (df.groupby("mode", as_index=False)
                 .agg(survival_probability=("retained", "mean"),
                      mean_survival_score=("survival_score", "mean"),
                      mean_probe_ttft_s=("probe_ttft_s", "mean"),
                      mean_pressure_before_probe_s=(
                          "pressure_before_probe_s", "mean"),
                      mean_deferred_drain_s=("deferred_drain_s", "mean"),
                      mean_goodput=("goodput", "mean")))
    summary.to_csv(out / "p2_pressure_admission_summary.csv", index=False)
    print("\n  --- pressure-aware admission ---")
    print(summary.to_string(index=False))
    print(f"  -> {out / 'p2_pressure_admission_summary.csv'}")


# -----------------------------------------------------------------
def _save(rows, path: Path):
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  -> {path}")


def main():
    ap = argparse.ArgumentParser()
    for f in ("all", "v1", "v2", "v3", "v4", "p1", "p2"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--out", default="results_vllm")
    ap.add_argument("--p1-taus", default="0,8,32",
                    help="comma-separated minimum park durations in seconds")
    ap.add_argument("--p1-neighbours", default="0,4,8,16,24",
                    help="comma-separated concurrent neighbour counts")
    ap.add_argument("--p1-target-ctx", type=int, default=16000)
    ap.add_argument("--p1-neighbour-ctx", type=int, default=4000)
    ap.add_argument("--p1-reps", type=int, default=3)
    ap.add_argument("--p2-neighbours", type=int, default=24)
    ap.add_argument("--p2-admission-limit", type=int, default=8)
    ap.add_argument("--p2-target-ctx", type=int, default=16000)
    ap.add_argument("--p2-neighbour-ctx", type=int, default=4000)
    ap.add_argument("--p2-reps", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)

    if not (a.all or a.v4):
        m = vllm_metrics()
        if not m:
            raise SystemExit(f"no vLLM at {BASE}.  start it with serve.sh")
        print(f"vLLM up at {BASE}, model={MODEL}")

    if a.all or a.v1: print("\n[V1] prefix cache");   v1_prefix_cache(out)
    if a.all or a.v2: print("\n[V2] parking");        v2_parking(out)
    if a.all or a.v3: print("\n[V3] batch curve");    v3_batch_curve(out)
    if a.all or a.v4: print("\n[V4] cold start");     v4_cold_start(out)
    if a.p1:
        taus = tuple(float(x) for x in a.p1_taus.split(",") if x)
        neighbours = tuple(int(x) for x in a.p1_neighbours.split(",") if x)
        print("\n[P1] prefix-cache survival")
        p1_cache_survival(out, taus=taus, neighbours=neighbours,
                          target_ctx=a.p1_target_ctx,
                          neighbour_ctx=a.p1_neighbour_ctx,
                          reps=a.p1_reps)
    if a.p2:
        print("\n[P2] pressure-aware admission")
        p2_pressure_admission(
            out, neighbours=a.p2_neighbours,
            admission_limit=a.p2_admission_limit,
            target_ctx=a.p2_target_ctx,
            neighbour_ctx=a.p2_neighbour_ctx,
            reps=a.p2_reps)
    print("\ndone.")


if __name__ == "__main__":
    main()
