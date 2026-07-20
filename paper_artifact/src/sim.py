"""
sim.py -- discrete-time simulator of an elastic LLM serving cluster under
agentic workloads.

MODEL
-----
Cluster of N replicas.  Each replica has
  * KV capacity  : kv_capacity tokens of resident context
  * Compute      : `service_rate` token-equivalents per second, shared
                   processor-sharing style among its active turns

A turn costs  prefill_tokens * w_prefill + decode_tokens * w_decode
token-equivalents.  Prefill is cheap per token and decode expensive per
token, which is why w_decode >> w_prefill.

An agent program alternates: run a turn on a replica, then *park* for its
tool think-time. While parked it does no compute. Its prefix-cache blocks
may remain physically present, but vLLM returns them to the reclaimable
pool: they are neither reserved for the program nor counted by the
exported KV-usage metric. A parked program is therefore invisible to all
three deployed signals even when a cheap cache hit remains possible.

Two ways a program loses cached prefix blocks:
  1. pressure  : newer blocks reclaim its LRU blocks, possibly partially
  2. scale-in  : the autoscaler removes the replica holding those blocks
The next turn re-prefills whatever fraction is missing. We count those
tokens as `recomputed` -- pure waste.

Everything reported by this simulator is computed from the simulation.
Nothing is asserted.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Callable

from workload import Program


def recommended_drain_s(programs: list[Program], hw: "HW") -> float:
    """Tail window long enough for an intrinsically slow program to finish.

    Arrivals stop at the experiment horizon. Without a drain window, long
    agent programs near that boundary are mechanically counted as failures
    even with unlimited replicas.
    """
    if not programs:
        return 0.0
    intrinsic = []
    for p in programs:
        seconds = 0.0
        for turn in p.turns:
            work = (turn.prefill_tokens * hw.w_prefill +
                    turn.decode_tokens * hw.w_decode)
            seconds += work * (1.0 + hw.k_half) / hw.service_rate
            seconds += turn.think_s
        intrinsic.append(seconds)
    return max(600.0, hw.cold_start_s + 1.5 * max(intrinsic))


# ----------------------------------------------------------------------
@dataclass
class HW:
    """One serving replica.

    Defaults are calibrated from the Qwen2.5-14B/vLLM 0.25.1 hardware run.
    V1 fixes prefill cost, V3 jointly fixes service rate, decode weight and
    k_half, V4 fixes process cold start, and the server log reports KV
    capacity. Broader capacities and cold starts remain explicit sweeps.
    """
    kv_capacity: int = 75_216         # measured GPU KV cache tokens
    service_rate: float = 19_607.0    # calibrated token-equivalents / s
    w_prefill: float = 1.0
    w_decode: float = 94.59            # jointly calibrated with V1 and V3
    max_batch: int = 32               # concurrent running turns
    k_half: float = 4.606              # V3 fit, R^2=0.99995
    cold_start_s: float = 38.155       # V4 process cold start, warm page cache
    drain_s: float = 5.0


@dataclass
class SLO:
    """SLO on per-turn *slowdown*, the standard metric when service times
    are heavy-tailed: a turn's latency divided by what that same turn
    would have taken on an unloaded replica.  An absolute latency bound
    is meaningless here because turn service times span two orders of
    magnitude (a 500-token turn and a 30k-token turn are both normal)."""
    max_slowdown: float = 3.0


@dataclass
class Obs:
    """What an autoscaler is allowed to see at a decision point."""
    t: float
    replicas: int
    running: int                      # turns actively computing
    queued: int                       # turns waiting for a replica slot
    parked: int                       # programs in tool think-time
    kv_used: int                      # active allocation exported by vLLM
    kv_capacity: int
    cache_resident: int               # active + reclaimable cached content
    parked_cached: int                # reclaimable content of parked programs
    gpu_util: float                   # running / (replicas*max_batch)
    kv_util: float


@dataclass
class Result:
    policy: str
    turns_per_program: int
    think_mean_s: float
    # outcomes
    n_programs: int = 0
    n_completed: int = 0
    p50_program_s: float = 0.0
    p99_program_s: float = 0.0
    p99_slowdown: float = 0.0
    slo_attain: float = 0.0           # fraction of turns meeting TTFT SLO
    gpu_seconds: float = 0.0
    recomputed_tokens: int = 0
    useful_prefill_tokens: int = 0
    waste_ratio: float = 0.0
    kv_kills: int = 0
    kills_evict: int = 0
    kills_scalein: int = 0
    evicted_tokens: int = 0
    scalein_tokens: int = 0
    scale_events: int = 0
    scale_ins: int = 0
    mean_replicas: float = 0.0
    max_replicas: int = 0
    unfinished_turns: int = 0
    goodput: float = 0.0


# ----------------------------------------------------------------------
class Replica:
    __slots__ = ("rid", "ready_at", "dying_at", "resident", "kv_used")

    def __init__(self, rid: int, ready_at: float):
        self.rid = rid
        self.ready_at = ready_at
        self.dying_at: float | None = None
        self.resident: dict[int, Program] = {}   # pid -> program with KV here
        self.kv_used = 0

    def ready(self, t: float) -> bool:
        return t >= self.ready_at and self.dying_at is None


# ----------------------------------------------------------------------
class Cluster:
    def __init__(self, programs: list[Program], hw: HW, slo: SLO,
                 policy: "Policy", dt: float = 0.5,
                 min_replicas: int = 1, max_replicas: int = 64,
                 decision_interval_s: float = 15.0,
                 init_replicas: int = 4,
                 warmup_s: float = 300.0):
        self.progs = programs
        self.hw = hw
        self.slo = slo
        self.policy = policy
        self.dt = dt
        self.min_r = min_replicas
        self.max_r = max_replicas
        self.decision_interval = decision_interval_s
        self.warmup_s = warmup_s

        self.replicas: list[Replica] = [Replica(i, 0.0) for i in range(init_replicas)]
        self._next_rid = init_replicas

        self.pending: list[Program] = sorted(programs, key=lambda p: p.arrival_s)
        self.queue: list[Program] = []          # waiting for a slot
        self.running: dict[int, Program] = {}   # pid -> program, computing
        self.parked: dict[int, Program] = {}    # pid -> program, in think time
        self.done: list[Program] = []

        self._work_left: dict[int, float] = {}  # pid -> token-equivalents left
        self._turn_start: dict[int, float] = {}
        self._turn_iso: dict[int, float] = {}    # isolated service time
        self._park_until: dict[int, float] = {}
        self.ttfts: list[float] = []
        self.slo_ok = 0
        self.slo_total = 0
        self.unfinished_turns = 0

        self.gpu_seconds = 0.0
        self.recomputed = 0
        self.useful_prefill = 0
        self.kv_kills = 0
        self.kills_evict = 0
        self.kills_scalein = 0
        self.evicted_tokens = 0
        self.scalein_tokens = 0
        self.scale_events = 0
        self.scale_ins = 0
        self._replica_hist: list[int] = []
        self.timeline: list[dict] = []

    # -------------------------------------------------------------- helpers
    def _ready_replicas(self, t):
        return [r for r in self.replicas if r.ready(t)]

    def _turn_work(self, turn, prefill_tokens: int) -> float:
        return (prefill_tokens * self.hw.w_prefill
                + turn.decode_tokens * self.hw.w_decode)

    def _observe(self, t) -> Obs:
        ready = self._ready_replicas(t)
        kv_cap = max(1, len(ready) * self.hw.kv_capacity)
        ready_ids = {r.rid for r in ready}
        cache_resident = sum(r.kv_used for r in ready)
        # vLLM 0.25 reports blocks allocated to active requests. Prefix-cache
        # content of completed requests remains reclaimable but reads as free.
        kv_used = sum(p.cached_tokens for p in self.running.values()
                      if p.replica in ready_ids)
        parked_cached = sum(p.cached_tokens for p in self.parked.values()
                            if p.replica in ready_ids)
        # gpu_util mirrors DCGM_FI_DEV_GPU_UTIL: fraction of SM-busy time.
        # A replica with at least one running turn is busy; one with only
        # parked programs is idle, however much memory it is holding.
        busy_rids = {p.replica for p in self.running.values()}
        n_busy = len([r for r in ready if r.rid in busy_rids])
        return Obs(t=t, replicas=len(ready), running=len(self.running),
                   queued=len(self.queue), parked=len(self.parked),
                   kv_used=kv_used, kv_capacity=kv_cap,
                   cache_resident=cache_resident,
                   parked_cached=parked_cached,
                   gpu_util=n_busy / max(1, len(ready)),
                   kv_util=kv_used / kv_cap)

    def _place(self, p: Program, t) -> Replica | None:
        """Prefer the replica already holding this program's KV."""
        running_per_replica: dict[int, int] = {}
        for q in self.running.values():
            running_per_replica[q.replica] = running_per_replica.get(q.replica, 0) + 1
        if p.replica is not None:
            for r in self.replicas:
                if (r.rid == p.replica and r.ready(t) and
                        running_per_replica.get(r.rid, 0) < self.hw.max_batch):
                    return r
        ready = [r for r in self._ready_replicas(t)
                 if running_per_replica.get(r.rid, 0) < self.hw.max_batch]
        if not ready:
            return None
        return min(ready, key=lambda r: (running_per_replica.get(r.rid, 0),
                                         len(r.resident), r.rid))

    def _evict_if_needed(self, r: Replica, t):
        """Reclaim oldest parked prefix blocks, allowing partial eviction."""
        while r.kv_used > self.hw.kv_capacity and r.resident:
            victims = [p for p in r.resident.values() if p.pid in self.parked]
            if not victims:
                # Active blocks cannot be reclaimed. A production scheduler
                # would queue or preempt before reaching this case.
                break
            v = min(victims, key=lambda p: (p.cache_last_used_s, p.pid))
            drop = min(v.cached_tokens, r.kv_used - self.hw.kv_capacity)
            if drop <= 0:
                break
            v.cached_tokens -= drop
            r.kv_used -= drop
            v.killed += 1
            self.kv_kills += 1
            self.kills_evict += 1
            self.evicted_tokens += drop
            if v.cached_tokens == 0:
                del r.resident[v.pid]

    def _kill_replica(self, r: Replica, t):
        for p in list(r.resident.values()):
            self.scalein_tokens += p.cached_tokens
            p.cached_tokens = 0
            p.killed += 1
            self.kv_kills += 1
            self.kills_scalein += 1
        r.resident.clear()
        r.kv_used = 0
        self.replicas = [x for x in self.replicas if x.rid != r.rid]

    # -------------------------------------------------------------- main
    def run(self, horizon_s: float, drain_s: float = 0.0) -> Result:
        t = 0.0
        next_decision = 0.0
        end_s = horizon_s + max(0.0, drain_s)
        while t < end_s:
            # ---- admissions
            while self.pending and self.pending[0].arrival_s <= t:
                p = self.pending.pop(0)
                p.start_s = t
                self.queue.append(p)

            # ---- autoscaler
            if t >= next_decision:
                obs = self._observe(t)
                target = int(np.clip(self.policy.decide(obs),
                                     self.min_r, self.max_r))
                cur = len([r for r in self.replicas if r.dying_at is None])
                if target > cur:
                    for _ in range(target - cur):
                        self.replicas.append(
                            Replica(self._next_rid, t + self.hw.cold_start_s))
                        self._next_rid += 1
                    self.scale_events += 1
                elif target < cur:
                    busy_rids = {p.replica for p in self.running.values()}
                    idle = sorted((r for r in self._ready_replicas(t)
                                   if r.rid not in busy_rids),
                                  key=lambda r: (len(r.resident), r.rid))
                    victims = idle[:cur - target]
                    for r in victims:
                        self._kill_replica(r, t)
                    if victims:
                        self.scale_events += 1
                        self.scale_ins += 1
                next_decision = t + self.decision_interval

            # ---- unpark
            for pid in [pid for pid, until in self._park_until.items() if until <= t]:
                p = self.parked.pop(pid)
                del self._park_until[pid]
                self.queue.append(p)

            # ---- dispatch
            still_q = []
            for p in self.queue:
                if len(self.running) >= len(self._ready_replicas(t)) * self.hw.max_batch:
                    still_q.append(p); continue
                r = self._place(p, t)
                if r is None:
                    still_q.append(p); continue
                turn = p.turns[p.turn_idx]
                # how much prefill must we actually do?
                if p.cached_tokens > 0 and r.rid == p.replica:
                    prefill = max(0, turn.prefill_tokens - p.cached_tokens)
                else:
                    prefill = turn.prefill_tokens
                # After turn zero, every missing prefix token was previously
                # computed and is therefore recomputation, including partial
                # block-LRU eviction measured by P1.
                if p.turn_idx > 0 and prefill > 0:
                    p.recomputed_tokens += prefill
                    self.recomputed += prefill
                self.useful_prefill += prefill
                p.replica = r.rid
                p.cache_last_used_s = t
                r.resident[p.pid] = p
                # KV footprint after this turn = full transcript so far
                p.cached_tokens = (turn.prefill_tokens + turn.decode_tokens
                                   + turn.tool_out_tokens)
                r.kv_used = sum(q.cached_tokens for q in r.resident.values())
                self._evict_if_needed(r, t)
                w = self._turn_work(turn, prefill)
                self._work_left[p.pid] = w
                self._turn_start[p.pid] = t
                # isolated = this turn alone on a replica (k=1)
                self._turn_iso[p.pid] = w * (1.0 + self.hw.k_half) / self.hw.service_rate
                self.running[p.pid] = p
            self.queue = still_q

            # ---- compute
            ready = self._ready_replicas(t)
            if t >= self.warmup_s:
                self.gpu_seconds += len(ready) * self.dt
            if self.running and ready:
                per_replica: dict[int, list[Program]] = {}
                for p in self.running.values():
                    per_replica.setdefault(p.replica, []).append(p)
                for rid, ps in per_replica.items():
                    k = len(ps)
                    share = self.hw.service_rate / (k + self.hw.k_half)
                    for p in ps:
                        self._work_left[p.pid] -= share * self.dt

            # ---- completions
            for pid in [pid for pid, w in self._work_left.items() if w <= 0]:
                p = self.running.pop(pid)
                p.cache_last_used_s = t
                del self._work_left[pid]
                lat = t - self._turn_start.pop(pid)
                iso = max(1e-6, self._turn_iso.pop(pid))
                slowdown = lat / iso
                if p.arrival_s >= self.warmup_s:
                    self.ttfts.append(slowdown)
                    self.slo_total += 1
                    if slowdown <= self.slo.max_slowdown:
                        self.slo_ok += 1
                # feed the reward signal to learning policies
                if hasattr(self.policy, "pending_turns"):
                    self.policy.pending_turns += 1
                    if slowdown > self.slo.max_slowdown:
                        self.policy.pending_violations += 1
                turn = p.turns[p.turn_idx]
                p.turn_idx += 1
                if p.turn_idx >= p.n_turns:
                    p.finish_s = t
                    self.done.append(p)
                    for r in self.replicas:
                        if r.rid == p.replica and p.pid in r.resident:
                            del r.resident[p.pid]
                            r.kv_used = sum(q.cached_tokens for q in r.resident.values())
                    p.cached_tokens = 0
                else:
                    if turn.think_s > 0:
                        self.parked[p.pid] = p
                        self._park_until[p.pid] = t + turn.think_s
                    else:
                        self.queue.append(p)

            if t >= self.warmup_s:
                self._replica_hist.append(len(ready))
            if len(self.timeline) < 100000 and int(t / self.dt) % 20 == 0:
                o = self._observe(t)
                self.timeline.append(asdict(o))
            t += self.dt
            if not (self.pending or self.queue or self.running or self.parked):
                break

        return self._result()

    def _finalise_unfinished(self):
        """A turn that never completed is an SLO violation, not a missing
        sample.  Without this, a policy that starves the cluster scores
        perfectly because only the few fast turns are ever observed."""
        stuck = 0
        active = (list(self.pending) + list(self.queue) +
                  list(self.running.values()) + list(self.parked.values()))
        for p in active:
            if p.arrival_s >= self.warmup_s:
                stuck += max(0, p.n_turns - p.turn_idx)
        self.slo_total += stuck
        self.unfinished_turns = stuck

    def _result(self) -> Result:
        self._finalise_unfinished()
        fin = [p for p in self.done if p.arrival_s >= self.warmup_s]
        lat = np.array([p.finish_s - p.start_s for p in fin]) if fin else np.array([0.0])
        tt = np.array(self.ttfts) if self.ttfts else np.array([0.0])
        hist = np.array(self._replica_hist) if self._replica_hist else np.array([0])
        return Result(
            policy=self.policy.name,
            turns_per_program=self.progs[0].n_turns if self.progs else 0,
            think_mean_s=0.0,
            n_programs=len(self.progs),
            n_completed=len(fin),
            p50_program_s=float(np.percentile(lat, 50)),
            p99_program_s=float(np.percentile(lat, 99)),
            p99_slowdown=float(np.percentile(tt, 99)),
            slo_attain=self.slo_ok / max(1, self.slo_total),
            gpu_seconds=self.gpu_seconds,
            recomputed_tokens=self.recomputed,
            useful_prefill_tokens=self.useful_prefill,
            waste_ratio=self.recomputed / max(1, self.useful_prefill),
            kv_kills=self.kv_kills,
            kills_evict=self.kills_evict,
            kills_scalein=self.kills_scalein,
            evicted_tokens=self.evicted_tokens,
            scalein_tokens=self.scalein_tokens,
            scale_events=self.scale_events,
            scale_ins=self.scale_ins,
            mean_replicas=float(hist.mean()),
            max_replicas=int(hist.max()),
            unfinished_turns=self.unfinished_turns,
            goodput=len(fin) / max(1, len([
                p for p in self.progs if p.arrival_s >= self.warmup_s])),
        )
