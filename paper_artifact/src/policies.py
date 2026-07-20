"""
policies.py -- autoscaling policies.

The four baselines are not straw men.  Each is what a real, currently
deployed system does:

  HpaGpuUtil   Kubernetes HPA on GPU utilisation.  The default a team gets
               by following the standard GPU-autoscaling guide.
  KedaQueue    KEDA on `vllm:num_requests_waiting`.  This is the metric the
               vLLM production guides and llm-d docs recommend.
  KvUtil       Scale on KV-cache utilisation.  This is what llm-d's
               Workload Variant Autoscaler does, and what HeteroScale
               lists as its planned direction.
  Oracle       Offline lower bound: the smallest replica count that would
               have met the SLO, computed with full knowledge of the future.

ParkAware is ours. Its whole content is one line: a program parked in a
tool call is invisible future demand with an evictable warm prefix, so
count it even though all native resource metrics currently read zero.
"""
from __future__ import annotations

import numpy as np
from sim import Obs


class Policy:
    name = "base"
    def decide(self, o: Obs) -> int:
        raise NotImplementedError
    def reset(self):
        pass


# ----------------------------------------------------------------------
class Static(Policy):
    def __init__(self, n: int):
        self.n = n
        self.name = f"static-{n}"
    def decide(self, o: Obs) -> int:
        return self.n


# ----------------------------------------------------------------------
class HpaGpuUtil(Policy):
    """Kubernetes HPA, the textbook formula:
         desired = ceil( current * currentMetric / targetMetric )
    with a stabilisation window on scale-down (default 300 s)."""
    name = "hpa-gpu"

    def __init__(self, target: float = 0.70, stabilize_s: float = 300.0):
        self.target = target
        self.stabilize_s = stabilize_s
        self._recent: list[tuple[float, int]] = []

    def decide(self, o: Obs) -> int:
        want = int(np.ceil(o.replicas * max(o.gpu_util, 1e-6) / self.target))
        want = max(1, want)
        # HPA scale-down stabilisation: take the max recommendation in window
        self._recent.append((o.t, want))
        self._recent = [(t, w) for t, w in self._recent
                        if t >= o.t - self.stabilize_s]
        if want < o.replicas:
            want = max(w for _, w in self._recent)
        return want


# ----------------------------------------------------------------------
class KedaQueue(Policy):
    """KEDA on queue depth.  Target: `target_per_replica` waiting requests."""
    name = "keda-queue"

    def __init__(self, target_per_replica: float = 5.0,
                 stabilize_s: float = 300.0):
        self.target = target_per_replica
        self.stabilize_s = stabilize_s
        self._recent: list[tuple[float, int]] = []

    def decide(self, o: Obs) -> int:
        want = int(np.ceil(o.queued / self.target)) if o.queued else 1
        # KEDA keeps replicas while there is any work in flight
        want = max(want, int(np.ceil(o.running / 32)), 1)
        self._recent.append((o.t, want))
        self._recent = [(t, w) for t, w in self._recent
                        if t >= o.t - self.stabilize_s]
        if want < o.replicas:
            want = max(w for _, w in self._recent)
        return want


# ----------------------------------------------------------------------
class KvUtil(Policy):
    """Scale on active KV allocation (llm-d Workload Variant Autoscaler).

    vLLM reports completed prefix-cache blocks as free/reclaimable, so a
    parked application registers here only while one of its turns is active.
    """
    name = "kv-util"

    def __init__(self, target: float = 0.80, stabilize_s: float = 300.0):
        self.target = target
        self.stabilize_s = stabilize_s
        self._recent: list[tuple[float, int]] = []

    def decide(self, o: Obs) -> int:
        want = int(np.ceil(o.replicas * max(o.kv_util, 1e-6) / self.target))
        want = max(1, want)
        self._recent.append((o.t, want))
        self._recent = [(t, w) for t, w in self._recent
                        if t >= o.t - self.stabilize_s]
        if want < o.replicas:
            want = max(w for _, w in self._recent)
        return want


# ----------------------------------------------------------------------
class ParkAware(Policy):
    """Ours.

    A parked program will return, but native vLLM metrics expose neither
    that future demand nor its opportunistically cached prefix. If pressure
    or scale-in reclaims the prefix, its next turn pays a partial or full
    re-prefill. The cost of being wrong on scale-in is asymmetric.

    desired = ceil( (running + parked) / (max_batch * target) )

    plus a shorter scale-in stabilisation guard. The parked count is an
    application-level signal, not a GPU or KV-util metric.
    """
    name = "park-aware"

    def __init__(self, target: float = 0.70, max_batch: int = 32,
                 scale_in_guard: bool = True, stabilize_s: float = 60.0):
        self.target = target
        self.max_batch = max_batch
        self.guard = scale_in_guard
        self.stabilize_s = stabilize_s
        self._recent: list[tuple[float, int]] = []

    def decide(self, o: Obs) -> int:
        # 1. compute-side demand: parked programs will come back, count them
        occupancy = o.running + o.parked + o.queued
        want = int(np.ceil(occupancy / (self.max_batch * self.target)))
        # 2. active memory demand: exported vLLM KV allocation is a floor
        per_replica_kv = o.kv_capacity / max(1, o.replicas)
        if per_replica_kv > 0:
            kv_floor = int(np.ceil(o.kv_used / (per_replica_kv * self.target)))
            want = max(want, kv_floor)
        want = max(1, want)
        self._recent.append((o.t, want))
        self._recent = [(t, w) for t, w in self._recent
                        if t >= o.t - self.stabilize_s]
        if want < o.replicas and self.guard:
            want = max(w for _, w in self._recent)
        return want


# ----------------------------------------------------------------------
class Oracle(Policy):
    """Offline lower bound.

    Replays a precomputed replica schedule that was found by binary search
    on a static configuration meeting the SLO at each point.  This is not
    implementable; it exists to bound how much any policy could gain.
    """
    name = "oracle"

    def __init__(self, schedule: list[tuple[float, int]]):
        self.schedule = sorted(schedule)
    def decide(self, o: Obs) -> int:
        n = 1
        for t, k in self.schedule:
            if t <= o.t:
                n = k
            else:
                break
        return n
