"""
rl_policy.py -- learning-based autoscalers.

Two of them, because "RL autoscaler" is not one thing:

  QLearnScaler   Tabular Q-learning over a discretised state.  This is the
                 classic formulation in the cloud-autoscaling literature
                 (Barrett et al., Concurrency & Computation 2013; see also
                 the Gari et al. 2021 survey).  It learns a policy from
                 reward, with no model of the system.

  PredictScaler  Proactive scaling on a forecast.  Holt-style double
                 exponential smoothing on recent arrivals, then provision
                 for the predicted load one cold-start ahead.  This is the
                 "predictive" branch of the standard autoscaling taxonomy
                 and is what you would build if you knew cold start was
                 the problem.

Both are trained/tuned on a disjoint prefix of the trace and evaluated on
the suffix, so nothing is fit on the data it is scored on.
"""
from __future__ import annotations

import numpy as np
from sim import Obs


# ----------------------------------------------------------------------
class QLearnScaler:
    """Tabular Q-learning.

    state  = (replica bucket, queue bucket, gpu-util bucket, kv-util bucket)
    action = delta replicas in {-2,-1,0,+1,+2}
    reward = -(replica cost) - lambda * (SLO violations in the interval)

    The agent sees exactly the signals a real autoscaler can read.  It is
    NOT given the parked count -- that is the whole question.  We also
    report a variant that does see it, to separate "RL is better" from
    "the signal is better".
    """
    name = "rl-qlearn"

    def __init__(self, see_parked: bool = False, alpha: float = 0.2,
                 gamma: float = 0.9, eps: float = 0.15,
                 cost_weight: float = 1.0, slo_weight: float = 12.0,
                 seed: int = 20260720):
        self.see_parked = see_parked
        self.name = "rl-qlearn+parked" if see_parked else "rl-qlearn"
        self.alpha, self.gamma, self.eps = alpha, gamma, eps
        self.cw, self.sw = cost_weight, slo_weight
        self.rng = np.random.default_rng(seed)
        self.actions = [-2, -1, 0, 1, 2]
        self.Q: dict[tuple, np.ndarray] = {}
        self._prev: tuple | None = None
        self._prev_a: int | None = None
        self.training = True
        # the environment tells us this between decisions
        self.pending_violations = 0
        self.pending_turns = 0

    # -- state discretisation ------------------------------------------
    def _state(self, o: Obs) -> tuple:
        rb = int(np.clip(np.log2(max(1, o.replicas)), 0, 6))
        qb = int(np.clip(np.log1p(o.queued) / np.log(4), 0, 4))
        gb = int(np.clip(o.gpu_util * 4, 0, 3))
        kb = int(np.clip(o.kv_util * 4, 0, 3))
        if self.see_parked:
            pb = int(np.clip(np.log1p(o.parked) / np.log(4), 0, 4))
            return (rb, qb, gb, kb, pb)
        return (rb, qb, gb, kb)

    def _q(self, s: tuple) -> np.ndarray:
        if s not in self.Q:
            self.Q[s] = np.zeros(len(self.actions))
        return self.Q[s]

    # -- called by the simulator every decision interval ---------------
    def decide(self, o: Obs) -> int:
        s = self._state(o)
        q = self._q(s)

        # learn from the previous step
        if self._prev is not None:
            viol = self.pending_violations / max(1, self.pending_turns)
            r = -(self.cw * o.replicas / 8.0) - self.sw * viol
            self._q(self._prev)[self._prev_a] += self.alpha * (
                r + self.gamma * q.max() - self._q(self._prev)[self._prev_a])
        self.pending_violations = 0
        self.pending_turns = 0

        if self.training and self.rng.random() < self.eps:
            a = int(self.rng.integers(len(self.actions)))
        else:
            a = int(np.argmax(q))
        self._prev, self._prev_a = s, a
        return max(1, o.replicas + self.actions[a])

    def reset_episode(self):
        self._prev = self._prev_a = None


# ----------------------------------------------------------------------
class PredictScaler:
    """Proactive scaling on a Holt double-exponential-smoothing forecast.

    Forecasts arrivals one cold-start horizon ahead and provisions for
    that, which is the textbook answer to "the cold start is too long to
    react".  It is a strong baseline precisely because it addresses the
    symptom our results are full of.
    """
    name = "predictive"

    def __init__(self, horizon_s: float = 180.0, alpha: float = 0.4,
                 beta: float = 0.2, target: float = 0.70,
                 max_batch: int = 32, decision_s: float = 15.0):
        self.h = horizon_s / decision_s      # steps ahead
        self.a, self.b = alpha, beta
        self.target, self.max_batch = target, max_batch
        self.level: float | None = None
        self.trend: float = 0.0

    def decide(self, o: Obs) -> int:
        x = float(o.running + o.queued)
        if self.level is None:
            self.level, self.trend = x, 0.0
        else:
            prev = self.level
            self.level = self.a * x + (1 - self.a) * (self.level + self.trend)
            self.trend = self.b * (self.level - prev) + (1 - self.b) * self.trend
        pred = max(0.0, self.level + self.h * self.trend)
        want = int(np.ceil(pred / (self.max_batch * self.target)))
        # memory floor, same as everyone else gets
        per = o.kv_capacity / max(1, o.replicas)
        if per > 0:
            want = max(want, int(np.ceil(o.kv_used / per)))
        return max(1, want)
