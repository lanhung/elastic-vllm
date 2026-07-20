import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from policies import Static
from sim import Cluster, HW, SLO
from workload import Program, Turn


def program(pid, cached=0, last_used=0.0):
    p = Program(pid=pid, arrival_s=0.0,
                turns=[Turn(prefill_tokens=1000, decode_tokens=1,
                            think_s=0.0)])
    p.cached_tokens = cached
    p.cache_last_used_s = last_used
    p.replica = 0
    return p


class CacheSemanticsTest(unittest.TestCase):
    def cluster(self):
        return Cluster([], HW(kv_capacity=75_216), SLO(), Static(1),
                       init_replicas=1, warmup_s=0)

    def test_parked_cache_is_resident_but_metric_invisible(self):
        c = self.cluster()
        parked = program(1, cached=16_000)
        c.parked[parked.pid] = parked
        c.replicas[0].resident[parked.pid] = parked
        c.replicas[0].kv_used = 16_000

        obs = c._observe(0.0)
        self.assertEqual(obs.kv_used, 0)
        self.assertEqual(obs.kv_util, 0)
        self.assertEqual(obs.cache_resident, 16_000)
        self.assertEqual(obs.parked_cached, 16_000)

    def test_capacity_pressure_partially_reclaims_oldest_parked_prefix(self):
        c = self.cluster()
        target = program(1, cached=16_000, last_used=1.0)
        active = program(2, cached=64_000, last_used=2.0)
        c.parked[target.pid] = target
        c.running[active.pid] = active
        r = c.replicas[0]
        r.resident = {target.pid: target, active.pid: active}
        r.kv_used = 80_000

        c._evict_if_needed(r, 3.0)

        self.assertEqual(r.kv_used, 75_216)
        self.assertEqual(target.cached_tokens, 11_216)
        self.assertEqual(active.cached_tokens, 64_000)
        self.assertEqual(c.evicted_tokens, 4_784)

    def test_place_respects_per_replica_batch_limit(self):
        c = Cluster([], HW(max_batch=1), SLO(), Static(2),
                    init_replicas=2, warmup_s=0)
        active = program(1)
        active.replica = 0
        c.running[active.pid] = active
        returning = program(2, cached=1000)
        returning.replica = 0

        placed = c._place(returning, 0.0)

        self.assertIsNotNone(placed)
        self.assertEqual(placed.rid, 1)

    def test_drain_completes_program_arriving_near_horizon(self):
        late = Program(pid=1, arrival_s=0.9,
                       turns=[Turn(prefill_tokens=1, decode_tokens=0,
                                   think_s=0.0)])
        hw = HW(service_rate=1000, w_decode=1, k_half=0,
                cold_start_s=0)
        c = Cluster([late], hw, SLO(), Static(1), dt=0.1,
                    init_replicas=1, warmup_s=0)

        result = c.run(1.0, drain_s=1.0)

        self.assertEqual(result.n_completed, 1)
        self.assertEqual(result.goodput, 1.0)


if __name__ == "__main__":
    unittest.main()
