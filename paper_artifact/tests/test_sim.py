import sys
import unittest
import json
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
ARTIFACT = SRC.parent
sys.path.insert(0, str(SRC))

from policies import PressureAwareAdmission, Static
from sim import Cluster, HW, SLO
from workload import Program, Turn, build_programs


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

    def test_hw_defaults_match_machine_readable_calibration(self):
        summary = json.loads(
            (ARTIFACT / "vllm_measured/summary.json").read_text())
        calibration = summary["sim_calibration"]

        self.assertAlmostEqual(
            HW().service_rate,
            calibration["service_rate_token_equiv_s"], places=0)
        self.assertAlmostEqual(
            HW().w_decode, calibration["decode_weight"], places=3)

    def test_published_headlines_match_regenerated_results(self):
        results = ARTIFACT / "results"

        e2 = pd.read_csv(results / "e2_metric_divergence.csv")
        t8 = e2[(e2["T"] == 8) & (e2["tau"] == 8)].iloc[0]
        self.assertEqual(round(t8.parked_frac, 3), 0.895)
        self.assertEqual(round(t8.invisible_parked_frac, 3), 0.276)

        e3 = pd.read_csv(results / "e3_policy_sweep.csv")
        agentic = e3[(e3.turns_per_program > 1) & (e3.think_mean_s > 0)]
        means = agentic.groupby("policy").slo_attain.mean()
        self.assertEqual(round(means["hpa-gpu"], 3), 0.935)
        self.assertEqual(round(means["kv-util"], 3), 0.931)
        self.assertEqual(round(means["park-aware"], 3), 0.909)
        self.assertEqual(round(means["pressure-aware"], 3), 0.988)

        e4 = pd.read_csv(results / "e4_scale_in_damage.csv")
        baseline = e4[(e4["T"] > 1) & (e4.policy != "pressure-aware")]
        evicted = baseline.evicted_tokens.sum()
        scalein = baseline.scalein_tokens.sum()
        self.assertEqual(round(evicted / (evicted + scalein), 4), 0.9703)
        protected = e4[(e4["T"] > 1) &
                       (e4.policy == "pressure-aware")]
        self.assertEqual(int(protected.evicted_tokens.sum()), 0)

        e5 = pd.read_csv(results / "e5_coldstart.csv")
        measured = e5[e5.cold_start_s == 38.155].set_index("policy").slo
        self.assertEqual(round(measured["hpa-gpu"], 3), 0.997)
        self.assertEqual(round(measured["park-aware"], 3), 0.727)

        e8 = pd.read_csv(results / "e8_high_load.csv")
        high = e8[(e8.load_multiplier.isin([4, 8])) &
                  (e8.policy == "pressure-aware")]
        self.assertTrue((high.slo_attain == 1.0).all())

    def test_load_multiplier_superposes_phase_shifted_trace_copies(self):
        trace = pd.DataFrame({"rel_s": [0.0, 10.0],
                              "ctx": [100, 200], "gen": [10, 20]})
        programs = build_programs(trace, turns_per_program=1,
                                  think_mean_s=0, horizon_s=20,
                                  load_multiplier=4)

        self.assertEqual(len(programs), 8)
        self.assertEqual(len({p.pid for p in programs}), 8)
        self.assertTrue(all(0 <= p.arrival_s < 20 for p in programs))

    def test_pressure_admission_refuses_prefix_destructive_placement(self):
        c = Cluster([], HW(kv_capacity=75_216), SLO(),
                    PressureAwareAdmission(), init_replicas=1, warmup_s=0)
        parked = program(1, cached=70_000)
        c.parked[parked.pid] = parked
        c.replicas[0].resident[parked.pid] = parked
        c.replicas[0].kv_used = 70_000
        incoming = Program(pid=2, arrival_s=0.0,
                           turns=[Turn(prefill_tokens=10_000,
                                       decode_tokens=1, think_s=0.0)])

        self.assertIsNone(c._place(incoming, 0.0))


if __name__ == "__main__":
    unittest.main()
