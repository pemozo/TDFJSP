"""EDD+SLOT: kuerzester Zeitslot fuer manuelle Operationen, gewolltes Warten bleibt bei der Ausfuehrung erhalten."""
from __future__ import annotations

import unittest
from pathlib import Path

from baselines.dispatcher import build_plan, run_dispatcher
from evaluation import evaluate
from generator import generate_instance
from monitoring.schedule_metrics import manual_profile_metrics
from schedule_validator import load_config
from simulator import execute_plan, execution_instance, plan_from_schedule

ROOT = Path(__file__).resolve().parent.parent


class SlotRuleTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "problem_config_generalization.json")
        self.instances = [generate_instance(seed=seed, n_jobs=6, n_machines=3, n_ops=5, n_workers=2,
                                            config=self.config) for seed in (3, 4, 5)]

    def test_slot_plans_are_feasible_and_execution_keeps_planned_starts(self):
        for instance in self.instances:
            result = run_dispatcher(self.config, instance, "EDD", "SLOT", "time_aware")
            planned = sorted((e["op"], e["resource"], e["start"], e["end"]) for e in result["planned"])
            executed = sorted((e["op"], e["resource"], e["start"], e["end"]) for e in result["executed"])
            self.assertEqual(planned, executed)
            score = evaluate(self.config, execution_instance(self.config, instance), result["executed"])
            self.assertTrue(score["feasible"])

    def test_slot_uses_cheaper_windows_than_ect(self):
        slot = ect = 0.0
        for instance in self.instances:
            for routing in ("SLOT", "ECT"):
                executed = run_dispatcher(self.config, instance, "EDD", routing, "time_aware")["executed"]
                factor = manual_profile_metrics(self.config, instance, executed)["weighted_manual_factor"]
                if routing == "SLOT":
                    slot += factor
                else:
                    ect += factor
        self.assertLess(slot, ect)

    def test_without_kept_starts_execution_is_unchanged_for_other_rules(self):
        instance = self.instances[0]
        planned = build_plan(self.config, instance, "EDD", "ECT", "time_aware")
        plain = plan_from_schedule(planned)
        self.assertNotIn("not_before", plain)
        kept = plan_from_schedule(planned, keep_starts=True)
        self.assertEqual(execute_plan(self.config, instance, plain), execute_plan(self.config, instance, kept))

    def test_static_view_behaves_like_ect(self):
        for instance in self.instances:
            slot = build_plan(self.config, instance, "EDD", "SLOT", "static")
            ect = build_plan(self.config, instance, "EDD", "ECT", "static")
            self.assertEqual(slot, ect)


if __name__ == "__main__":
    unittest.main()
