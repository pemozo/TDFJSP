"""Instance-owned calendars: generation, scalar validation, static planning and replay."""
from __future__ import annotations

import copy
import json
import math
import unittest
from pathlib import Path

from baselines.dispatcher import build_plan, run_dispatcher
from evaluation import evaluate
from generator import generate_instance
from schedule_validator import (check_config, check_schedule, config_fingerprint, duration,
                                factor_at, instance_config, load_config, time_profile_id,
                                validate_instance)
from simulator import execute_plan, execution_instance, plan_from_schedule, view_config


ROOT = Path(__file__).resolve().parent.parent


class InstanceProfilesTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "problem_config_zeitprofile.json")
        self.config["generation"]["instance_time_profiles"] = {
            "enabled": True, "min_factor": 1.0, "max_factor": 1.8, "jitter": 0.05}

    def generate(self, seed=42, **kwargs):
        return generate_instance(seed=seed, n_jobs=4, n_machines=3, n_ops=3,
                                 n_workers=2, config=self.config, **kwargs)

    def test_profiles_reproduce_and_do_not_advance_job_or_setup_rng(self):
        before = copy.deepcopy(self.config)
        first, same, other = self.generate(), self.generate(), self.generate(43)
        self.assertEqual(first, same)
        self.assertNotEqual(first["time_profiles"], other["time_profiles"])
        self.assertNotEqual(first["time_profile_id"], other["time_profile_id"])
        self.assertEqual(first["time_profile_id"], time_profile_id(first["time_profiles"]))
        self.assertEqual(self.config, before)
        self.config["generation"]["instance_time_profiles"]["enabled"] = False
        fixed = self.generate()
        self.assertEqual(first["operations"], fixed["operations"])
        self.assertEqual(first["setups"], fixed["setups"])
        self.assertNotIn("time_profiles", fixed)
        for row in [*first["time_profiles"]["manual_stations"], first["time_profiles"]["setup"]]:
            self.assertEqual([v is None for v in row],
                             [w["factor"] is None for w in before["time_model"]["day_windows"]])
            values = [v for v in row if v is not None]
            self.assertEqual(min(values), 1.0)
            self.assertLessEqual(max(values), 1.8)
            self.assertGreater(max(values), 1.0)

    def test_legacy_disabled_is_identical_to_absent_setting(self):
        self.config["generation"]["instance_time_profiles"]["enabled"] = False
        disabled = self.generate()
        del self.config["generation"]["instance_time_profiles"]
        legacy = self.generate()
        self.assertEqual(disabled, legacy)
        self.assertIs(instance_config(self.config, legacy), self.config)

    def test_generated_reference_and_serialized_instance_validate(self):
        instance = json.loads(json.dumps(self.generate()))
        self.assertEqual(validate_instance(self.config, instance), [])
        self.assertEqual(check_schedule(self.config, instance, instance["reference_schedule"]), [])
        self.assertEqual(evaluate(self.config, instance, instance["reference_schedule"])["total_tardiness"], 0)
        effective = instance_config(self.config, instance)
        self.assertEqual(config_fingerprint(effective), config_fingerprint(self.config))
        self.assertEqual(validate_instance(effective, instance), [])
        self.assertIs(instance_config(effective, instance), effective)
        for i, window in enumerate(effective["time_model"]["day_windows"]):
            if window["factor"] is not None:
                expected = sum(row[i] for row in instance["time_profiles"]["manual_stations"]) / 2
                self.assertAlmostEqual(window["factor"], expected)

    def test_missing_or_malformed_instance_profiles_are_rejected(self):
        instance = self.generate()
        missing = copy.deepcopy(instance)
        del missing["time_profiles"]
        self.assertTrue(validate_instance(self.config, missing))
        with self.assertRaises(ValueError):
            instance_config(self.config, missing)
        for bad in (None, [], {}, {"manual_stations": [], "setup": []}):
            malformed = {**instance, "time_profiles": bad}
            self.assertTrue(validate_instance(self.config, malformed))
            self.assertTrue(check_schedule(self.config, malformed, instance["reference_schedule"]))
        for bad in (0, -1, float("nan"), float("inf"), True, "1.5", None):
            malformed = copy.deepcopy(instance)
            malformed["time_profiles"]["manual_stations"][0][0] = bad
            self.assertTrue(validate_instance(self.config, malformed), bad)
        malformed = copy.deepcopy(instance)
        break_index = next(i for i, w in enumerate(self.config["time_model"]["day_windows"]) if w["factor"] is None)
        malformed["time_profiles"]["setup"][break_index] = 1.0
        self.assertTrue(validate_instance(self.config, malformed))

    def test_scalar_durations_and_static_view_both_binding_orders(self):
        instance = self.generate()
        for i, row in enumerate(instance["time_profiles"]["manual_stations"]):
            row[:] = [None if v is None else (1.8 if i == 0 else 1.1) for v in row]
        instance["time_profiles"]["setup"][:] = [None if v is None else 1.6
                                                  for v in instance["time_profiles"]["setup"]]
        effective = instance_config(self.config, instance)
        self.assertEqual(duration(self.config, instance, "manual", 30, 60, "H1"), 54)
        self.assertEqual(duration(self.config, instance, "manual", 30, 60, "H2"), 33)
        self.assertEqual(duration(self.config, instance, "setup", 10, 60), 16)
        self.assertEqual(duration(self.config, instance, "automatic", 30, 60, "M1"), 30)
        self.assertAlmostEqual(factor_at(effective, 60), 1.45)
        for static in (view_config(effective, "static"), instance_config(view_config(self.config, "static"), instance)):
            self.assertEqual(duration(static, instance, "manual", 30, 60, "H1"), 30)
            self.assertEqual(duration(static, instance, "setup", 10, 60), 10)
            with self.assertRaises(ValueError):
                duration(static, instance, "manual", 30, 360, "H1")

    def test_dispatcher_and_replay_use_instance_factors(self):
        instance = self.generate()
        for view in ("static", "time_aware"):
            result = run_dispatcher(self.config, instance, "EDD", "ECT", view)
            extended = execution_instance(self.config, instance)
            self.assertEqual(check_schedule(view_config(self.config, view), extended, result["planned"]), [])
            self.assertTrue(evaluate(self.config, extended, result["executed"])["feasible"])
            operations = {op["id"]: op for op in instance["operations"]}
            for entry in result["executed"]:
                op = operations[entry["op"]]
                factor = factor_at(instance_config(self.config, instance), entry["start"], "manual",
                                   instance["resources"]["manual_stations"].index(entry["resource"])) if op["kind"] == "manual" else 1
                self.assertEqual(entry["end"] - entry["start"],
                                 math.ceil(op["base_processing"][entry["resource"]] * factor - 1e-9))
        planned = build_plan(self.config, instance, "EDD", "ECT", "time_aware")
        replay = execute_plan(self.config, instance, plan_from_schedule(planned))
        self.assertEqual(sorted(planned, key=lambda e: e["op"]), sorted(replay, key=lambda e: e["op"]))

    def test_invalid_generator_settings_are_rejected(self):
        for override in ({"enabled": "yes"}, {"min_factor": -1}, {"max_factor": 1},
                         {"jitter": -0.1}, {"jitter": float("nan")}, {"typo": True}):
            config = copy.deepcopy(self.config)
            config["generation"]["instance_time_profiles"].update(override)
            self.assertTrue(check_config(config), override)


if __name__ == "__main__":
    unittest.main()
