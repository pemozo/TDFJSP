"""Zeitprofile je Handarbeitsplatz und fuer die Ruestung (time_model.resource_factors).

Aufruf aus dem Projektordner:
    python -m unittest tests.test_resource_factors
"""
from __future__ import annotations

import copy
import unittest
from pathlib import Path

import torch

from generator import generate_class_instance
from monitoring.schedule_metrics import time_metrics, worker_metrics
from rl.env import TDFJSPEnv
from rl.observation import CANDIDATE_FEATURES, ObservationBuilder, first_feasible
from rl.policies import edd_ect, rollout
from rl.time_model import KIND_IDS, MANUAL, TimeModel
from schedule_validator import (check_config, check_schedule, config_fingerprint, duration, factor_at, load_config,
                                validate_instance)
from simulator import execute_plan, execution_instance, first_feasible_start, plan_from_schedule, view_config
from tests import test_rl_env

ROOT = Path(__file__).resolve().parent.parent
CONFIG = load_config(ROOT / "problem_config_zeitprofile.json")
CONFIG["environment"].update(device="cpu", allow_wait=True, view="time_aware", generator_workers=0)
CONFIG["instance_classes"]["test_klein"] = {"jobs": 8, "operations_per_job": 5, "workers": 2, "machines": 4, "days": 1}
CONFIG["instance_classes"]["test_gross"] = {"jobs": 12, "operations_per_job": 5, "workers": 4, "machines": 8, "days": 1}
INSTANCE = {"days": 2, "resources": {"manual_stations": ["H1", "H2", "H3", "H4"], "machines": ["M1"]}}
NAMES = INSTANCE["resources"]["manual_stations"]


class ConfigTest(unittest.TestCase):
    def test_project_config_stays_without_profiles(self):
        config = load_config(ROOT / "problem_config.json")
        self.assertNotIn("resource_factors", config["time_model"])
        model = TimeModel(config)
        self.assertIsNone(model._profile_factor)
        # Ohne Profile aendert die Ressource nichts an der Dauer
        base, start = torch.full((720,), 30), torch.arange(720)
        with_resource = model.duration(MANUAL, base, start, 1440, torch.zeros_like(start))
        without = model.duration(MANUAL, base, start, 1440)
        self.assertTrue(torch.equal(with_resource[0], without[0]))
        self.assertTrue(torch.equal(with_resource[1], without[1]))

    def test_profiles_are_different_and_have_large_drops(self):
        profiles = CONFIG["time_model"]["resource_factors"]
        rows = [tuple(row) for row in profiles["manual_stations"]] + [tuple(profiles["setup"])]
        self.assertEqual(len(set(rows)), len(rows))
        for row in rows:
            usable = [value for value in row if value is not None]
            drops = [a - b for a, b in zip(usable, usable[1:]) if a > b]
            self.assertGreaterEqual(max(drops), 0.5)

    def test_invalid_profiles_are_rejected(self):
        wrong_length = copy.deepcopy(CONFIG)
        wrong_length["time_model"]["resource_factors"]["setup"].append(1.0)
        wrong_break = copy.deepcopy(CONFIG)
        wrong_break["time_model"]["resource_factors"]["manual_stations"][0][6] = 1.0
        negative = copy.deepcopy(CONFIG)
        negative["time_model"]["resource_factors"]["setup"][0] = 0
        too_few = copy.deepcopy(CONFIG)
        too_few["time_model"]["resource_factors"]["manual_stations"].pop()
        for config in (wrong_length, wrong_break, negative, too_few):
            self.assertTrue(check_config(config))
        self.assertEqual(check_config(CONFIG), [])

    def test_profiles_change_the_fingerprint(self):
        changed = copy.deepcopy(CONFIG)
        changed["time_model"]["resource_factors"]["setup"][0] = 1.1
        self.assertNotEqual(config_fingerprint(changed), config_fingerprint(CONFIG))

    def test_static_view_flattens_all_profiles_and_keeps_breaks(self):
        static = view_config(CONFIG, "static")
        profiles = static["time_model"]["resource_factors"]
        for row in (*profiles["manual_stations"], profiles["setup"]):
            self.assertEqual({value for value in row if value is not None}, {1.0})
            self.assertIsNone(row[6])
        self.assertNotEqual(CONFIG["time_model"]["resource_factors"]["setup"][2], 1.0)


class DurationTest(unittest.TestCase):
    """Die Tensor-Formel muss je Platz und fuer die Ruestung dasselbe liefern wie der Validator."""

    def assert_same(self, config: dict) -> None:
        model = TimeModel(config)
        limit = 1440
        bases = torch.tensor([1, 13, 25, 37, 60]).unsqueeze(1).expand(-1, limit)
        starts = torch.arange(limit).expand(5, -1)
        cases = [(KIND_IDS["setup"], "setup", None), (KIND_IDS["automatic"], "automatic", None)]
        cases += [(MANUAL, "manual", station) for station in range(4)]
        for kind_id, kind, station in cases:
            resource = None if station is None else torch.full_like(starts, station)
            lengths, allowed = model.duration(kind_id, bases, starts, limit, resource)
            name = None if station is None else NAMES[station]
            for row, base in enumerate((1, 13, 25, 37, 60)):
                for t in range(0, limit, 3):
                    try:
                        expected = duration(config, INSTANCE, kind, base, t, name)
                    except ValueError:
                        expected = None
                    self.assertEqual(bool(allowed[row, t]), expected is not None, (kind, station, base, t))
                    if expected is not None:
                        self.assertEqual(int(lengths[row, t]), expected, (kind, station, base, t))

    def test_matches_validator(self):
        self.assert_same(CONFIG)

    def test_matches_validator_in_static_view(self):
        self.assert_same(view_config(CONFIG, "static"))

    def test_each_station_uses_its_own_profile(self):
        # 14:00: H1 Faktor 1,0, H2 Faktor 1,8; Ruestung 1,0
        self.assertEqual(duration(CONFIG, INSTANCE, "manual", 30, 480, "H1"), 30)
        self.assertEqual(duration(CONFIG, INSTANCE, "manual", 30, 480, "H2"), 54)
        # 15:00: Ruestung Faktor 1,7 auf jeder Maschine, automatische Operation ohne Faktor
        self.assertEqual(duration(CONFIG, INSTANCE, "setup", 10, 540), 17)
        self.assertEqual(duration(CONFIG, INSTANCE, "setup", 10, 540, "M1"), 17)
        self.assertEqual(duration(CONFIG, INSTANCE, "automatic", 30, 540, "M1"), 30)

    def test_finish_after_per_station(self):
        model = TimeModel(CONFIG)
        model.ensure_finish_table(60)
        for station, name in enumerate(NAMES):
            for base in (1, 25, 60):
                for t in range(0, 720, 11):
                    start = t
                    while True:
                        try:
                            length = duration(CONFIG, {**INSTANCE, "days": 3}, "manual", base, start, name)
                            break
                        except ValueError:
                            start += 1
                    found = model.finish_after(torch.tensor(MANUAL), torch.tensor(base), torch.tensor(t),
                                               torch.tensor(station))
                    self.assertEqual(int(found), start + length, (name, base, t))

    def test_first_feasible_per_station(self):
        model = TimeModel(CONFIG)
        limit = 1440
        times = torch.arange(0, limit)
        for station, name in enumerate(NAMES):
            for setup in (0, 13):
                start, setup_len, processing, found = first_feasible(
                    model, torch.full_like(times, MANUAL), torch.full_like(times, 37), torch.full_like(times, setup),
                    times, torch.full_like(times, limit), torch.full_like(times, station))
                for t in range(0, limit, 7):
                    expected = first_feasible_start(CONFIG, INSTANCE, "manual", 37, setup, t, name)
                    if expected is None:
                        self.assertFalse(bool(found[t]))
                    else:
                        self.assertEqual((int(start[t]), int(setup_len[t]), int(processing[t])), expected,
                                         (name, setup, t))

    def test_window_profile_per_station(self):
        model = TimeModel(CONFIG)
        kinds = torch.full((4,), MANUAL)
        lengths = model.window_profile(kinds, torch.full((4,), 30), torch.arange(4))
        for station, name in enumerate(NAMES):
            for k, start in enumerate(model.window_starts.tolist()):
                try:
                    expected = duration(CONFIG, INSTANCE, "manual", 30, start, name)
                except ValueError:
                    expected = 0
                self.assertEqual(int(lengths[station, k]), expected, (name, start))


class EnvironmentTest(unittest.TestCase):
    def instances(self, class_name: str, count: int, seed: int = 500) -> list[dict]:
        return [generate_class_instance(CONFIG, class_name, seed + i) for i in range(count)]

    def test_generated_instances_are_valid(self):
        for instance in self.instances("test_gross", 3):
            self.assertEqual(validate_instance(CONFIG, instance), [])
            self.assertEqual(check_schedule(CONFIG, instance, instance["reference_schedule"]), [])

    def test_env_matches_simulator_in_every_state(self):
        player = test_rl_env.SimulatorEquivalenceTest()
        for class_name in ("test_klein", "test_gross"):
            for seed, instance in enumerate(self.instances(class_name, 2, 700)):
                player.play_against_simulator(CONFIG, instance, seed, 0.2)

    def test_plans_pass_the_validator_and_replay(self):
        env = TDFJSPEnv(CONFIG)
        instances = self.instances("test_gross", 3, 800)
        td, _ = rollout(env, env.load_instances(instances), edd_ect)
        env.check_solution_validity(td, instances)
        for index, instance in enumerate(instances):
            schedule = env.to_schedule(td, index, instance)
            executed = execute_plan(CONFIG, instance, plan_from_schedule(schedule))
            self.assertEqual(check_schedule(CONFIG, execution_instance(CONFIG, instance), executed), [])
            self.assertEqual(sorted(executed, key=lambda e: e["op"]), sorted(schedule, key=lambda e: e["op"]))

    def test_candidates_use_the_station_profile(self):
        env = TDFJSPEnv(CONFIG)
        builder = ObservationBuilder(env)
        instance = self.instances("test_gross", 1, 900)[0]
        operations = {o["id"]: o for o in instance["operations"]}
        op_ids = [op for job in instance["jobs"] for op in job["operations"]]
        extended = execution_instance(CONFIG, instance)
        stations = instance["resources"]["manual_stations"]
        td = env.load_instances([instance])
        builder.static_inputs(td)
        processing_time = CANDIDATE_FEATURES.index("processing_time")
        checked = 0
        while not td["done"].all():
            data = builder.dynamic_inputs(td)
            now = int(td["time"][0])
            for job, resource in data["candidate_valid"][0].nonzero().tolist():
                op = operations[op_ids[int(td["next_op"][0, job])]]
                if resource >= len(stations) or op["kind"] != "manual":
                    continue
                name = stations[resource]
                earliest = max(now, int(td["job_ready"][0, job]), int(td["busy_until"][0, resource]))
                expected = first_feasible_start(CONFIG, extended, "manual", op["base_processing"][name], 0, earliest, name)
                self.assertAlmostEqual(float(data["candidates"][0, job, resource, processing_time]) * 60, expected[2], places=2)
                checked += 1
            td["action"] = edd_ect(td)
            td = env.step(td)["next"]
        self.assertGreater(checked, 50)

    def test_encoder_profiles_differ_per_station(self):
        env = TDFJSPEnv(CONFIG)
        builder = ObservationBuilder(env)
        instance = self.instances("test_gross", 1, 950)[0]
        inputs = builder.static_inputs(env.load_instances([instance]))
        manual = next(o for o in instance["operations"] if o["kind"] == "manual" and len(o["base_processing"]) > 1)
        op_ids = [op for job in instance["jobs"] for op in job["operations"]]
        rows = inputs["proc_profiles"][0, op_ids.index(manual["id"])]
        first, second = (int(name[1:]) - 1 for name in list(manual["base_processing"])[:2])
        self.assertFalse(torch.equal(rows[first], rows[second]))

    def test_agent_without_time_information_sees_factor_one(self):
        config = copy.deepcopy(CONFIG)
        config["model"]["time_information"] = "none"
        env = TDFJSPEnv(config)
        builder = ObservationBuilder(env)
        instance = self.instances("test_klein", 1, 975)[0]
        inputs = builder.static_inputs(env.load_instances([instance]))
        values = inputs["proc_profiles"][inputs["proc_profiles"] > 0]
        self.assertTrue(torch.all(values == 1.0))


class MetricsTest(unittest.TestCase):
    def test_worker_time_falls_when_waiting_for_the_cheap_window(self):
        instance = {"operations": [{"id": "a", "kind": "manual", "base_processing": {"H1": 30}}], "setups": [],
                    "resources": {"manual_stations": ["H1"], "machines": []}, "days": 1}
        now = worker_metrics(CONFIG, instance, [{"op": "a", "resource": "H1", "start": 470, "end": 524}])
        waited = worker_metrics(CONFIG, instance, [{"op": "a", "resource": "H1", "start": 480, "end": 510}])
        self.assertEqual(now, {"worker_busy_time": 54, "manual_extra_time": 24, "setup_extra_time": 0})
        self.assertEqual(waited, {"worker_busy_time": 30, "manual_extra_time": 0, "setup_extra_time": 0})

    def test_mean_factors_use_the_right_profile(self):
        instance = generate_class_instance(CONFIG, "test_klein", 123)
        schedule = [{"op": next(o["id"] for o in instance["operations"] if o["kind"] == "manual"),
                     "resource": "H2", "start": 480, "end": 534}]
        metrics = time_metrics(CONFIG, instance, schedule, None)
        self.assertEqual(metrics["mean_processing_factor"], factor_at(CONFIG, 480, "manual", 1))
        self.assertEqual(metrics["mean_processing_factor"], 1.8)


if __name__ == "__main__":
    unittest.main()
