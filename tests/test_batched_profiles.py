"""Independent scalar checks for mixed instance profiles in one environment batch."""
from __future__ import annotations

import copy
from pathlib import Path
import unittest

import torch

from generator import generate_class_instance
from rl.env import TDFJSPEnv
from rl.parser import instances_to_td
from rl.policies import edd_ect, rollout
from rl.time_model import KIND_IDS, MANUAL, SETUP, TimeModel
from schedule_validator import check_schedule, duration, instance_config, load_config
from simulator import execution_instance


ROOT = Path(__file__).resolve().parent.parent


class BatchedProfilesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.config = load_config(ROOT / "problem_config_zeitprofile.json")
        cls.config["environment"].update(device="cpu", generator_workers=0, allow_wait=False)
        cls.config["instance_classes"]["profile_test"] = {
            "jobs": 4, "operations_per_job": 3, "workers": 2, "machines": 2, "days": 1}
        base = generate_class_instance(cls.config, "profile_test", 73)
        cls.instances = []
        for index in range(2):
            instance = copy.deepcopy(base)
            rows = []
            for station in range(2):
                rows.append([None if window["factor"] is None else
                             (1.0 if (k + station + index) % 3 else 1.5 + 0.2 * index)
                             for k, window in enumerate(cls.config["time_model"]["day_windows"])])
            setup = [None if window["factor"] is None else 1.0 + 0.3 * ((k + index) % 2)
                     for k, window in enumerate(cls.config["time_model"]["day_windows"])]
            instance["time_profiles"] = {"manual_stations": rows, "setup": setup}
            cls.instances.append(instance)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def bound(self, view="time_aware"):
        td = instances_to_td(self.config, self.instances)
        model = TimeModel(self.config, view)
        model.bind(td)
        return model, td

    def test_parser_preserves_each_profile_and_reference(self):
        original = copy.deepcopy(self.instances)
        _, td = self.bound()
        self.assertEqual(tuple(td["time_profile_factors"].shape), (2, 3, 12))
        for b, instance in enumerate(self.instances):
            effective = instance_config(self.config, instance)
            for k, window in enumerate(effective["time_model"]["day_windows"]):
                expected = -1 if window["factor"] is None else window["factor"]
                self.assertAlmostEqual(float(td["time_profile_reference"][b, k]), expected)
        self.assertEqual(original, self.instances)

    def test_mixed_duration_matches_scalar_validator(self):
        model, _ = self.bound()
        times = torch.arange(0, 1440, 17).expand(2, -1)
        base = torch.full_like(times, 29)
        cases = [(MANUAL, "manual", 0, "H1"), (MANUAL, "manual", 1, "H2"),
                 (SETUP, "setup", None, None), (KIND_IDS["automatic"], "automatic", 2, "M1")]
        for kind, name, resource, resource_name in cases:
            resources = None if resource is None else torch.full_like(times, resource)
            lengths, allowed = model.duration(kind, base, times, 1440, resources)
            for b, instance in enumerate(self.instances):
                executed = execution_instance(self.config, instance)
                for i, start in enumerate(times[b].tolist()):
                    try:
                        expected = duration(self.config, executed, name, 29, start, resource_name)
                    except ValueError:
                        expected = None
                    self.assertEqual(bool(allowed[b, i]), expected is not None, (b, name, start))
                    if expected is not None:
                        self.assertEqual(int(lengths[b, i]), expected, (b, name, start))

    def test_subset_indices_preserve_instance_identity(self):
        model, _ = self.bound()
        batch_index = torch.tensor([1, 0, 1])
        times = torch.tensor([180, 0, 300])
        base = torch.full_like(times, 30)
        resource = torch.tensor([1, 0, 0])
        lengths, allowed = model.duration(MANUAL, base, times, 1440, resource, batch_index)
        self.assertTrue(bool(allowed.all()))
        for i, b in enumerate(batch_index.tolist()):
            expected = duration(self.config, self.instances[b], "manual", 30, int(times[i]),
                                f"H{int(resource[i]) + 1}")
            self.assertEqual(int(lengths[i]), expected)
        with self.assertRaisesRegex(ValueError, "batch_index"):
            model.duration(MANUAL, base, times, 1440, resource)

    def test_finish_table_matches_first_feasible_scalar_start(self):
        model, _ = self.bound()
        model.ensure_finish_table(35)
        for resource, name, kind in [(0, "H1", MANUAL), (1, "H2", MANUAL), (None, None, SETUP)]:
            for initial in (0, 52, 119, 344, 365, 690, 719, 1060):
                base, times = torch.tensor([20, 35]), torch.full((2,), initial)
                resources = None if resource is None else torch.full((2,), resource)
                finishes = model.finish_after(kind, base, times, resources)
                for b, instance in enumerate(self.instances):
                    start = initial
                    while True:
                        try:
                            length = duration(self.config, {**instance, "days": 3},
                                              "setup" if kind == SETUP else "manual",
                                              int(base[b]), start, name)
                            break
                        except ValueError:
                            start += 1
                    self.assertEqual(int(finishes[b]), start + length, (b, resource, initial))
        index = torch.tensor([1, 0, 1])
        subset = model.finish_after(MANUAL, torch.tensor([20, 20, 20]), torch.tensor([344, 52, 719]),
                                    torch.tensor([0, 1, 0]), index)
        self.assertEqual(subset.shape, (3,))

    def test_mean_view_has_constant_time_weighted_resource_factors(self):
        model, td = self.bound("mean")
        times = torch.tensor([[0, 120, 390, 660], [0, 120, 390, 660]])
        weights = model.window_lengths.to(torch.float64)
        for station in (0, 1, 2):
            rows = td["time_profile_factors"][:, station]
            active = rows >= 0
            expected = (rows.clamp(min=0) * weights).sum(1) / (active * weights).sum(1)
            kind = SETUP if station == 2 else MANUAL
            resources = None if station == 2 else torch.full_like(times, station)
            actual = model._factor_for(torch.tensor(kind), times, resources)
            self.assertTrue(torch.allclose(actual, expected[:, None].expand_as(actual)))
        static, _ = self.bound("static")
        durations, _ = static.duration(MANUAL, torch.full_like(times, 20), times, 1440,
                                       torch.zeros_like(times))
        self.assertTrue(torch.equal(durations, torch.full_like(times, 20)))

    def test_batch_rollout_matches_singletons_and_reordered_batch(self):
        env = TDFJSPEnv(self.config)
        state, _ = rollout(env, env.load_instances(self.instances), edd_ect)
        batch_schedules = [env.to_schedule(state, b, instance) for b, instance in enumerate(self.instances)]
        for b, instance in enumerate(self.instances):
            self.assertEqual(check_schedule(self.config, execution_instance(self.config, instance),
                                            batch_schedules[b]), [])
            single, _ = rollout(env, env.load_instances([instance]), edd_ect)
            self.assertEqual(env.to_schedule(single, 0, instance), batch_schedules[b])
        reversed_instances = list(reversed(self.instances))
        reordered, _ = rollout(env, env.load_instances(reversed_instances), edd_ect)
        for b, instance in enumerate(reversed_instances):
            self.assertEqual(env.to_schedule(reordered, b, instance), batch_schedules[1 - b])

    def test_legacy_fixed_profile_path_and_reset_are_preserved(self):
        model, _ = self.bound()
        self.assertIsNotNone(model._batch_profiles)
        plain = copy.deepcopy(self.instances[0])
        del plain["time_profiles"]
        model.bind(instances_to_td(self.config, [plain]))
        self.assertIsNone(model._batch_profiles)
        expected = [w["factor"] for w in self.config["time_model"]["day_windows"]]
        self.assertEqual(model._factor[model.window_starts].tolist(),
                         [-1.0 if x is None else x for x in expected])


if __name__ == "__main__":
    unittest.main()
