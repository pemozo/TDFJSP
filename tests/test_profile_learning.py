"""Integration checks for profile ablation, flow features and the new objective."""
from __future__ import annotations

import copy
from pathlib import Path
import unittest

import torch

from generator import generate_class_instances
from monitoring.schedule_metrics import manual_profile_metrics, machine_metrics
from rl.env import TDFJSPEnv
from rl.observation import ObservationBuilder, CANDIDATE_FEATURES, FLOW_CANDIDATE_FEATURES
from rl.policy import build_model
from rl.policies import edd_ect, rollout
from rl.ppo import collect_rollout, ppo_update
from schedule_validator import load_config, instance_config, time_profile_id


CONFIG_PATH = Path(__file__).resolve().parents[1] / "problem_config_generalization.json"


def config():
    cfg = load_config(CONFIG_PATH)
    cfg["environment"].update(device="cpu", generator_workers=0)
    cfg["model"].update(d_model=32, hidden_dim=32, num_layers=1)
    return cfg


class ProfileLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = config()
        cls.instances, _, _ = generate_class_instances(cls.cfg, "klein_11", 73000, 2, CONFIG_PATH)

    def test_mean_ablation_retains_due_clock_but_removes_profile_order(self):
        first = copy.deepcopy(self.instances[0])
        second = copy.deepcopy(first)
        first["time_profiles"]["manual_stations"][0][:2] = [1.0, 1.6]
        second["time_profiles"]["manual_stations"][0][:2] = [1.6, 1.0]
        for instance in (first, second):
            instance["time_profile_id"] = time_profile_id(instance["time_profiles"])
        observations = {}
        for information in ("full", "mean"):
            cfg = copy.deepcopy(self.cfg)
            cfg["model"]["time_information"] = information
            env = TDFJSPEnv(cfg)
            td = env.load_instances([first, second])
            builder = ObservationBuilder(env)
            static, dynamic = builder.static_inputs(td), builder.dynamic_inputs(td)
            self.assertTrue((static["op_features"][..., 8] > 0).any())
            self.assertTrue((dynamic["time_features"][:, 4:] > 0).any())
            self.assertTrue(torch.isfinite(dynamic["candidates"]).all())
            self.assertEqual(dynamic["candidates"].shape[-1], len(CANDIDATE_FEATURES) + len(FLOW_CANDIDATE_FEATURES))
            observations[information] = static["proc_profiles"]
        self.assertFalse(torch.equal(*observations["full"].unbind(0)))
        self.assertTrue(torch.equal(*observations["mean"].unbind(0)))

    def test_new_reward_matches_independent_schedule_diagnostics(self):
        env = TDFJSPEnv(copy.deepcopy(self.cfg))
        initial = env.load_instances(self.instances)
        initial_cost = env.cost.cost(initial, env._completion(initial))
        td = initial
        accumulated = torch.zeros(2, dtype=torch.float64)
        while not td["done"].all():
            td["action"] = edd_ect(td)
            td = env.step(td)["next"]
            accumulated += td["reward"]
        env.check_solution_validity(td, self.instances)
        values = env.get_objectives(td)
        for i, instance in enumerate(self.instances):
            schedule = env.to_schedule(td, i, instance)
            manual = manual_profile_metrics(env.config, instance, schedule)
            machines = machine_metrics(instance, schedule)
            self.assertEqual(values["manual_profile_regret"][i].item(), manual["manual_profile_regret"])
            self.assertEqual(values["machine_flow_time"][i].item(), machines["machine_flow_time"])
        expected = initial_cost / env.cost.normalizer(initial) + env.get_reward(td)
        torch.testing.assert_close(accumulated, expected)
        # Delaying an automatic operation is never rewarded merely for being busy.
        delayed = {**td, "op_end": td["op_end"].clone()}
        automatic = td["op_kind"] == 1
        delayed["op_end"][automatic] += 5
        self.assertTrue((env.cost.cost(delayed, env._completion(td)) > env.cost.cost(td, env._completion(td))).all())

    def test_full_and_mean_can_collect_and_update_mixed_profile_batches(self):
        torch.manual_seed(13)
        for information in ("full", "mean"):
            cfg = copy.deepcopy(self.cfg)
            cfg["model"]["time_information"] = information
            cfg["training"]["ppo_epochs"] = 1
            env = TDFJSPEnv(cfg)
            model = build_model(env)
            builder = ObservationBuilder(env)
            td = env.load_instances(self.instances)
            data = collect_rollout(env, model, builder, td, cfg["training"])
            self.assertTrue(torch.isfinite(data.returns).all())
            loss = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-4), data, cfg["training"])
            self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in loss.values()))


if __name__ == "__main__":
    unittest.main()
