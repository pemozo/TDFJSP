"""Tests des neuronalen Modells, der Beobachtung und des PPO-Trainings.

Aufruf aus dem Projektordner: python -m unittest tests.test_model
Der ueberwachte Lerntest steht in tests/synthetic_learning.py und wird hier mit einem Muster geprueft.
"""
from __future__ import annotations

import copy
import csv
import math
import random
import tempfile
import unittest
from pathlib import Path

import torch

from generator import generate_class_instance, save_instance
from rl.encoder import masked_softmax
from rl.env import TDFJSPEnv
from rl.evaluate import evaluate_policy
from rl.observation import CANDIDATE_FEATURES, ObservationBuilder, first_feasible
from rl.policies import edd_ect, rollout
from rl.policy import NeuralPolicy, VARIANTS, build_model, load_checkpoint
from rl.ppo import collect_rollout, compute_returns, ppo_update
from rl.time_model import KIND_IDS, TimeModel
from rl.train import train
from schedule_validator import KINDS, all_resources, duration, load_config, setup_resources
from simulator import execution_instance, first_feasible_start, setup_lookup
from tests.synthetic_learning import train_and_measure

CONFIG = load_config()
CONFIG["environment"].update(device="cpu", allow_wait=True, view="time_aware", compute_features=False,
                             train_classes=["mini"], generator_workers=0)
CONFIG["reward"].update(mode="stepwise", levels=["total_tardiness", "makespan"], level_weights=None)
CONFIG["simulation"]["overflow_days"] = 1
CONFIG["instance_classes"].update(
    mini={"jobs": 4, "operations_per_job": 3, "workers": 1, "machines": 2, "days": 1},
    test_klein={"jobs": 10, "operations_per_job": 5, "workers": 2, "machines": 4, "days": 1})
CONFIG["model"].update(variant="A3", time_information="full", d_model=128, num_layers=3, num_heads=4)
FEATURE = {name: i for i, name in enumerate(CANDIDATE_FEATURES)}


def variant_config(variant: str = "A3", time_information: str = "full", small: bool = False,
                   **environment) -> dict:
    config = copy.deepcopy(CONFIG)
    config["model"].update(variant=variant, time_information=time_information)
    if small:
        config["model"].update(d_model=32, num_layers=2, num_heads=4, hidden_dim=32)
    config["environment"].update(environment)
    return config


def instances(class_name: str = "test_klein", count: int = 3, first_seed: int = 5000) -> list[dict]:
    return [generate_class_instance(CONFIG, class_name, first_seed + i) for i in range(count)]


def play_until(env: TDFJSPEnv, td: dict, minute: int) -> dict:
    while not td["done"].all() and int(td["time"].min()) < minute:
        td["action"] = edd_ect(td)
        td = env.step(td)["next"]
    return td


class ShapeTest(unittest.TestCase):
    def test_all_variants_have_documented_shapes(self):
        for variant in VARIANTS:
            env = TDFJSPEnv(variant_config(variant))
            model = build_model(env)
            builder = ObservationBuilder(env)
            td = env.load_instances(instances())
            static, dynamic = builder.static_inputs(td), builder.dynamic_inputs(td)
            B, O, R, J = 3, 50, 6, 10
            K = len(env.time_model.window_starts)
            self.assertEqual(static["proc_profiles"].shape, (B, O, R, K))
            self.assertEqual(static["eligible"].shape, (B, O, R))
            self.assertEqual(static["precedence"].shape, (B, O, O))
            self.assertEqual(static["op_features"].shape[:2], (B, O))
            self.assertEqual(static["machine_features"].shape[:2], (B, R))
            self.assertEqual(dynamic["candidates"].shape, (B, J, R, len(CANDIDATE_FEATURES)))
            h_op, h_ma = model.encode(static)
            self.assertEqual(h_op.shape, (B, O, 128))
            self.assertEqual(h_ma.shape, (B, R, 128))
            logits, value = model(h_op, h_ma, static["pad_mask"], dynamic)
            self.assertEqual(logits.shape, (B, 1 + J * R))
            self.assertEqual(value.shape, (B,))

    def test_configured_size_is_used(self):
        settings = CONFIG["model"]
        self.assertEqual((settings["d_model"], settings["num_layers"], settings["num_heads"]), (128, 3, 4))


class MaskingTest(unittest.TestCase):
    def test_forbidden_actions_are_never_chosen(self):
        env = TDFJSPEnv(variant_config(small=True))
        model = build_model(env)
        policy = NeuralPolicy(env, model, "sample", seed=3)
        td = env.load_instances(instances("mini", 4))
        while not td["done"].all():
            builder_logits, _ = model(*policy._hidden, policy._pad_mask, policy.builder.dynamic_inputs(td)) \
                if policy._hidden is not None else (None, None)
            action = policy(td)
            self.assertTrue(bool(td["action_mask"].gather(1, action.unsqueeze(1)).all()))
            if builder_logits is not None:
                self.assertTrue(bool((builder_logits[~td["action_mask"]] <= -1e8).all()))
            td["action"] = action
            td = env.step(td)["next"]

    def test_masked_softmax(self):
        scores = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        mask = torch.tensor([[True, False, True], [False, False, False]])
        result = masked_softmax(scores, mask, dim=1)
        self.assertAlmostEqual(float(result[0].sum()), 1.0, places=5)
        self.assertEqual(float(result[0, 1]), 0.0)
        self.assertEqual(result[1].tolist(), [0.0, 0.0, 0.0])


class IsolatedAndPaddedNodesTest(unittest.TestCase):
    def _padded(self, static: dict, extra: int) -> dict:
        batch, ops, resources = static["eligible"].shape
        padded = dict(static)
        padded["proc_profiles"] = torch.cat((static["proc_profiles"], torch.zeros(batch, extra, resources, static["proc_profiles"].shape[-1])), 1)
        padded["proc_ref"] = torch.cat((static["proc_ref"], torch.zeros(batch, extra, resources)), 1)
        padded["eligible"] = torch.cat((static["eligible"], torch.zeros(batch, extra, resources, dtype=torch.bool)), 1)
        size = ops + extra
        precedence = torch.zeros(batch, size, size, dtype=torch.bool)
        precedence[:, :ops, :ops] = static["precedence"]
        padded["precedence"] = precedence
        padded["pad_mask"] = torch.cat((static["pad_mask"], torch.ones(batch, extra, dtype=torch.bool)), 1)
        padded["op_features"] = torch.cat((static["op_features"], torch.ones(batch, extra, static["op_features"].shape[-1]) * 99), 1)
        return padded

    def test_padding_changes_nothing_and_stays_zero(self):
        env = TDFJSPEnv(variant_config())
        model = build_model(env).eval()
        builder = ObservationBuilder(env)
        static = builder.static_inputs(env.load_instances(instances()))
        with torch.no_grad():
            h_op, h_ma = model.encode(static)
            h_op_padded, h_ma_padded = model.encode(self._padded(static, 5))
        self.assertTrue(torch.allclose(h_op, h_op_padded[:, :50], atol=1e-5))
        self.assertTrue(torch.allclose(h_ma, h_ma_padded, atol=1e-5))
        self.assertEqual(float(h_op_padded[:, 50:].abs().sum()), 0.0)

    def test_isolated_nodes_stay_finite_with_finite_gradients(self):
        env = TDFJSPEnv(variant_config())
        model = build_model(env)
        builder = ObservationBuilder(env)
        static = builder.static_inputs(env.load_instances(instances()))
        static["eligible"] = static["eligible"].clone()
        static["eligible"][:, 0, :] = False     # Operation ohne Maschine
        static["eligible"][:, :, 2] = False     # Maschine ohne Operation
        h_op, h_ma = model.encode(static)
        self.assertTrue(bool(torch.isfinite(h_op).all()) and bool(torch.isfinite(h_ma).all()))
        (h_op.sum() + h_ma.sum()).backward()
        for parameter in model.encoder.parameters():
            if parameter.grad is not None:
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))


class GradientAndProfileTest(unittest.TestCase):
    def test_gradients_reach_attention_and_edge_value_mlps(self):
        env = TDFJSPEnv(variant_config("A3"))
        model = build_model(env)
        builder = ObservationBuilder(env)
        td = env.load_instances(instances())
        static, dynamic = builder.static_inputs(td), builder.dynamic_inputs(td)
        logits, value = model(*model.encode(static), static["pad_mask"], dynamic)
        (logits[dynamic["action_mask"]].sum() + value.sum()).backward()
        for layer in model.encoder.layers:
            for direction in (layer.machines_to_ops, layer.ops_to_machines):
                for mlp in (direction.attention_mlp, direction.edge_value_mlp):
                    size = sum(float(p.grad.abs().sum()) for p in mlp.parameters())
                    self.assertGreater(size, 0.0)

    def test_a2_has_no_edge_value_mlp(self):
        model = build_model(TDFJSPEnv(variant_config("A2")))
        self.assertIsNone(model.encoder.layers[0].machines_to_ops.edge_value_mlp)

    def test_profile_change_reaches_temporal_encoders_only(self):
        for variant, changes in (("A0", False), ("A1", False), ("A2", True), ("A3", True)):
            env = TDFJSPEnv(variant_config(variant))
            model = build_model(env).eval()
            static = ObservationBuilder(env).static_inputs(env.load_instances(instances()))
            changed = dict(static)
            changed["proc_profiles"] = static["proc_profiles"] * 1.5
            with torch.no_grad():
                difference = float((model.encode(static)[0] - model.encode(changed)[0]).abs().max())
            self.assertEqual(difference > 1e-6, changes, variant)


class PermutationTest(unittest.TestCase):
    def test_renaming_machines_only_permutes_outputs(self):
        env = TDFJSPEnv(variant_config("A3"))
        model = build_model(env).eval()
        builder = ObservationBuilder(env)
        td = env.load_instances(instances())
        static, dynamic = builder.static_inputs(td), builder.dynamic_inputs(td)
        J, R = 10, 6
        order = torch.tensor([3, 0, 5, 1, 4, 2])

        moved = dict(static)
        for key in ("proc_profiles", "proc_ref", "eligible"):
            moved[key] = static[key][:, :, order]
        moved["machine_features"] = static["machine_features"][:, order]
        moved_dynamic = dict(dynamic)
        moved_dynamic["candidates"] = dynamic["candidates"][:, :, order]
        moved_dynamic["candidate_valid"] = dynamic["candidate_valid"][:, :, order]
        pair_mask = dynamic["action_mask"][:, 1:].view(-1, J, R)[:, :, order]
        moved_dynamic["action_mask"] = torch.cat((dynamic["action_mask"][:, :1], pair_mask.flatten(1)), 1)

        with torch.no_grad():
            h_op, h_ma = model.encode(static)
            h_op_moved, h_ma_moved = model.encode(moved)
            logits, value = model(h_op, h_ma, static["pad_mask"], dynamic)
            logits_moved, value_moved = model(h_op_moved, h_ma_moved, static["pad_mask"], moved_dynamic)
        self.assertTrue(torch.allclose(h_op, h_op_moved, atol=1e-4))
        self.assertTrue(torch.allclose(h_ma[:, order], h_ma_moved, atol=1e-4))
        pairs = logits[:, 1:].view(-1, J, R)[:, :, order].flatten(1)
        self.assertTrue(torch.allclose(pairs, logits_moved[:, 1:], atol=1e-3))
        self.assertTrue(torch.allclose(logits[:, 0], logits_moved[:, 0], atol=1e-3))
        self.assertTrue(torch.allclose(value, value_moved, atol=1e-4))


class StartTimeTest(unittest.TestCase):
    """Ruestung und Bearbeitung muessen an ihren tatsaechlichen Startzeiten bewertet werden."""

    def test_profiles_match_validator_duration_at_window_starts(self):
        model = TimeModel(CONFIG)
        instance = {"days": 2}
        kinds = torch.tensor([KIND_IDS["manual"], KIND_IDS["automatic"]])
        for base in (1, 7, 25, 40, 60):
            profile = model.window_profile(kinds, torch.tensor([base, base]))
            for k, start in enumerate(model.window_starts.tolist()):
                for row, kind in enumerate(("manual", "automatic")):
                    try:
                        expected = duration(CONFIG, instance, kind, base, start)
                    except ValueError:
                        expected = 0
                    self.assertEqual(int(profile[row, k]), expected, (kind, base, start))

    def test_first_feasible_matches_simulator_on_full_grid(self):
        model = TimeModel(CONFIG)
        instance = {"days": 2}
        limit = 1440
        for kind in ("manual", "automatic"):
            for base in (20, 37, 60, 200):
                for setup in (0, 5, 13, 40):
                    times = torch.arange(0, limit)
                    kinds = torch.full_like(times, KIND_IDS[kind])
                    start, setup_len, proc_len, found = first_feasible(
                        model, kinds, torch.full_like(times, base), torch.full_like(times, setup),
                        times, torch.full_like(times, limit))
                    for t in range(0, limit, 7):
                        expected = first_feasible_start(CONFIG, instance, kind, base, setup, t)
                        if expected is None:
                            self.assertFalse(bool(found[t]), (kind, base, setup, t))
                        else:
                            self.assertEqual((int(start[t]), int(setup_len[t]), int(proc_len[t])),
                                             expected, (kind, base, setup, t))

    def test_candidates_match_simulator_in_real_states(self):
        env = TDFJSPEnv(variant_config())
        builder = ObservationBuilder(env)
        builder.static_inputs(env.load_instances(instances("test_klein", 1)))
        instance = instances("test_klein", 1, 5100)[0]
        resources = all_resources(instance)
        with_setup = set(setup_resources(CONFIG, instance))
        setups = setup_lookup(instance)
        op_ids = [op for job in instance["jobs"] for op in job["operations"]]
        operations = {o["id"]: o for o in instance["operations"]}
        extended = execution_instance(CONFIG, instance)
        td = env.load_instances([instance])
        builder.static_inputs(td)
        checked = 0
        while not td["done"].all():
            data = builder.dynamic_inputs(td)
            now = int(td["time"][0])
            for job, resource in data["candidate_valid"][0].nonzero().tolist():
                op = operations[op_ids[int(td["next_op"][0, job])]]
                name = resources[resource]
                previous = int(td["last_op"][0, resource])
                setup = 0
                if name in with_setup and previous >= 0:
                    setup = setups[(name, op_ids[previous], op["id"])]
                earliest = max(now, int(td["job_ready"][0, job]), int(td["busy_until"][0, resource]))
                expected = first_feasible_start(CONFIG, extended, op["kind"], op["base_processing"][name],
                                                setup, earliest)
                row = data["candidates"][0, job, resource]
                if expected is None:
                    continue
                start, setup_len, processing = expected
                self.assertAlmostEqual(float(row[FEATURE["setup_start_offset"]]) * 60, start - now, places=2)
                self.assertAlmostEqual(float(row[FEATURE["setup_time"]]) * 60, setup_len, places=2)
                self.assertAlmostEqual(float(row[FEATURE["processing_start_offset"]]) * 60,
                                       start + setup_len - now, places=2)
                self.assertAlmostEqual(float(row[FEATURE["processing_time"]]) * 60, processing, places=2)
                checked += 1
                if bool(td["pair_mask"][0, job, resource]):
                    self.assertEqual(int(td["opt_setup"][0, job, resource]), setup_len)
                    self.assertEqual(int(td["opt_proc"][0, job, resource]), processing)
                    self.assertEqual(start, now)
            td["action"] = edd_ect(td)
            td = env.step(td)["next"]
        self.assertGreater(checked, 500)

    def test_manual_operation_before_break_moves_behind_the_break(self):
        model = TimeModel(CONFIG)
        one = torch.tensor([KIND_IDS["manual"]])
        limit = torch.tensor([1440])
        # 11:45, 25 Minuten mal Faktor 1,2 = 30 Minuten: wuerde in die Pause laufen
        start, _, processing, found = first_feasible(model, one, torch.tensor([25]), torch.tensor([0]),
                                                     torch.tensor([345]), limit)
        self.assertTrue(bool(found[0]))
        self.assertEqual(int(start[0]), 390)    # 12:30
        self.assertEqual(int(processing[0]), 40)    # dort Faktor 1,6


class TerminationAndFeasibilityTest(unittest.TestCase):
    def test_waiting_always_moves_time_and_ends(self):
        env = TDFJSPEnv(variant_config())
        td = env.load_instances(instances("mini", 2))
        steps = 0
        while not td["done"].all():
            before = td["time"].clone()
            waiting = td["action_mask"][:, 0] & ~td["done"]
            td["action"] = torch.where(td["action_mask"][:, 0], 0, edd_ect(td))
            td = env.step(td)["next"]
            self.assertTrue(bool((td["time"][waiting] > before[waiting]).all()))
            steps += 1
            self.assertLess(steps, 2000)

    def test_wait_is_available_while_operations_could_start(self):
        env = TDFJSPEnv(variant_config())
        td = env.load_instances(instances("mini", 1))
        self.assertTrue(bool(td["pair_mask"].any()))
        self.assertTrue(bool(td["action_mask"][0, 0]))
        without = TDFJSPEnv(variant_config(allow_wait=False)).load_instances(instances("mini", 1))
        self.assertFalse(bool(without["action_mask"][0, 0]))

    def test_all_variants_produce_valid_schedules(self):
        for variant in VARIANTS:
            for time_information in ("full", "none"):
                env = TDFJSPEnv(variant_config(variant, time_information, small=True))
                chosen = instances("mini", 4, 5200)
                policy = NeuralPolicy(env, build_model(env), "sample", seed=1)
                td, _ = rollout(env, env.load_instances(chosen), policy)
                self.assertTrue(bool(td["done"].all()))
                env.check_solution_validity(td, chosen)

    def test_evaluation_keeps_voluntary_waiting(self):
        env = TDFJSPEnv(variant_config(small=True))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "instance_000.json"
            save_instance(instances("mini", 1, 5300)[0], path)
            first = {"done": False}

            def waits_once_then_rule(td):
                if not first["done"] and bool(td["action_mask"][0, 0]):
                    first["done"] = True
                    return torch.zeros(1, dtype=torch.long)
                return edd_ect(td)

            row = evaluate_policy(env, waits_once_then_rule, "wartet", "mini", [path])[0]
            self.assertTrue(row["feasible"], row["status"])
            self.assertEqual(row["makespan"], row["planned_makespan"])


class TimeInformationTest(unittest.TestCase):
    """model.time_information = none: Der Agent sieht keine Faktoren, die Umgebung bleibt echt."""

    def _changed_factors(self, time_information: str) -> dict:
        config = variant_config("A3", time_information)
        for window in config["time_model"]["day_windows"]:
            if window["factor"] is not None and window["factor"] > 1:
                window["factor"] = 1.9
        return config

    def test_static_inputs_ignore_factors_without_time_information(self):
        chosen = instances()
        for time_information, same in (("none", True), ("full", False)):
            first = TDFJSPEnv(variant_config("A3", time_information))
            second = TDFJSPEnv(self._changed_factors(time_information))
            a = ObservationBuilder(first).static_inputs(first.load_instances(chosen))["proc_profiles"]
            b = ObservationBuilder(second).static_inputs(second.load_instances(chosen))["proc_profiles"]
            self.assertEqual(torch.equal(a, b), same, time_information)

    def test_agent_sees_base_durations_while_environment_stays_real(self):
        env = TDFJSPEnv(variant_config("A3", "none"))
        builder = ObservationBuilder(env)
        td = env.load_instances(instances("test_klein", 1, 5400))
        static = builder.static_inputs(td)
        td = play_until(env, td, 420)       # 13:00, Faktor 1,6
        self.assertTrue(set(static["proc_profiles"].unique().tolist()) <= {0.0, 1.0})
        data = builder.dynamic_inputs(td)
        self.assertEqual(float(data["time_features"].abs().sum()), 0.0)
        self.assertEqual(float(static["op_features"][..., 8].abs().sum()), 0.0)
        manual = td["op_kind"].gather(1, td["next_op"]) == KIND_IDS["manual"]
        pairs = td["pair_mask"] & manual.unsqueeze(2)
        if pairs.any():
            shown = data["candidates"][..., FEATURE["processing_time"]][pairs] * 60
            base = td["opt_base"][pairs].to(torch.float32)
            self.assertTrue(torch.allclose(shown, base, atol=1e-2))
            self.assertTrue(bool((td["opt_proc"][pairs] > td["opt_base"][pairs]).any()))

    def test_full_information_shows_real_duration_at_start(self):
        env = TDFJSPEnv(variant_config("A3", "full"))
        builder = ObservationBuilder(env)
        td = env.load_instances(instances("test_klein", 1, 5400))
        builder.static_inputs(td)
        td = play_until(env, td, 420)
        data = builder.dynamic_inputs(td)
        self.assertGreater(float(data["time_features"].abs().sum()), 0.0)
        pairs = td["pair_mask"]
        if pairs.any():
            shown = data["candidates"][..., FEATURE["processing_time"]][pairs] * 60
            self.assertTrue(torch.allclose(shown, td["opt_proc"][pairs].to(torch.float32), atol=1e-2))


class PPOTest(unittest.TestCase):
    def test_returns_without_discount_sum_all_later_rewards(self):
        rewards = torch.tensor([[1.0, 2.0], [3.0, 0.0], [5.0, 0.0]])
        alive = torch.tensor([[True, True], [True, False], [True, False]])
        values = torch.tensor([[0.5, 0.1], [0.2, 0.0], [0.3, 0.0]])
        _, returns = compute_returns(rewards, values, alive, gamma=1.0, gae_lambda=1.0)
        self.assertTrue(torch.allclose(returns[alive], torch.tensor([9.0, 2.0, 8.0, 5.0])))

    def test_one_update_changes_parameters_and_stays_finite(self):
        torch.manual_seed(0)
        env = TDFJSPEnv(variant_config(small=True))
        model = build_model(env)
        builder = ObservationBuilder(env)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = [p.detach().clone() for p in model.parameters()]
        td = env.load_instances(instances("mini", 4))
        data = collect_rollout(env, model, builder, td, CONFIG["training"])
        result = ppo_update(model, optimizer, data, {**CONFIG["training"], "minibatch_size": 16, "ppo_epochs": 1})
        self.assertTrue(all(math.isfinite(v) for v in result.values()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
        self.assertTrue(bool(data.final_state["done"].all()))

    def test_training_is_reproducible_and_checkpoint_loads(self):
        with tempfile.TemporaryDirectory() as folder:
            config = variant_config(small=True)
            config["data_splits"]["val"]["folder"] = str(Path(folder) / "val" / "{class}")
            config["training"].update(output_folder=str(Path(folder) / "runs"), instances_per_iteration=3,
                                      validate_every=1, validation_instances=2, minibatch_size=32, ppo_epochs=1)
            for i in range(2):
                save_instance(instances("mini", 2, 5500)[i], Path(folder) / "val" / "mini" / f"instance_{i:03d}.json")

            def log_without_time(output: Path) -> list[dict]:
                with (output / "training_log.csv").open(encoding="utf-8") as file:
                    return [{k: v for k, v in row.items() if k != "seconds"} for row in csv.DictReader(file)]

            first = log_without_time(train(copy.deepcopy(config), 7, 2, Path(folder) / "problem_config.json"))
            output = train(copy.deepcopy(config), 7, 2, Path(folder) / "problem_config.json")
            self.assertEqual(first, log_without_time(output))
            self.assertEqual(output.name, "A3_zeit_warten_ppo_seed7_r2")

            env, model, name = load_checkpoint(output / "best.pt", Path(folder) / "problem_config.json")
            self.assertEqual(name, "A3_zeit_warten_ppo_seed7_r2")
            chosen = instances("mini", 2, 5600)
            td, _ = rollout(env, env.load_instances(chosen), NeuralPolicy(env, model))
            env.check_solution_validity(td, chosen)


class SupervisedLearningTest(unittest.TestCase):
    """Lernfaehigkeit ohne PPO; die ganze Tabelle zeigt python -m tests.synthetic_learning."""

    def test_temporal_encoder_learns_time_dependent_choice(self):
        self.assertGreaterEqual(train_and_measure("A3", "two_phase", candidates_visible=False), 0.95)

    def test_static_encoder_cannot_without_duration_in_decoder(self):
        self.assertLess(train_and_measure("A1", "two_phase", candidates_visible=False), 0.65)

    def test_decoder_learns_from_duration_at_actual_start(self):
        self.assertGreaterEqual(train_and_measure("A1", "two_phase", candidates_visible=True), 0.95)


if __name__ == "__main__":
    unittest.main()
