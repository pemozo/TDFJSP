"""Abnahmetests der RL-Umgebung gegen die unabhaengigen Module des Projekts.

Aufruf aus dem Projektordner:
    python -m unittest tests.test_rl_env                  schnelle Tests (ca. 1 Minute)
    RL_FULL_TEST=1 python -m unittest tests.test_rl_env   zusaetzlich alle Testinstanzen

In PowerShell fuer den vollen Test: $env:RL_FULL_TEST = "1"; python -m unittest tests.test_rl_env
Auf einem Rechner mit GPU laeuft zusaetzlich der Vergleich CPU gegen GPU.
"""
from __future__ import annotations

import copy
import math
import os
import random
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from baselines.dispatcher import build_plan
from evaluation import evaluate
from generator import generate_class_instance, generate_class_instances, save_instance
from rl.env import FEATURE_NAMES, TDFJSPEnv, resolve_device
from rl.evaluate import evaluate_policy
from rl.generators import TDFJSPFileGenerator, TDFJSPGenerator
from rl.parser import instances_to_td
from rl.policies import edd_ect, edd_spt, make_random_policy, rollout
from rl.time_model import KIND_IDS, TimeModel
from run_experiments import run_instance
from schedule_validator import (KINDS, all_resources, check_config, check_schedule, duration,
                                horizon, instance_files, load_config, selected_classes, setup_resources,
                                split_folder)
from simulator import (SimulationState, execute_plan, execution_instance, make_entry,
                       plan_from_schedule, view_config)

CONFIG = load_config()
PROJECT_CLASSES = selected_classes(CONFIG, "all")
# Die Tests legen diese Einstellungen selbst fest, damit sie nicht von der aktuellen Config abhaengen
CONFIG["environment"].update(device="cpu", allow_wait=False, view="time_aware", train_classes="all",
                             generator_workers=0, compute_features=True)
CONFIG["reward"].update(levels=["total_tardiness", "makespan"], level_weights=None, mode="stepwise")
CONFIG["simulation"]["overflow_days"] = 1
CONFIG["instance_classes"].update(
    mini={"jobs": 4, "operations_per_job": 3, "workers": 1, "machines": 2, "days": 1},
    test_klein={"jobs": 10, "operations_per_job": 5, "workers": 2, "machines": 4, "days": 1},
    test_mittel={"jobs": 15, "operations_per_job": 5, "workers": 3, "machines": 6, "days": 1},
    test_voll={"jobs": 14, "operations_per_job": 5, "workers": 2, "machines": 4, "days": 1})
TEST_CLASSES = ("mini", "test_klein", "test_mittel")


def variant(**sections) -> dict:
    """Kopie der Konfiguration; je Abschnitt werden die genannten Werte ersetzt."""
    config = copy.deepcopy(CONFIG)
    for section, values in sections.items():
        config[section].update(values)
    return config


def instance_of(class_name: str, seed: int, config: dict = CONFIG) -> dict:
    return generate_class_instance(config, class_name, seed)


def env_options(env: TDFJSPEnv, td: dict, instance: dict) -> list[tuple[int, dict]]:
    """Alle Paare der Maske einer Einzelinstanz als (Aktion, Schedule-Eintrag)."""
    op_ids = [op_id for job in instance["jobs"] for op_id in job["operations"]]
    resources = all_resources(instance)
    with_setup = set(setup_resources(env.config, instance))
    now = int(td["time"][0])
    result = []
    for job, resource in td["pair_mask"][0].nonzero().tolist():
        previous = int(td["last_op"][0, resource])
        name = resources[resource]
        setup = (None if previous < 0 else op_ids[previous], 0) if name in with_setup else None
        entry = make_entry(op_ids[int(td["next_op"][0, job])], name, setup, now,
                           int(td["opt_setup"][0, job, resource]),
                           int(td["opt_proc"][0, job, resource]))
        result.append((1 + job * len(resources) + resource, entry))
    return result


def key(entry: dict) -> tuple:
    return tuple(sorted(entry.items(), key=lambda item: item[0]))


class TimeModelTest(unittest.TestCase):
    """Die Tensor-Formel muss in jedem Fall dasselbe liefern wie schedule_validator.duration."""

    def assert_same_as_validator(self, config: dict, view: str = "time_aware") -> None:
        reference = view_config(config, view)
        model = TimeModel(config, view)
        instance = {"days": 2}
        limit = horizon(reference, instance)
        bases = torch.arange(1, 101).unsqueeze(1).expand(-1, limit)
        starts = torch.arange(limit).expand(100, -1)
        for kind in KINDS:
            lengths, allowed = model.duration(KIND_IDS[kind], bases, starts, limit)
            for b in range(100):
                for t in range(limit):
                    try:
                        expected = duration(reference, instance, kind, b + 1, t)
                    except ValueError:
                        expected = None
                    if bool(allowed[b, t]) != (expected is not None) or (
                            expected is not None and int(lengths[b, t]) != expected):
                        self.fail(f"{kind} Basis {b + 1} Start {t}: "
                                  f"{int(lengths[b, t])}/{bool(allowed[b, t])} statt {expected}")

    def test_duration_matches_validator(self):
        self.assert_same_as_validator(CONFIG)

    def test_duration_matches_validator_in_static_view(self):
        self.assert_same_as_validator(CONFIG, "static")

    def test_duration_follows_changed_rules(self):
        config = copy.deepcopy(CONFIG)
        config["rules"]["allowed_overnight"]["automatic"] = True
        config["rules"]["allowed_in_break"]["manual"] = True
        config["rules"]["time_dependent"]["automatic"] = True
        config["time_model"]["day_windows"][1]["factor"] = 1.37
        self.assert_same_as_validator(config)

    def test_boundaries_match_simulator(self):
        instance = instance_of("mini", 42)
        state = SimulationState(CONFIG, instance)
        model = TimeModel(CONFIG)
        limit = horizon(CONFIG, execution_instance(CONFIG, instance))
        times = torch.arange(limit)
        boundary, exists = model.next_boundary(times, torch.full_like(times, limit))
        for t in range(limit):
            later = [b for b in state._boundaries if b > t]
            self.assertEqual(bool(exists[t]), bool(later))
            if later:
                self.assertEqual(int(boundary[t]), later[0])

    def test_finish_after_is_first_feasible_start_plus_duration(self):
        model = TimeModel(CONFIG)
        model.ensure_finish_table(80)
        instance = {"days": 3}
        for kind in KINDS:
            for base in (1, 5, 37, 80):
                for t in range(0, 720, 7):
                    start = t
                    while True:
                        try:
                            length = duration(CONFIG, instance, kind, base, start)
                            break
                        except ValueError:
                            start += 1
                    found = model.finish_after(KIND_IDS[kind], torch.tensor(base), torch.tensor(t))
                    self.assertEqual(int(found), start + length, (kind, base, t))


class SimulatorEquivalenceTest(unittest.TestCase):
    """Die Umgebung muss in jedem Zustand dieselben Optionen und Uhrzeiten haben wie SimulationState."""

    def play_against_simulator(self, config: dict, instance: dict, seed: int, wait_share: float):
        env = TDFJSPEnv(config)
        state = SimulationState(config, instance, env.view)
        td = env.load_instances([instance])
        rng = random.Random(seed)

        def settle() -> bool:
            """Simulator wie die Umgebung vorstellen. False = Horizont reicht nicht."""
            while not state.done() and not state.candidate_operations():
                if not state.advance():
                    return False
            return True

        running = settle()
        while running and not state.done():
            self.assertEqual(int(td["time"][0]), state.time)
            options = env_options(env, td, instance)
            expected = [e for op in state.candidate_operations() for e in state.options(op["id"])]
            self.assertEqual(sorted(key(e) for _, e in options), sorted(key(e) for e in expected))

            if wait_share and rng.random() < wait_share and bool(td["action_mask"][0, 0]):
                self.assertTrue(state.advance())
                action = 0
            else:
                action, entry = rng.choice(options)
                state.apply(entry)
            td["action"] = torch.tensor([action])
            td = env.step(td)["next"]
            running = settle()

        self.assertEqual(bool(td["truncated"][0]), not running)
        self.assertTrue(bool(td["done"][0]))
        self.assertEqual(env.to_schedule(td, 0, instance), state.schedule)
        return td, env

    def test_random_play_matches_options(self):
        for class_name in TEST_CLASSES:
            for seed in range(3):
                self.play_against_simulator(CONFIG, instance_of(class_name, 200 + seed), seed, 0.0)

    def test_random_play_with_waiting(self):
        config = variant(environment={"allow_wait": True})
        for class_name in TEST_CLASSES:
            for seed in range(3):
                self.play_against_simulator(config, instance_of(class_name, 300 + seed), seed, 0.2)

    def test_static_view_matches_options(self):
        config = variant(environment={"view": "static"})
        for seed in range(3):
            self.play_against_simulator(config, instance_of("test_klein", 400 + seed), seed, 0.0)

    def test_rules_replay_exactly(self):
        for class_name in TEST_CLASSES:
            for seed in range(4):
                instance = instance_of(class_name, 500 + seed)
                self.assert_replay(instance)

    def assert_replay(self, instance: dict) -> None:
        for view in ("time_aware", "static"):
            env = TDFJSPEnv(variant(environment={"view": view}))
            for routing, policy in (("ECT", edd_ect), ("SPT", edd_spt)):
                td, _ = rollout(env, env.load_instances([instance]), policy)
                expected = build_plan(env.config, instance, "EDD", routing, view)
                self.assertEqual(env.to_schedule(td, 0, instance), expected, (view, routing))

    @unittest.skipUnless(os.environ.get("RL_FULL_TEST"), "nur mit RL_FULL_TEST=1")
    def test_rules_replay_on_all_test_instances(self):
        from schedule_validator import load_json

        for class_name in PROJECT_CLASSES:
            files = instance_files([split_folder(CONFIG, class_name, "test")])
            self.assertTrue(files, class_name)
            for path in files:
                self.assert_replay(load_json(path))


class SolutionTest(unittest.TestCase):
    """Plaene der Umgebung muessen den unabhaengigen Validator bestehen."""

    def test_random_plans_are_valid_and_objectives_match_evaluation(self):
        config = variant(reward={"levels": ["late_jobs", "total_tardiness", "makespan"]})
        env = TDFJSPEnv(config)
        for class_name in ("mini", "test_klein"):
            instances = [instance_of(class_name, 600 + i) for i in range(4)]
            td, _ = rollout(env, env.load_instances(instances), make_random_policy(1))
            env.check_solution_validity(td, instances)
            values = env.get_objectives(td)
            for i, instance in enumerate(instances):
                self.assertFalse(bool(td["truncated"][i]))
                result = evaluate(config, execution_instance(config, instance),
                                  env.to_schedule(td, i, instance))
                self.assertTrue(result["feasible"], result["violations"])
                for level in ("total_tardiness", "late_jobs", "max_tardiness", "makespan"):
                    self.assertEqual(int(values[level][i]), result[level], level)

    def test_static_plan_can_be_executed_in_reality(self):
        env = TDFJSPEnv(variant(environment={"view": "static"}))
        instance = instance_of("test_klein", 610)
        td, _ = rollout(env, env.load_instances([instance]), edd_ect)
        executed = execute_plan(CONFIG, instance, plan_from_schedule(env.to_schedule(td, 0, instance)))
        self.assertEqual(check_schedule(CONFIG, execution_instance(CONFIG, instance), executed), [])

    def test_forbidden_action_is_rejected(self):
        env = TDFJSPEnv(CONFIG)
        td = env.load_instances([instance_of("mini", 620)])
        forbidden = (~td["action_mask"][0]).nonzero()[0].item()
        td["action"] = torch.tensor([forbidden])
        with self.assertRaises(ValueError):
            env.step(td)

    def test_waiting_too_long_ends_with_truncation_and_penalty(self):
        config = variant(environment={"allow_wait": True}, reward={"mode": "terminal"},
                         simulation={"overflow_days": 0})
        env = TDFJSPEnv(config)
        td = env.load_instances([instance_of("mini", 630, config)])
        steps = 0
        while not td["done"].all():
            # Warten ist nur erlaubt, solange noch ein Ereignis kommt; sonst muss etwas starten
            td["action"] = torch.where(td["action_mask"][:, 0], 0, edd_ect(td))
            td = env.step(td)["next"]
            steps += 1
            self.assertLess(steps, 1000)
        self.assertTrue(bool(td["truncated"][0]))
        values = env.get_objectives(td)
        open_ops = int((~td["op_scheduled"]).sum())
        self.assertGreater(open_ops, 0)
        self.assertGreaterEqual(int(values["total_tardiness"][0]), open_ops * 720)
        self.assertLessEqual(int(values["makespan"][0]), int(td["horizon"][0]))

    def test_step_keeps_earlier_states_unchanged(self):
        env = TDFJSPEnv(CONFIG)
        td = env.load_instances([instance_of("mini", 640), instance_of("mini", 641)])
        before = {k: v.clone() for k, v in td.items()}
        td["action"] = edd_ect(td)
        following = env.step(td)["next"]
        for k, value in before.items():
            self.assertTrue(torch.equal(td[k], value), k)
        self.assertIs(following["setup_times"], td["setup_times"])  # feste Daten nicht kopiert
        self.assertGreater(int(following["num_scheduled"].sum()), 0)


class RewardTest(unittest.TestCase):
    def test_stepwise_rewards_sum_to_true_cost_difference(self):
        env = TDFJSPEnv(CONFIG)
        instances = [instance_of("test_klein", 700 + i) for i in range(5)]
        td, total = rollout(env, env.load_instances(instances), make_random_policy(3))
        norm = env.cost.normalizer(td)
        expected = (td["initial_cost_estimate"] - env.cost.cost(td, env._completion(td))) / norm
        self.assertTrue(torch.allclose(total, expected, rtol=0, atol=1e-9))

    def test_terminal_reward_arrives_only_at_the_end(self):
        env = TDFJSPEnv(variant(reward={"mode": "terminal"}))
        td = env.load_instances([instance_of("mini", 710), instance_of("mini", 711)])
        seen = torch.zeros(2, dtype=torch.float64)
        while not td["done"].all():
            before = td["done"].clone()
            td["action"] = edd_ect(td)
            td = env.step(td)["next"]
            newly = td["done"] & ~before
            self.assertTrue(bool((td["reward"][~newly] == 0).all()))
            seen = seen + td["reward"]
        self.assertTrue(torch.allclose(seen, env.get_reward(td), rtol=0, atol=1e-12))

    def assert_lexicographic(self, levels: list[str]) -> None:
        env = TDFJSPEnv(variant(reward={"levels": levels, "mode": "terminal"}))
        rng = random.Random(1)
        td, _ = rollout(env, env.load_instances([instance_of("mini", 720)]), edd_ect)
        weights = env.cost.weights(td)[0]
        bounds = env.cost._upper_bounds(td)

        def cost(values):
            return sum(float(v) * float(w) for v, w in zip(values, weights))

        for _ in range(3000):
            a = [rng.randint(0, int(bounds[level][0])) for level in levels]
            b = [rng.randint(0, int(bounds[level][0])) for level in levels]
            # auch nahe beieinander: oberste Ebene gleich, untere frei
            if rng.random() < 0.3:
                b[0] = a[0]
            order = (a > b) - (a < b)
            difference = cost(a) - cost(b)
            self.assertEqual(order, (difference > 0) - (difference < 0), (a, b))

    def test_hierarchy_is_exactly_lexicographic(self):
        self.assert_lexicographic(["total_tardiness", "makespan"])
        self.assert_lexicographic(["late_jobs", "total_tardiness", "makespan"])

    def test_one_minute_tardiness_outweighs_every_makespan(self):
        env = TDFJSPEnv(CONFIG)
        td, _ = rollout(env, env.load_instances([instance_of("mini", 730)]), edd_ect)
        tardiness_weight, makespan_weight = env.cost.weights(td)[0].tolist()
        self.assertGreater(tardiness_weight, int(td["horizon"][0]) * makespan_weight)

    def test_fixed_weights_replace_hierarchy(self):
        env = TDFJSPEnv(variant(reward={"level_weights": [60, 1]}))
        td, _ = rollout(env, env.load_instances([instance_of("mini", 740)]), edd_ect)
        values = env.get_objectives(td)
        expected = 60 * float(values["total_tardiness"][0]) + float(values["makespan"][0])
        self.assertAlmostEqual(float(env.cost.cost(td, env._completion(td))[0]), expected)
        self.assertAlmostEqual(float(env.get_reward(td)[0]), -expected / (60 * 4 * 60))


class BatchTest(unittest.TestCase):
    def test_batch_equals_single_runs(self):
        env = TDFJSPEnv(CONFIG)
        instances = [instance_of("test_klein", 800 + i) for i in range(4)]
        td, total = rollout(env, env.load_instances(instances), edd_ect)
        for i, instance in enumerate(instances):
            alone, alone_total = rollout(env, env.load_instances([instance]), edd_ect)
            self.assertEqual(env.to_schedule(td, i, instance), env.to_schedule(alone, 0, instance))
            self.assertAlmostEqual(float(total[i]), float(alone_total[0]), places=9)
            for name in FEATURE_NAMES:
                self.assertEqual(td[name].shape[2:], alone[name].shape[2:], name)

    def test_mixed_sizes_are_rejected(self):
        with self.assertRaises(ValueError):
            instances_to_td(CONFIG, [instance_of("mini", 810), instance_of("test_klein", 811)])

    def test_episode_finished_earlier_stays_unchanged(self):
        env = TDFJSPEnv(CONFIG)
        td = env.load_instances([instance_of("mini", 820), instance_of("mini", 821)])
        while not td["done"].all():
            frozen = {k: td[k].clone() for k in ("time", "op_end")}
            was_done = td["done"].clone()
            td["action"] = edd_ect(td)
            td = env.step(td)["next"]
            for k, value in frozen.items():
                self.assertTrue(torch.equal(td[k][was_done], value[was_done]), k)
            self.assertTrue(bool((td["reward"][was_done] == 0).all()))


class FeatureTest(unittest.TestCase):
    def test_shapes_and_values_stay_valid_during_an_episode(self):
        env = TDFJSPEnv(CONFIG)
        instance = instance_of("test_klein", 900)
        td = env.load_instances([instance, instance_of("test_klein", 901)])
        jobs, resources, ops = 10, 6, 50
        while True:
            self.assertEqual(td["op_features"].shape, (2, ops, len(FEATURE_NAMES["op_features"])))
            self.assertEqual(td["resource_features"].shape,
                             (2, resources, len(FEATURE_NAMES["resource_features"])))
            self.assertEqual(td["pair_features"].shape,
                             (2, jobs, resources, len(FEATURE_NAMES["pair_features"])))
            self.assertEqual(td["time_features"].shape, (2, len(FEATURE_NAMES["time_features"])))
            for name in FEATURE_NAMES:
                self.assertTrue(bool(torch.isfinite(td[name]).all()), name)
                self.assertEqual(td[name].dtype, torch.float32, name)
            self.assertTrue(torch.equal(td["pair_features"][..., 4] > 0, td["pair_mask"]))
            if td["done"].all():
                break
            td["action"] = edd_ect(td)
            td = env.step(td)["next"]

    def test_clock_features_follow_the_config(self):
        env = TDFJSPEnv(CONFIG)
        horizon_minutes = torch.full((6,), 1440)
        times = torch.tensor([0, 200, 300, 370, 400, 650])
        features = env.time_model.time_features(times, horizon_minutes, 60)
        names = FEATURE_NAMES["time_features"]
        factor = features[:, names.index("factor_now")].tolist()
        self.assertEqual([round(f, 3) for f in factor], [1.0, 1.3, 1.2, 0.0, 1.6, 1.1])
        self.assertAlmostEqual(float(features[0, names.index("minutes_to_break")]), 6.0)
        self.assertAlmostEqual(float(features[3, names.index("minutes_to_break")]), 0.0)
        self.assertAlmostEqual(float(features[0, names.index("minutes_to_next_window")]), 3.0)
        self.assertAlmostEqual(float(features[5, names.index("minutes_to_day_end")]), 70 / 60, places=5)

    def test_static_view_hides_factors_but_not_the_break(self):
        env = TDFJSPEnv(variant(environment={"view": "static"}))
        times = torch.tensor([200, 370, 400])
        features = env.time_model.time_features(times, torch.full((3,), 1440), 60)
        factor = features[:, FEATURE_NAMES["time_features"].index("factor_now")].tolist()
        self.assertEqual(factor, [1.0, 0.0, 1.0])

    def test_features_can_be_switched_off(self):
        env = TDFJSPEnv(variant(environment={"compute_features": False},
                                reward={"mode": "terminal"}))
        td = env.load_instances([instance_of("mini", 910)])
        self.assertNotIn("op_features", td)
        td, _ = rollout(env, td, edd_ect)
        self.assertLess(float(env.get_reward(td)[0]), 0)


class GeneratorTest(unittest.TestCase):
    def test_same_config_gives_same_instances(self):
        first = TDFJSPGenerator(CONFIG, ["mini"]).sample_instances(3)
        second = TDFJSPGenerator(CONFIG, ["mini"]).sample_instances(3)
        self.assertEqual(first, second)

    def test_seeds_continue_and_do_not_touch_val_or_test(self):
        generator = TDFJSPGenerator(CONFIG, ["mini"])
        seeds = [i["seed"] for i in generator.sample_instances(2) + generator.sample_instances(2)]
        self.assertEqual(len(set(seeds)), 4)
        splits = CONFIG["data_splits"]
        self.assertGreaterEqual(min(seeds), splits["train"]["seed_start"])
        self.assertGreater(splits["train"]["seed_start"], splits["val"]["seed_start"])

    def test_classes_change_per_batch(self):
        generator = TDFJSPGenerator(CONFIG, ["mini", "test_klein"])
        sizes = [generator(2)["due"].shape[1] for _ in range(3)]
        self.assertEqual(sizes, [4, 10, 4])

    def test_all_configured_classes_are_used_by_default(self):
        self.assertEqual(TDFJSPGenerator(CONFIG).classes, selected_classes(CONFIG, "all"))

    def test_new_class_only_needs_the_config(self):
        config = copy.deepcopy(CONFIG)
        config["instance_classes"]["neu"] = {"jobs": 3, "operations_per_job": 2, "workers": 1,
                                             "machines": 1, "days": 2, "due_slack": 0.0}
        config["environment"]["train_classes"] = ["neu"]
        env = TDFJSPEnv(config)
        td = env.reset(batch_size=2)
        self.assertEqual(td["due"].shape, (2, 3))
        self.assertEqual(td["op_kind"].shape, (2, 6))
        self.assertEqual(td["horizon"].tolist(), [3 * 720, 3 * 720])

    def test_processes_do_not_change_the_result(self):
        config = variant(environment={"generator_workers": 2})
        parallel = TDFJSPGenerator(config, ["mini"])
        try:
            self.assertEqual(parallel.sample_instances(5),
                             TDFJSPGenerator(CONFIG, ["mini"]).sample_instances(5))
        finally:
            parallel.close()

    def test_seeds_without_solution_are_skipped_in_a_fixed_way(self):
        # 14 Jobs auf kleiner Klasse: viele Seeds passen nicht in den Tag
        instances, next_seed, skipped = generate_class_instances(CONFIG, "test_voll", 1000, 2)
        self.assertEqual(len(instances), 2)
        self.assertEqual(next_seed, 1000 + 2 + skipped)
        self.assertEqual(instances[1]["seed"], next_seed - 1)

    def test_impossible_class_stops_with_clear_error(self):
        config = copy.deepcopy(CONFIG)
        config["instance_classes"]["zu_gross"] = {"jobs": 60, "operations_per_job": 5, "workers": 1,
                                                  "machines": 1, "days": 1, "schedule_attempts": 1}
        config["generation"]["max_consecutive_seed_failures"] = 3
        with self.assertRaisesRegex(ValueError, "zu_gross"):
            generate_class_instances(config, "zu_gross", 1, 2)

    def test_file_generator_validates_and_serves_batches(self):
        with tempfile.TemporaryDirectory() as folder:
            config = copy.deepcopy(CONFIG)
            config["data_splits"]["val"]["folder"] = str(Path(folder) / "{class}")
            for i in range(3):
                save_instance(instance_of("mini", 950 + i), Path(folder) / "mini" / f"instance_{i:03d}.json")
            files = TDFJSPFileGenerator(config, "mini", "val")
            self.assertEqual(files(2)["instance_index"].tolist(), [0, 1])
            self.assertEqual(files(2)["instance_index"].tolist(), [2, 0])
            self.assertEqual(files()["seed"].tolist(), [950, 951, 952])

            env = TDFJSPEnv(config)
            loader = DataLoader(env.dataset(phase="val", class_name="mini"), batch_size=2)
            batch = next(iter(loader))
            td = env.reset(batch)
            td, _ = rollout(env, td, edd_ect)
            self.assertTrue(bool(td["done"].all()))

            changed = copy.deepcopy(config)
            changed["time_model"]["day_windows"][1]["factor"] = 1.5
            with self.assertRaises(ValueError):
                TDFJSPFileGenerator(changed, "mini", "val")

    def test_instance_files_finds_subfolders(self):
        with tempfile.TemporaryDirectory() as folder:
            save_instance(instance_of("mini", 955), Path(folder) / "a" / "instance_000.json")
            save_instance(instance_of("mini", 956), Path(folder) / "b" / "instance_000.json")
            self.assertEqual(len(instance_files([Path(folder)])), 2)

    def test_missing_files_give_a_hint(self):
        config = copy.deepcopy(CONFIG)
        config["data_splits"]["val"]["folder"] = "gibt/es/nicht/{class}"
        with self.assertRaisesRegex(FileNotFoundError, "generator.py"):
            TDFJSPFileGenerator(config, "mini", "val")


class ConfigTest(unittest.TestCase):
    def assert_rejected(self, change) -> None:
        config = copy.deepcopy(CONFIG)
        change(config)
        self.assertTrue(check_config(config))
        with self.assertRaises(ValueError):
            TDFJSPEnv(config)

    def test_invalid_settings_are_rejected(self):
        self.assert_rejected(lambda c: c["environment"].update(view="falsch"))
        self.assert_rejected(lambda c: c["environment"].update(device="mps"))
        self.assert_rejected(lambda c: c["environment"].update(train_classes=["gibt_es_nicht"]))
        self.assert_rejected(lambda c: c["reward"].update(levels=["makespan", "makespan"]))
        self.assert_rejected(lambda c: c["reward"].update(levels=["unbekannt"]))
        self.assert_rejected(lambda c: c["reward"].update(mode="falsch"))
        self.assert_rejected(lambda c: c["reward"].update(level_weights=[1]))
        self.assert_rejected(lambda c: c["simulation"].update(overflow_days=-1))
        self.assert_rejected(lambda c: c["instance_classes"]["mini"].update(jobs=0))
        self.assert_rejected(lambda c: c["data_splits"]["train"].update(seed_start=50))
        self.assert_rejected(lambda c: c["data_splits"]["val"].update(seed_start=50))

    def test_missing_section_is_reported(self):
        config = copy.deepcopy(CONFIG)
        del config["environment"]
        self.assertIn("Abschnitt fehlt: environment", check_config(config))

    def test_overflow_days_come_from_the_config(self):
        for days in (0, 1, 3):
            config = variant(simulation={"overflow_days": days})
            env = TDFJSPEnv(config)
            td = env.load_instances([instance_of("mini", 960, config)])
            self.assertEqual(int(td["horizon"][0]), (1 + days) * 720)
            self.assertEqual(SimulationState(config, instance_of("mini", 960, config)).instance["days"],
                             1 + days)

    def test_device_setting(self):
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))
        self.assertEqual(resolve_device("auto").type, "cuda" if torch.cuda.is_available() else "cpu")

    def test_changed_time_windows_reach_the_environment(self):
        config = copy.deepcopy(CONFIG)
        config["time_model"]["day_windows"][0]["factor"] = 2.0
        instance = instance_of("mini", 970, config)
        env = TDFJSPEnv(config)
        td = env.load_instances([instance])
        manual = td["op_kind"][0] == KIND_IDS["manual"]
        next_ops = td["next_op"][0]
        job = int(manual[next_ops].nonzero()[0])
        resource = int((td["opt_base"][0, job] > 0).nonzero()[0])
        base = int(td["opt_base"][0, job, resource])
        self.assertEqual(int(td["opt_proc"][0, job, resource]), math.ceil(base * 2.0))


class EvaluationPipelineTest(unittest.TestCase):
    def test_policy_rows_equal_rule_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for i in range(3):
                path = Path(folder) / f"instance_{i:03d}.json"
                save_instance(instance_of("test_klein", 980 + i), path)
                paths.append(path)
            env = TDFJSPEnv(CONFIG)
            for name, policy, rule in (("edd_ect", edd_ect, "EDD+ECT zeitbewusst"),
                                       ("edd_spt", edd_spt, "EDD+SPT zeitbewusst")):
                rows = evaluate_policy(env, policy, name, "test_klein", paths)
                for path, row in zip(paths, rows):
                    expected = next(r for r in run_instance(CONFIG, path, "test_klein")
                                    if r["method"] == rule)
                    for column in ("late_jobs", "total_tardiness", "max_tardiness", "makespan",
                                   "planned_late_jobs", "planned_makespan", "feasible", "status"):
                        self.assertEqual(row[column], expected[column], column)

    def test_truncated_episode_is_reported_as_not_executable(self):
        config = variant(environment={"allow_wait": True}, simulation={"overflow_days": 0})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "instance_000.json"
            save_instance(instance_of("mini", 990, config), path)
            env = TDFJSPEnv(config)
            rows = evaluate_policy(env, lambda td: torch.where(td["action_mask"][:, 0], 0, edd_ect(td)),
                                   "wartet", "mini", [path])
            self.assertFalse(rows[0]["feasible"])
            self.assertIn("nicht ausfuehrbar", rows[0]["status"])


@unittest.skipUnless(torch.cuda.is_available(), "keine GPU vorhanden")
class GpuTest(unittest.TestCase):
    def test_gpu_gives_the_same_plans_as_cpu(self):
        instances = [instance_of("test_klein", 1000 + i) for i in range(3)]
        cpu = TDFJSPEnv(CONFIG)
        gpu = TDFJSPEnv(variant(environment={"device": "cuda"}))
        cpu_td, cpu_total = rollout(cpu, cpu.load_instances(instances), edd_ect)
        gpu_td, gpu_total = rollout(gpu, gpu.load_instances(instances), edd_ect)
        self.assertEqual(gpu_td["time"].device.type, "cuda")
        for i, instance in enumerate(instances):
            self.assertEqual(cpu.to_schedule(cpu_td, i, instance), gpu.to_schedule(gpu_td, i, instance))
        self.assertTrue(torch.allclose(cpu_total, gpu_total.cpu()))
        random_td, _ = rollout(gpu, gpu.load_instances(instances), make_random_policy(1, "cuda"))
        gpu.check_solution_validity(random_td, instances)


if __name__ == "__main__":
    unittest.main()
