"""Tests des Experiment-Trackings und der PPO-Kennzahlen.

Aufruf aus dem Projektordner: python -m unittest tests.test_monitoring
"""
from __future__ import annotations

import copy
import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import pandas as pd
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.distributions import Categorical

from baselines.dispatcher import run_dispatcher
from evaluation import evaluate
from generator import generate_class_instance, save_instance
from monitoring.comparison_plotter import (Unavailable, load_baselines, make_all, real_duration_rows, reference_gap,
                                           temporal_gains)
from monitoring.experiment_logger import INSTANCE_COLUMNS, ExperimentLogger, NullLogger, reserve_run_id
from monitoring.metrics_aggregator import MetricsAggregator, summarize_instances, summarize_temporal
from monitoring.schedule_metrics import gain_metrics, relative_gain, start_shares, temporal_variability, time_metrics
from monitoring.training_monitor import TrainingMonitor
from rl.decoder import MASKED_LOGIT
from rl.env import TDFJSPEnv
from rl.generators import TDFJSPFileGenerator
from rl.instance_results import evaluate_class, evaluate_instances, rule_rows
from rl.observation import ObservationBuilder
from rl.policies import edd_ect
from rl.policy import build_model
from rl.ppo import collect_rollout, explained_variance, ppo_update
from rl.train import train
from schedule_validator import all_resources, check_config, load_config
from simulator import execution_instance

CONFIG = load_config()
CONFIG["environment"].update(device="cpu", allow_wait=True, view="time_aware", compute_features=False,
                             train_classes=["mini"], generator_workers=0)
CONFIG["simulation"]["overflow_days"] = 1
CONFIG["instance_classes"]["mini"] = {"jobs": 4, "operations_per_job": 3, "workers": 1, "machines": 2, "days": 1}
CONFIG["model"].update(variant="A3", time_information="full", d_model=32, num_layers=2, num_heads=4, hidden_dim=32)


def mini_instances(count: int = 3, first_seed: int = 6000) -> list[dict]:
    return [generate_class_instance(CONFIG, "mini", first_seed + i) for i in range(count)]


def read_events(folder: Path) -> EventAccumulator:
    events = EventAccumulator(str(folder), size_guidance={"scalars": 0, "images": 0, "histograms": 0})
    events.Reload()
    return events


def scalar_steps(folder: Path, tag: str) -> list[int]:
    return [event.step for event in read_events(folder).Scalars(tag)]


def run_config(folder: Path, **training) -> dict:
    """Kleine Trainings-Konfiguration mit Validierungs- und Testinstanzen in `folder`."""
    config = copy.deepcopy(CONFIG)
    for split, seed in (("val", 6100), ("test", 6200)):
        config["data_splits"][split]["folder"] = str(folder / split / "{class}")
        for i, instance in enumerate(mini_instances(2, seed)):
            path = folder / split / "mini" / f"instance_{i:03d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_instance(instance, path)
    config["training"].update(output_folder=str(folder / "out" / "checkpoints"), instances_per_iteration=3,
                              validate_every=1, validation_instances=2, minibatch_size=32, ppo_epochs=1)
    config["training"].update(training)
    config["monitoring"].update(folder=str(folder / "out"), log_every_updates=1, gantt_every_validations=1,
                                gantt_instances=1)
    return config


def log_without_time(output: Path) -> list[dict]:
    with (output / "training_log.csv").open(encoding="utf-8") as file:
        return [{k: v for k, v in row.items() if k != "seconds"} for row in csv.DictReader(file)]


class LoggerTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.addCleanup(self.folder.cleanup)

    def test_1_logger_initializes_without_training_code(self):
        code = ("import sys, tempfile, pathlib\n"
                "from monitoring.experiment_logger import ExperimentLogger\n"
                "folder = tempfile.mkdtemp()\n"
                "logger = ExperimentLogger('probe', folder)\n"
                "logger.log_train_metrics({'x': 1.0}, 1)\n"
                "logger.close()\n"
                "import shutil; shutil.rmtree(folder, ignore_errors=True)\n"
                "assert not [m for m in sys.modules if m == 'rl' or m.startswith('rl.')], 'rl importiert'\n"
                "assert 'rl4co' not in sys.modules and 'lightning' not in sys.modules\n")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                cwd=Path(__file__).parent.parent)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_2_metric_is_stored_in_tensorboard(self):
        with ExperimentLogger("run", self.root) as logger:
            logger.log_train_metrics({"reward_mean": -1.5}, 7)
            logger.log_ppo_metrics({"ppo/approx_kl": 0.25}, 8)
            logger.log_validation_metrics({"reward_mean": -1.25}, 9)
            logger.log_temporal_metrics({"wait_share": 0.5, "start_times": [10, 20, 30]}, 9)
            run = logger.run_id
        events = read_events(self.root / "tensorboard" / run)
        self.assertEqual([(e.step, e.value) for e in events.Scalars("train/reward_mean")], [(7, -1.5)])
        self.assertEqual([(e.step, e.value) for e in events.Scalars("ppo/approx_kl")], [(8, 0.25)])
        self.assertEqual(events.Scalars("val/reward_mean")[0].value, -1.25)
        self.assertIn("temporal/start_times", events.Tags()["histograms"])

        with (self.root / "results" / run / "scalars.csv").open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertIn({"step": "7", "tag": "train/reward_mean", "value": "-1.5"},
                      [{k: r[k] for k in ("step", "tag", "value")} for r in rows])
        metadata = json.loads((self.root / "results" / run / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "finished")
        self.assertGreaterEqual(metadata["runtime_seconds"], 0)

    def test_3_runs_stay_separate(self):
        with ExperimentLogger("run", self.root) as first, ExperimentLogger("run", self.root) as second:
            first.log_train_metrics({"only_first": 1.0}, 1)
            second.log_train_metrics({"only_second": 2.0}, 1)
        self.assertEqual((first.run_id, second.run_id), ("run", "run_r2"))
        first_tags = read_events(self.root / "tensorboard" / "run").Tags()["scalars"]
        second_tags = read_events(self.root / "tensorboard" / "run_r2").Tags()["scalars"]
        self.assertEqual((first_tags, second_tags), (["train/only_first"], ["train/only_second"]))

    def test_3b_reserved_names_cover_all_folders(self):
        (self.root / "b" / "run").mkdir(parents=True)
        self.assertEqual(reserve_run_id("run", [self.root / "a", self.root / "b"]), "run_r2")
        self.assertTrue((self.root / "b" / "run_r2").is_dir())

    def test_4_global_steps_stay_monotonic(self):
        with ExperimentLogger("run", self.root) as logger:
            logger.log_train_metrics({"value": 1.0}, 5)
            logger.log_train_metrics({"value": 2.0}, 10)
            with self.assertWarns(RuntimeWarning):
                logger.log_train_metrics({"value": 3.0}, 7)
            logger.log_train_metrics({"value": 4.0}, 10)
            logger.log_train_metrics({"value": 5.0}, 12)
            logger.log_ppo_metrics({"value": 1.0}, 1)   # jede Gruppe zaehlt fuer sich
        self.assertEqual(scalar_steps(self.root / "tensorboard" / "run", "train/value"), [5, 10, 10, 12])

    def test_6_instance_results_are_fully_readable(self):
        rows = [{"instance_id": "mini/a", "makespan": 100, "feasible": True, "reward": -0.5},
                {"instance_id": "mini/b", "makespan": 120, "feasible": False, "reward": None}]
        with ExperimentLogger("run", self.root, metadata={"model": "A3", "seed": 4}) as logger:
            logger.save_instance_results(rows, "val", 3)
            logger.save_instance_results([{"instance_id": "mini/c", "makespan": 90, "new_metric": 1.5}], "val", 6)
            logger.save_instance_results(rows, "test", 6)
        val = pd.read_csv(self.root / "results" / "run" / "val_instances.csv")
        self.assertEqual(len(val), 3)
        self.assertEqual(list(val.columns[:len(INSTANCE_COLUMNS)]), list(INSTANCE_COLUMNS))
        self.assertEqual(val["makespan"].tolist(), [100, 120, 90])
        self.assertEqual(val["step"].tolist(), [3, 3, 6])
        self.assertEqual(val["new_metric"].iloc[2], 1.5)
        self.assertTrue(math.isnan(val["new_metric"].iloc[0]))
        self.assertTrue(math.isnan(val["reward"].iloc[1]))
        self.assertEqual(val["seed"].tolist(), [4, 4, 4])
        self.assertEqual(len(pd.read_csv(self.root / "results" / "run" / "test_instances.csv")), 2)
        self.assertFalse((self.root / "results" / "run" / "train_instances.csv").exists())

    def test_8_unavailable_metrics_do_not_crash(self):
        with ExperimentLogger("run", self.root) as logger:
            logger.log_validation_metrics({"present": 1.0, "missing": None}, 1)
            with self.assertWarns(RuntimeWarning):
                logger.log_ppo_metrics({"nan": float("nan"), "inf": float("inf"), "fine": 2.0}, 1)
            self.assertIsNone(logger.log_gantt([], "mini/a", 1))
            logger.log_temporal_metrics({"start_times": []}, 1)
        metadata = json.loads((self.root / "results" / "run" / "run.json").read_text(encoding="utf-8"))
        self.assertIn("val/missing", metadata["unavailable_metrics"])
        self.assertIn("gantt/mini/a", metadata["unavailable_metrics"])
        self.assertEqual(metadata["non_finite_metrics"], {"ppo/nan": 1, "ppo/inf": 1})
        tags = read_events(self.root / "tensorboard" / "run").Tags()["scalars"]
        self.assertEqual(sorted(tags), ["ppo/fine", "val/present"])

    def test_8b_missing_tensorboard_package_keeps_csv(self):
        with mock.patch.dict(sys.modules, {"torch.utils.tensorboard": None}):
            with self.assertWarns(RuntimeWarning):
                logger = ExperimentLogger("run", self.root)
            logger.log_train_metrics({"value": 1.0}, 1)
            logger.close()
        self.assertEqual(len(pd.read_csv(self.root / "results" / "run" / "scalars.csv")), 1)
        self.assertIn("tensorboard", logger.metadata["unavailable_metrics"])

    def test_8c_values_without_a_number_are_marked_unavailable(self):
        with ExperimentLogger("run", self.root) as logger:
            logger.log_train_metrics({"text": "abc", "tensor": torch.tensor(2.5), "flag": True}, 1)
        self.assertIn("train/text", logger.metadata["unavailable_metrics"])
        values = read_events(self.root / "tensorboard" / "run")
        self.assertEqual(values.Scalars("train/tensor")[0].value, 2.5)
        self.assertEqual(values.Scalars("train/flag")[0].value, 1.0)

    def test_9b_resume_discards_what_came_after_the_checkpoint(self):
        with ExperimentLogger("run", self.root, metadata={"model": "A3"}) as logger:
            for step in (5, 10, 15, 20):
                logger.log_train_metrics({"value": float(step)}, step)
            logger.save_instance_results([{"instance_id": "a"}], "val", 10)
            logger.save_instance_results([{"instance_id": "b"}], "val", 20)
        with ExperimentLogger("run", self.root, resume_step=10) as logger:
            self.assertEqual(logger.run_id, "run")
            with self.assertWarns(RuntimeWarning):
                logger.log_train_metrics({"value": 9.0}, 9)     # liegt vor dem Checkpoint
            logger.log_train_metrics({"value": 12.0}, 12)
        events = read_events(self.root / "tensorboard" / "run").Scalars("train/value")
        self.assertEqual([(event.step, event.value) for event in events], [(5, 5.0), (10, 10.0), (12, 12.0)])
        scalars = pd.read_csv(self.root / "results" / "run" / "scalars.csv")
        self.assertEqual(scalars["step"].tolist(), [5, 10, 12])
        self.assertEqual(pd.read_csv(self.root / "results" / "run" / "val_instances.csv")["instance_id"].tolist(), ["a"])
        metadata = json.loads((self.root / "results" / "run" / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["model"], "A3")
        self.assertEqual(metadata["resumed"][0]["step"], 10)

    def test_null_logger_accepts_everything(self):
        logger = NullLogger()
        self.assertFalse(logger.enabled)
        self.assertIsNone(logger.log_train_metrics({"a": 1}, 1))
        self.assertIsNone(logger.close("finished"))


class AggregationTest(unittest.TestCase):
    def test_5_kpis_of_a_small_evaluation_set(self):
        rows = [{"reward": -1.0, "makespan": 100, "total_tardiness": 10, "late_jobs": 1, "jobs": 10, "feasible": True,
                 "setup_time_total": 30, "processing_time_total": 300, "inference_time": 0.1, "wait_actions": 2,
                 "decisions": 20, "voluntary_idle_time": 4, "unavoidable_idle_time": 40,
                 "start_share_a": 0.5, "start_share_b": 0.5},
                {"reward": -3.0, "makespan": 200, "total_tardiness": 30, "late_jobs": 5, "jobs": 30, "feasible": True,
                 "setup_time_total": 50, "processing_time_total": 500, "inference_time": 0.3, "wait_actions": 8,
                 "decisions": 60, "voluntary_idle_time": 6, "unavoidable_idle_time": 60,
                 "start_share_a": 1.0, "start_share_b": 0.0},
                {"reward": -8.0, "makespan": 300, "total_tardiness": 20, "late_jobs": 0, "jobs": 10, "feasible": False}]
        summary = summarize_instances(rows)
        self.assertEqual(summary["reward_mean"], -4.0)
        self.assertEqual(summary["makespan_mean"], 200)
        self.assertEqual(summary["tardiness_mean"], 20)
        # ueber alle Jobs, nicht Mittel der Anteile: 6 von 50 verspaetet
        self.assertAlmostEqual(summary["window_compliance"], 1 - 6 / 50)
        self.assertEqual(summary["setup_time_total_mean"], 40)      # die dritte Zeile hat keinen Plan
        self.assertAlmostEqual(summary["feasibility_rate"], 2 / 3)
        temporal = summarize_temporal(rows)
        self.assertEqual(temporal["wait_share"], 10 / 80)
        self.assertEqual(temporal["voluntary_idle_time_mean"], 5)
        self.assertEqual(temporal["unavoidable_idle_time_mean"], 50)
        self.assertEqual(temporal["start_share/a"], 0.75)
        self.assertIsNone(temporal["mean_setup_factor"])

    def test_episode_and_minibatch_means_are_weighted(self):
        aggregator = MetricsAggregator()
        aggregator.add({"loss": 1.0, "missing": None}, 3)
        aggregator.add({"loss": 5.0}, 1)
        self.assertEqual(aggregator.means(), {"loss": 2.0})
        aggregator.reset()
        self.assertFalse(aggregator)
        aggregator.add_values("reward", [1.0, 3.0])
        self.assertEqual(aggregator.stds()["reward"], 1.0)

    def test_environment_metrics_agree_with_independent_evaluation(self):
        env = TDFJSPEnv(CONFIG)
        chosen = mini_instances(4)
        results = evaluate_instances(env, edd_ect, chosen, [f"mini/{i}" for i in range(4)], "mini")
        for result, instance in zip(results, chosen):
            check = evaluate(CONFIG, execution_instance(CONFIG, instance), result.schedule)
            self.assertTrue(check["feasible"])
            self.assertEqual(result.row["feasible"], True)
            self.assertEqual(result.row["makespan"], check["makespan"])
            self.assertEqual(result.row["total_tardiness"], check["total_tardiness"])
            self.assertEqual(result.row["late_jobs"], check["late_jobs"])
            self.assertEqual(result.row["wait_actions"], 0)
            self.assertEqual(result.row["voluntary_idle_time"], 0)
            self.assertEqual(result.row["decisions"], 12)
            self.assertEqual(result.row["instance_size"], 12)


class TemporalMetricsTest(unittest.TestCase):
    def test_voluntary_and_unavoidable_idle_are_measured_separately(self):
        env = TDFJSPEnv(CONFIG)
        instance = mini_instances(1, 6300)[0]
        state = env.load_instances([instance])
        startable = int(state["pair_mask"].any(1)[0].sum())
        self.assertGreater(startable, 0)
        expected_until = int(env.wait_target(state)[0])
        used = {"done": False}

        def waits_once(td):
            if not used["done"]:
                used["done"] = True
                return torch.zeros(1, dtype=torch.long)
            return edd_ect(td)

        result = evaluate_instances(env, waits_once, [instance], ["mini/x"], "mini")[0]
        self.assertEqual(result.row["wait_actions"], 1)
        self.assertEqual(result.row["voluntary_idle_time"], startable * expected_until)
        self.assertEqual(sum(w["end"] - w["start"] for w in result.waits), startable * expected_until)
        total_idle = len(all_resources(instance)) * result.row["makespan"] - result.row["processing_time_total"] \
            - result.row["setup_time_total"]
        self.assertEqual(result.row["voluntary_idle_time"] + result.row["unavoidable_idle_time"], total_idle)
        self.assertGreaterEqual(result.row["unavoidable_idle_time"], 0)
        # waehrend der Wartezeit laeuft auf den genannten Ressourcen nichts
        for wait in result.waits:
            for entry in result.schedule:
                if entry["resource"] == wait["resource"]:
                    self.assertTrue(entry.get("setup_start", entry["start"]) >= wait["end"] or entry["end"] <= wait["start"])

    def test_factors_and_totals_from_a_hand_made_schedule(self):
        instance = {"operations": [{"id": "a", "kind": "manual"}, {"id": "b", "kind": "automatic"}],
                    "workers": ["W1"], "machines": ["M1"], "stations": ["H1"], "station_worker": {"H1": "W1"}}
        schedule = [{"op": "a", "resource": "H1", "start": 0, "end": 20},
                    {"op": "b", "resource": "M1", "start": 400, "end": 440, "setup_from": "x",
                     "setup_start": 390, "setup_end": 400}]
        with mock.patch("monitoring.schedule_metrics.all_resources", return_value=["H1", "M1"]):
            metrics = time_metrics(CONFIG, instance, schedule, voluntary_idle=5)
        self.assertEqual(metrics["processing_time_total"], 60)
        self.assertEqual(metrics["setup_time_total"], 10)
        self.assertEqual(metrics["mean_processing_factor"], 1.0)   # nur die manuelle Operation um 06:00
        self.assertEqual(metrics["mean_setup_factor"], 1.6)        # Ruestung um 12:30
        self.assertEqual(metrics["voluntary_idle_time"], 5)
        self.assertEqual(metrics["unavoidable_idle_time"], 2 * 440 - 70 - 5)
        shares = start_shares(CONFIG, schedule)
        self.assertEqual(shares["start_share_06:00-09:00"], 0.5)
        self.assertEqual(shares["start_share_12:30-14:00"], 0.5)
        self.assertAlmostEqual(sum(shares.values()), 1.0)

    def test_temporal_variability_is_zero_without_time_dependence(self):
        instance = mini_instances(1)[0]
        self.assertGreater(temporal_variability(CONFIG, instance), 0)
        flat = copy.deepcopy(CONFIG)
        for window in flat["time_model"]["day_windows"]:
            if window["factor"] is not None:
                window["factor"] = 1.0
        self.assertEqual(temporal_variability(flat, instance), 0)

    def test_gain_follows_the_requested_formula(self):
        self.assertEqual(relative_gain(200, 150), 25.0)
        self.assertEqual(relative_gain(0.5, 0.75, higher_is_better=True), 50.0)
        self.assertIsNone(relative_gain(0, 10))
        self.assertIsNone(relative_gain(None, 10))
        static = [{"instance_id": "a", "feasible": True, "makespan": 100, "total_tardiness": 40, "late_jobs": 4, "jobs": 10},
                  {"instance_id": "b", "feasible": True, "makespan": 300, "total_tardiness": 0, "late_jobs": 0, "jobs": 10},
                  {"instance_id": "c", "feasible": False, "makespan": 999, "total_tardiness": 0, "late_jobs": 0, "jobs": 10}]
        model = [{"instance_id": "a", "feasible": True, "makespan": 90, "total_tardiness": 10, "late_jobs": 1, "jobs": 10},
                 {"instance_id": "b", "feasible": True, "makespan": 270, "total_tardiness": 10, "late_jobs": 1, "jobs": 10},
                 {"instance_id": "c", "feasible": True, "makespan": 1, "total_tardiness": 0, "late_jobs": 0, "jobs": 10}]
        gains = gain_metrics(model, static)
        self.assertEqual(gains["gain_instances"], 2)
        self.assertAlmostEqual(gains["temporal_gain"], (200 - 180) / 200 * 100)
        self.assertAlmostEqual(gains["tardiness_gain"], (20 - 10) / 20 * 100)
        self.assertAlmostEqual(gains["compliance_gain"], (0.9 - 0.8) / 0.8 * 100)
        self.assertIsNone(gain_metrics(model, [])["temporal_gain"])


class PPOMetricsTest(unittest.TestCase):
    def test_entropy_counts_only_allowed_actions(self):
        logits = torch.tensor([[0.5, 1.5, -0.5, 2.0]])
        allowed = torch.tensor([[True, False, True, True]])
        masked = logits.masked_fill(~allowed, MASKED_LOGIT)
        direct = Categorical(logits=logits[allowed]).entropy()
        self.assertAlmostEqual(float(Categorical(logits=masked).entropy()), float(direct), places=5)

    def test_explained_variance(self):
        returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(explained_variance(returns, returns), 1.0)
        self.assertAlmostEqual(explained_variance(returns.mean().expand(4), returns), 0.0, places=5)
        self.assertIsNone(explained_variance(returns, torch.ones(4)))

    def test_update_metrics_come_from_the_real_computation(self):
        torch.manual_seed(0)
        env = TDFJSPEnv(CONFIG)
        model = build_model(env)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        settings = {**CONFIG["training"], "minibatch_size": 16, "ppo_epochs": 2}
        data = collect_rollout(env, model, ObservationBuilder(env), env.load_instances(mini_instances(3)), settings)
        seen = []
        result = ppo_update(model, optimizer, data, settings, lambda metrics, samples: seen.append((metrics, samples)))
        self.assertGreater(len(seen), 1)
        self.assertEqual(sum(samples for _, samples in seen), 2 * data.action.shape[0])
        expected = sum(metrics["grad_norm"] * samples for metrics, samples in seen) / sum(s for _, s in seen)
        self.assertAlmostEqual(result["grad_norm"], expected)
        first = seen[0][0]
        self.assertGreater(first["grad_norm"], 0.0)
        self.assertGreater(first["entropy"], 0.0)
        # Im ersten Minibatch hat sich das Netz noch nicht geaendert: Verhaeltnis 1, also KL 0 und nichts gekappt
        self.assertAlmostEqual(first["approx_kl"], 0.0, places=4)
        self.assertEqual(first["clip_fraction"], 0.0)
        self.assertAlmostEqual(result["explained_variance"], explained_variance(data.values, data.returns))
        self.assertEqual(int(data.wait_actions.sum()), int((data.action == 0).sum()))
        self.assertEqual(int(data.decisions.sum()), data.action.shape[0])


class GanttTest(unittest.TestCase):
    def test_7_gantt_from_a_small_schedule(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            instance = mini_instances(1, 6400)[0]
            schedule = run_dispatcher(CONFIG, instance, "EDD", "ECT", "time_aware")["executed"]
            waits = [{"resource": all_resources(instance)[0], "start": 0, "end": 15}]
            with ExperimentLogger("run", root, CONFIG) as logger:
                stem = logger.log_gantt(schedule, "mini/instance_000", 5, instance, waits)
            for suffix in (".png", ".pdf"):
                self.assertGreater(stem.with_suffix(suffix).stat().st_size, 2000)
            self.assertIn("gantt/mini/instance_000", read_events(root / "tensorboard" / "run").Tags()["images"])

    def test_monitor_draws_gantt_for_chosen_feasible_instances_only(self):
        with tempfile.TemporaryDirectory() as folder:
            env = TDFJSPEnv(CONFIG)
            results = evaluate_instances(env, edd_ect, mini_instances(2, 6400), ["mini/a", "mini/b"], "mini")
            broken = copy.copy(results[1])
            broken.row = {**results[1].row, "feasible": False}
            with ExperimentLogger("run", folder, CONFIG) as logger:
                monitor = TrainingMonitor(logger, CONFIG["monitoring"], gantt_ids=("mini/a", "mini/b"))
                monitor.after_validation([results[0], broken], 5)
            names = [path.name for path in (Path(folder) / "figures" / "run" / "gantt").glob("*.png")]
            self.assertEqual(names, ["mini_a_step5.png"])
            self.assertIn("gantt/mini/b", logger.metadata["unavailable_metrics"])

    def test_gantt_shows_extra_day_only_when_needed(self):
        import matplotlib.pyplot as plt

        from monitoring.gantt_plotter import save_gantt
        with tempfile.TemporaryDirectory() as folder:
            instance = mini_instances(1, 6400)[0]
            schedule = run_dispatcher(CONFIG, instance, "EDD", "ECT", "time_aware")["executed"]
            one_day = save_gantt(CONFIG, instance, schedule, "t", Path(folder) / "a")
            self.assertEqual(one_day.axes[0].get_xlim()[1], 720)
            late = [*schedule, {"op": schedule[-1]["op"], "resource": schedule[-1]["resource"], "start": 800, "end": 820}]
            two_days = save_gantt(CONFIG, instance, late, "t", Path(folder) / "b")
            self.assertEqual(two_days.axes[0].get_xlim()[1], 1440)
            plt.close("all")


class TrainingIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.addCleanup(self.folder.cleanup)
        self.config_path = self.root / "problem_config.json"

    def test_config_settings_are_checked(self):
        config = copy.deepcopy(CONFIG)
        config["monitoring"]["log_every_updates"] = 0
        config["training"]["best_metric"] = "val/unknown"
        errors = " ".join(check_config(config))
        self.assertIn("log_every_updates", errors)
        self.assertIn("best_metric", errors)

    def test_full_run_stores_all_metric_groups(self):
        config = run_config(self.root)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            output = train(copy.deepcopy(config), 3, 2, self.config_path)
        run = output.name
        self.assertEqual(run, "A3_zeit_warten_ppo_seed3")
        results = self.root / "out" / "results" / run
        for name in ("run.json", "config.json", "scalars.csv", "train_instances.csv", "val_instances.csv",
                     "test_instances.csv", "val_static_reference_instances.csv"):
            self.assertTrue((results / name).exists(), name)
        tags = set(read_events(self.root / "out" / "tensorboard" / run).Tags()["scalars"])
        required = {f"train/{k}" for k in ("reward_mean", "reward_std", "makespan_mean", "tardiness_mean",
                                           "window_compliance", "feasibility_rate", "episode_length", "learning_rate",
                                           "processed_instances", "elapsed_seconds")}
        required |= {f"ppo/{k}" for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
                                          "grad_norm", "explained_variance")}
        required |= {f"val/{k}" for k in ("reward_mean", "makespan_mean", "tardiness_mean", "window_compliance",
                                          "feasibility_rate", "inference_time_mean")}
        self.assertEqual(required - tags, set())
        metadata = json.loads((results / "run.json").read_text(encoding="utf-8"))
        for key in ("experiment_id", "variant", "seed", "hyperparameters", "encoder", "decoder", "algorithm", "objective",
                    "instance_generator", "time_profiles", "start_time", "runtime_seconds", "best"):
            self.assertIn(key, metadata)
        self.assertIsNone(metadata["git_commit"])
        self.assertEqual(metadata["best"]["metric"], "val/reward_mean")
        self.assertIn("temporal/decision_relevance", metadata["unavailable_metrics"])
        # Ein kaum trainiertes Netz kann am Horizont abbrechen: dann kein Bild, aber ein Vermerk
        pictures = list((self.root / "out" / "figures" / run / "gantt").glob("mini_instance_000_step*.pdf"))
        self.assertTrue(pictures or "gantt/mini/instance_000" in metadata["unavailable_metrics"])

        val = pd.read_csv(results / "val_instances.csv")
        self.assertEqual(sorted(val["instance_id"].unique()), ["mini/instance_000", "mini/instance_001"])
        self.assertEqual(len(val), 4)   # zwei Validierungen mit je zwei Instanzen
        test = pd.read_csv(results / "test_instances.csv")
        self.assertEqual(len(test), 2)
        self.assertEqual(test["inference_batch_size"].unique().tolist(), [1])
        train_rows = pd.read_csv(results / "train_instances.csv")
        self.assertEqual(len(train_rows), 6)

        checkpoint = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
        for key in ("model", "optimizer", "scheduler", "progress", "random_state", "config", "generator_seed"):
            self.assertIn(key, checkpoint)

    def test_9_resume_continues_steps_and_results(self):
        config = run_config(self.root)
        complete = train(copy.deepcopy(config), 3, 4, self.config_path)
        expected = log_without_time(complete)
        expected_steps = scalar_steps(self.root / "out" / "tensorboard" / complete.name, "train/reward_mean")

        partial = train(copy.deepcopy(config), 3, 2, self.config_path)
        self.assertEqual(partial.name, f"{complete.name}_r2")
        # Ein abgebrochener Verlauf nach dem Checkpoint darf nicht in den Daten bleiben
        results = self.root / "out" / "results" / partial.name
        with (results / "scalars.csv").open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow((10 ** 6, "train/reward_mean", 0.0, 0.0))
        resumed = train({}, 0, 4, self.config_path, resume=partial / "latest.pt")
        self.assertEqual(resumed, partial)
        self.assertEqual(log_without_time(resumed), expected)

        steps = scalar_steps(self.root / "out" / "tensorboard" / resumed.name, "train/reward_mean")
        self.assertEqual(steps, expected_steps)
        self.assertEqual(steps, sorted(steps))
        scalars = pd.read_csv(results / "scalars.csv")
        self.assertLess(scalars["step"].max(), 10 ** 6)
        metadata = json.loads((results / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "finished")
        self.assertEqual(len(metadata["resumed"]), 1)
        test = pd.read_csv(results / "test_instances.csv")     # nur die Auswertung des fertigen Laufs
        self.assertEqual(len(test), 2)
        self.assertEqual(test["step"].unique().tolist(), [scalars["step"].max()])

    def test_10_training_works_without_monitoring_and_gives_the_same_result(self):
        config = run_config(self.root)
        monitored = train(copy.deepcopy(config), 3, 2, self.config_path)
        config["monitoring"]["enabled"] = False
        plain = train(copy.deepcopy(config), 3, 2, self.config_path)
        self.assertEqual(log_without_time(plain), log_without_time(monitored))
        first = torch.load(monitored / "latest.pt", map_location="cpu", weights_only=False)["model"]
        second = torch.load(plain / "latest.pt", map_location="cpu", weights_only=False)["model"]
        self.assertTrue(all(torch.equal(first[key], second[key]) for key in first))
        self.assertFalse((self.root / "out" / "results" / plain.name).exists())
        self.assertFalse((self.root / "out" / "tensorboard" / plain.name).exists())

    def test_missing_validation_instances_leave_no_empty_run(self):
        config = run_config(self.root)
        config["data_splits"]["val"]["folder"] = str(self.root / "fehlt" / "{class}")
        with self.assertRaises(FileNotFoundError):
            train(copy.deepcopy(config), 3, 1, self.config_path)
        self.assertFalse((self.root / "out" / "results").exists())

    def test_monitoring_works_with_a_static_baseline(self):
        config = run_config(self.root)
        config["model"]["variant"] = "A0"
        config["model"]["time_information"] = "none"
        output = train(copy.deepcopy(config), 1, 1, self.config_path)
        self.assertEqual(output.name, "A0_ohnezeit_warten_ppo_seed1")
        self.assertTrue((self.root / "out" / "results" / output.name / "val_instances.csv").exists())

    def test_rule_rows_and_policy_rows_share_their_instances(self):
        config = run_config(self.root)
        generator = TDFJSPFileGenerator(config, "mini", "val", self.config_path)
        rule = {"sequencing": "EDD", "routing": "ECT", "view": "time_aware"}
        rows = rule_rows(config, generator, "mini", 2, rule)
        env = TDFJSPEnv(config, config_path=self.config_path)
        results = evaluate_class(env, edd_ect, generator, "mini", 2)
        self.assertEqual([row["instance_id"] for row in rows], [r.row["instance_id"] for r in results])
        for row, result in zip(rows, results):
            self.assertEqual(row["makespan"], result.row["makespan"])
            self.assertEqual(row["total_tardiness"], result.row["total_tardiness"])
            self.assertIsNone(row["voluntary_idle_time"])   # bei Regeln nicht gemessen, also nicht 0


def comparison_rows(model: str, seed: int, shift: float, view: str = "time_aware") -> list[dict]:
    return [{"run_id": f"{model}_s{seed}", "model": model, "seed": seed, "instance_id": f"k/{i}", "instance_size": size,
             "temporal_variability": 0.07 + i / 100, "makespan": 400 + 10 * i + shift, "total_tardiness": 20 + shift,
             "late_jobs": 1, "jobs": 10, "feasible": True, "reference_makespan": 390, "wait_actions": 3, "decisions": 50,
             "voluntary_idle_time": 12.0, "inference_time": 0.01 * size, "inference_batch_size": 1, "env_view": view}
            for i, size in enumerate((10, 10, 20, 20))]


def comparison_scalars(model: str, seed: int) -> list[dict]:
    return [{"step": step + seed, "tag": tag, "value": value + seed, "elapsed_seconds": 1.0, "run_id": f"{model}_s{seed}",
             "model": model, "seed": seed}
            for tag in ("val/reward_mean", "val/makespan_mean", "val/tardiness_mean", "val/window_compliance")
            for step, value in ((20, 2.0), (40, 1.0))]


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.out = Path(self.folder.name)
        self.addCleanup(self.folder.cleanup)

    def test_all_plots_from_stored_data(self):
        tests = pd.DataFrame(comparison_rows("A3", 0, -10) + comparison_rows("A3", 1, -6) + comparison_rows("A1", 0, 0)
                             + comparison_rows("A1", 1, 2) + comparison_rows("static", 0, 20))
        scalars = pd.DataFrame(sum((comparison_scalars(m, s) for m in ("A3", "A1") for s in (0, 1)), []))
        status = make_all(scalars, tests, self.out, "static")
        self.assertEqual([entry["status"] for entry in status], ["erzeugt"] * 10, status)
        for entry in status:
            self.assertGreater((self.out / entry["detail"]).stat().st_size, 2000)
            self.assertTrue((self.out / entry["detail"]).with_suffix(".pdf").exists())

    def test_missing_data_is_reported_not_invented(self):
        tests = pd.DataFrame(comparison_rows("A3", 0, 0))
        status = {entry["plot"]: entry for entry in make_all(pd.DataFrame(), tests, self.out, "static")}
        self.assertEqual(status["01_val_reward"]["status"], "nicht verfuegbar")
        self.assertEqual(status["06_gain_vs_variability"]["status"], "nicht verfuegbar")
        self.assertIn("einen Seed", status["10_seeds"]["detail"])
        self.assertFalse((self.out / "10_seeds.png").exists())
        self.assertFalse((self.out / "01_val_reward.png").exists())
        self.assertTrue((self.out / "status.csv").exists())

        only_rules = pd.DataFrame(comparison_rows("EDD", 0, 0)).drop(
            columns=["instance_size", "temporal_variability", "wait_actions", "decisions", "voluntary_idle_time"])
        status = {entry["plot"]: entry["status"] for entry in make_all(pd.DataFrame(), only_rules, self.out, "EDD")}
        self.assertEqual(status["07_waiting"], "nicht verfuegbar")
        self.assertEqual(status["08_inference_time"], "nicht verfuegbar")
        self.assertEqual(status["04_variants"], "erzeugt")

        empty = {entry["plot"]: entry["status"] for entry in make_all(pd.DataFrame(), pd.DataFrame(), self.out, "static")}
        self.assertEqual(set(empty.values()), {"nicht verfuegbar"})

    def test_static_environment_is_not_a_direct_comparison(self):
        rows = pd.DataFrame(comparison_rows("A3", 0, 0, view="static"))
        with self.assertRaises(Unavailable):
            real_duration_rows(rows)
        mixed = pd.DataFrame(comparison_rows("A3", 0, 0) + comparison_rows("static", 0, 20, view="static"))
        with self.assertRaises(Unavailable):
            temporal_gains(mixed, "static")

    def test_gain_and_reference_gap_values(self):
        rows = pd.DataFrame(comparison_rows("A3", 0, -10) + comparison_rows("static", 0, 10))
        gains = temporal_gains(rows, "static")
        first = gains[gains["instance_id"] == "k/0"]["temporal_gain"].iloc[0]
        self.assertAlmostEqual(first, (410 - 390) / 410 * 100)
        gaps = reference_gap(rows).set_index(["model", "instance_id"])["reference_gap"]
        self.assertEqual(gaps[("A3", "k/0")], 0.0)                              # gleich dem Referenz-Plan
        self.assertAlmostEqual(gaps[("static", "k/0")], (410 - 390) / 390 * 100)
        better = pd.DataFrame(comparison_rows("A3", 0, -30))                    # besser als der Referenz-Plan
        self.assertEqual(reference_gap(better).set_index("instance_id")["reference_gap"]["k/0"], 0.0)

    def test_baseline_table_must_match_the_instances(self):
        rule = pd.DataFrame([{"class": "k", "instance": "0.json", "seed": 5, "method": "EDD+ECT statisch", "jobs": 10,
                              "late_jobs": 1, "late_share": 0.1, "total_tardiness": 3, "makespan": 400,
                              "reference_makespan": 390, "feasible": True, "runtime_s": 0.1}])
        path = self.out / "rules.csv"
        rule.to_csv(path, index=False)
        runs = pd.DataFrame([{"instance_id": "k/0", "instance_seed": 5, "instance_size": 50, "temporal_variability": 0.07}])
        frame = load_baselines(path, runs)
        self.assertEqual(frame["instance_size"].iloc[0], 50)
        self.assertEqual(frame["window_compliance"].iloc[0], 0.9)
        with self.assertRaises(ValueError):
            load_baselines(path, runs.assign(instance_seed=6))


if __name__ == "__main__":
    unittest.main()
