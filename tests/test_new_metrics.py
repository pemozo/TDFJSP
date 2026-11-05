"""Independent checks for machine productivity, profile diagnostics and plan exports."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from generator import generate_class_instance
from monitoring.experiment_logger import ExperimentLogger
from monitoring.metrics_aggregator import summarize_instances
from monitoring.schedule_metrics import (PLAN_METRIC_NAMES, machine_metrics, manual_profile_metrics,
                                         profile_metadata, time_metrics, worker_metrics)
from monitoring.training_monitor import TrainingMonitor
from rl.env import TDFJSPEnv
from rl.instance_results import evaluate_instances, train_rows
from rl.policies import edd_ect, rollout
from schedule_validator import load_config, time_profile_id


def small_calendar():
    config = copy.deepcopy(load_config())
    config["time_model"].update(day_minutes=120, day_windows=[
        {"start": 0, "end": 40, "factor": 1.0, "label": "first"},
        {"start": 40, "end": 80, "factor": 1.0, "label": "second"},
        {"start": 80, "end": 90, "factor": None, "label": "break"},
        {"start": 90, "end": 120, "factor": 1.0, "label": "last"},
    ])
    config["time_model"].pop("resource_factors", None)
    return config


class ScheduleDiagnosticsTest(unittest.TestCase):
    def test_setup_increases_occupancy_but_never_productive_utilization(self):
        instance = {"resources": {"manual_stations": ["H1"], "machines": ["M1"]}}
        schedule = [{"op": "a", "resource": "M1", "start": 20, "end": 60,
                     "setup_start": 15, "setup_end": 20},
                    {"op": "b", "resource": "H1", "start": 80, "end": 100}]
        first = machine_metrics(instance, schedule)
        schedule[0]["setup_start"] = 5
        second = machine_metrics(instance, schedule)
        self.assertEqual(first["machine_productive_utilization"], .4)
        self.assertEqual(second["machine_productive_utilization"], .4)
        self.assertEqual(first["machine_occupied_utilization"], .45)
        self.assertEqual(second["machine_occupied_utilization"], .55)
        self.assertEqual(first["machine_idle_time"], 55)
        self.assertEqual(second["machine_idle_time"], 45)
        self.assertEqual(first["machine_flow_time"], 60)

    def test_instance_profile_and_base_weighted_factor(self):
        config = small_calendar()
        original = copy.deepcopy(config)
        profiles = {"manual_stations": [[2.0, 1.0, None, 1.0]], "setup": [1.0, 1.0, None, 1.0]}
        instance = {"days": 1, "resources": {"manual_stations": ["H1"], "machines": []},
                    "operations": [{"id": "a", "kind": "manual", "base_processing": {"H1": 20}},
                                   {"id": "b", "kind": "manual", "base_processing": {"H1": 40}}],
                    "setups": [], "time_profiles": profiles, "time_profile_id": time_profile_id(profiles)}
        schedule = [{"op": "a", "resource": "H1", "start": 0, "end": 40},
                    {"op": "b", "resource": "H1", "start": 40, "end": 80}]
        metrics = manual_profile_metrics(config, instance, schedule)
        self.assertEqual(metrics["manual_base_time"], 60)
        self.assertAlmostEqual(metrics["weighted_manual_factor"], 4/3)
        self.assertEqual(metrics["manual_profile_regret"], 20)
        self.assertEqual(metrics["manual_favourable_start_share"], .5)
        self.assertAlmostEqual(metrics["manual_favourable_base_share"], 2/3)
        self.assertEqual(worker_metrics(config, instance, schedule)["manual_extra_time"], 20)
        self.assertEqual(time_metrics(config, instance, schedule, 0)["mean_processing_factor"], 1.5)
        metadata = profile_metadata(config, instance)
        self.assertEqual(metadata["time_profile_id"], instance["time_profile_id"])
        changed = copy.deepcopy(instance)
        changed["time_profiles"]["manual_stations"][0][0] = 1.5
        self.assertNotEqual(metadata["time_profile_hash"], profile_metadata(config, changed)["time_profile_hash"])
        self.assertEqual(config, original)

    def test_calendar_infeasible_minimum_is_excluded(self):
        config = small_calendar()
        config["time_model"]["day_windows"] = [
            {"start": 0, "end": 100, "factor": 1.5, "label": "long"},
            {"start": 100, "end": 110, "factor": None, "label": "break"},
            {"start": 110, "end": 120, "factor": 1.0, "label": "too_short"}]
        instance = {"days": 1, "resources": {"manual_stations": ["H1"], "machines": []},
                    "operations": [{"id": "a", "kind": "manual", "base_processing": {"H1": 20}}]}
        metrics = manual_profile_metrics(config, instance, [{"op": "a", "resource": "H1", "start": 0, "end": 30}])
        self.assertEqual(metrics["manual_profile_regret"], 0)
        self.assertEqual(metrics["manual_favourable_start_share"], 1)

    def test_instance_means_do_not_pool_capacity(self):
        rows = [{"jobs": 2, "late_jobs": 0, "feasible": True, "machine_productive_utilization": .2},
                {"jobs": 8, "late_jobs": 4, "feasible": True, "machine_productive_utilization": .8}]
        summary = summarize_instances(rows)
        self.assertEqual(summary["machine_productive_utilization_mean"], .5)
        self.assertEqual(summary["window_compliance"], .6)

    def test_train_and_evaluation_diagnostics_agree(self):
        config = copy.deepcopy(load_config())
        config["environment"].update(device="cpu", compute_features=False)
        config["instance_classes"]["metric_mini"] = {"jobs": 4, "operations_per_job": 3,
                                                       "workers": 1, "machines": 2, "days": 1}
        instance = generate_class_instance(config, "metric_mini", 81234)
        # Use an explicit instance profile so the tensor and Python paths cannot
        # accidentally agree by both ignoring per-instance factors.
        profile = [None if w["factor"] is None else 1.1 for w in config["time_model"]["day_windows"]]
        instance["time_profiles"] = {"manual_stations": [profile], "setup": profile.copy()}
        instance["time_profile_id"] = time_profile_id(instance["time_profiles"])
        env = TDFJSPEnv(config)
        final, _ = rollout(env, env.load_instances([instance]), edd_ect)
        training = train_rows(env, final, torch.zeros(1, dtype=torch.long), torch.tensor([12]), "metric_mini")[0]
        evaluated = evaluate_instances(env, edd_ect, [instance], ["metric_mini/x"], "metric_mini")[0].row
        self.assertTrue(evaluated["feasible"])
        for name in PLAN_METRIC_NAMES:
            self.assertAlmostEqual(training[name], evaluated[name], msg=name)
        self.assertEqual(training["time_profile_hash"], evaluated["time_profile_hash"])
        self.assertEqual(training["time_profile_id"], evaluated["time_profile_id"])


class SchedulePersistenceTest(unittest.TestCase):
    def test_opt_in_test_only_export_and_resume(self):
        config = copy.deepcopy(load_config())
        settings = {**config["monitoring"], "save_schedules": True}
        row = {"instance_id": "mini/a", "instance_seed": 123, "jobs": 1, "late_jobs": 0,
               "feasible": True, "truncated": False, "makespan": 20, "reward": 0,
               "time_profile_id": "profile", "time_profile_hash": "hash"}
        result = SimpleNamespace(row=row, instance={}, start_times=[0],
                                 schedule=[{"op": "a", "resource": "H1", "start": 0, "end": 20}],
                                 waits=[{"resource": "M1", "start": 0, "end": 5}])
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(ExperimentLogger, "_open_writer", return_value=None):
            with ExperimentLogger("run", folder, config) as logger:
                monitor = TrainingMonitor(logger, settings)
                monitor.after_validation([result], 5)
                self.assertFalse((logger.results_dir / "val_schedules.jsonl").exists())
                monitor.after_test([result], 10)
                logger.save_schedules([result], "test", 20)
                records = [json.loads(line) for line in (logger.results_dir / "test_schedules.jsonl").read_text().splitlines()]
                self.assertEqual(records[0]["schedule"], result.schedule)
                self.assertEqual(records[0]["waits"], result.waits)
                self.assertEqual(records[0]["time_profile_hash"], "hash")
            with ExperimentLogger("run", folder, config, resume_step=10) as logger:
                records = [json.loads(line) for line in (logger.results_dir / "test_schedules.jsonl").read_text().splitlines()]
                self.assertEqual([r["step"] for r in records], [10])
                self.assertIn("machine_productive_utilization", logger.metadata["metric_definitions"])

    def test_training_metrics_logged_without_training_csv(self):
        logger = mock.Mock(enabled=True)
        config = load_config()
        monitor = TrainingMonitor(logger, {**config["monitoring"], "save_train_instances": False,
                                           "log_every_updates": 1})
        monitor.after_iteration({"reward": [-1], "makespan": [100], "tardiness": [2], "feasible": [True],
                                 "episode_length": [3]}, jobs=2, late_jobs=1, ppo={}, learning_rate=.001,
                                step=1, processed_instances=1, elapsed_seconds=0, wait_share=0,
                                rows=[{"machine_productive_utilization": .4, "manual_profile_regret": 12}])
        logger.save_instance_results.assert_not_called()
        values = logger.log_train_metrics.call_args.args[0]
        self.assertEqual(values["machine_productive_utilization_mean"], .4)
        self.assertEqual(values["manual_profile_regret_mean"], 12)


if __name__ == "__main__":
    unittest.main()
