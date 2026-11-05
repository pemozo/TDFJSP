"""Verbindet die Trainingsschleife mit dem Logger: sammelt Kennzahlen und achtet auf die Intervalle aus dem
Abschnitt monitoring der Config. Kennt weder Modell noch Umgebung; alles kommt als Zahlen und Zeilen an.

Episoden und Minibatches werden getrennt zusammengefasst:
    train/*   je Episode, ueber alle Episoden seit dem letzten Schreiben
    ppo/*     je Minibatch, gewichtet mit seiner Groesse; grad_norm je Optimizer-Schritt
Eine Iteration umfasst viele Optimizer-Schritte, train/* kann also erst am Ende einer Iteration erscheinen.
"""
from __future__ import annotations

from monitoring.metrics_aggregator import MetricsAggregator, summarize_instances, summarize_temporal
from monitoring.schedule_metrics import PLAN_METRIC_NAMES, gain_metrics

DECISION_RELEVANCE_NOTE = ("keine verlaessliche Referenz: fuer die Instanzen gibt es keine bewiesen optimalen "
                           "Plaene; temporal_variability misst nur die rechnerische Streuung der Dauern")


class TrainingMonitor:
    def __init__(self, logger, settings: dict, validations: int = 0, last_step: int = 0,
                 static_rows: list[dict] | None = None, gantt_ids: tuple[str, ...] = ()):
        self.logger = logger
        self.settings = settings
        self.validations = validations
        self.static_rows = static_rows or []
        self.gantt_ids = set(gantt_ids)
        self._episodes = MetricsAggregator()
        self._updates = MetricsAggregator()
        self._last_train_step = last_step
        logger.mark_unavailable("temporal/decision_relevance", DECISION_RELEVANCE_NOTE)

    def on_update(self, metrics: dict[str, float], samples: int, step: int) -> None:
        """Nach jedem Optimizer-Schritt."""
        if not self.logger.enabled:
            return
        self._updates.add({key: value for key, value in metrics.items() if key != "grad_norm"}, samples)
        self._updates.add({"grad_norm": metrics.get("grad_norm")})
        if step % self.settings["log_every_updates"] == 0:
            self._write_updates(step)

    def _write_updates(self, step: int) -> None:
        if self._updates:
            self.logger.log_ppo_metrics(self._updates.means(), step)
            self._updates.reset()

    def after_iteration(self, episodes: dict[str, list[float]], jobs: int, late_jobs: int, ppo: dict,
                        learning_rate: float, step: int, processed_instances: int, elapsed_seconds: float,
                        wait_share: float, rows: list[dict] | None = None) -> None:
        """episodes: je Kennzahl ein Wert je Episode; ppo: Rueckgabe von ppo_update; rows: Zeilen je Instanz."""
        if not self.logger.enabled:
            return
        for name, values in episodes.items():
            self._episodes.add_values(name, values)
        self._episodes.add({"window_compliance": 1 - late_jobs / jobs}, jobs)
        self._episodes.add({"wait_share": wait_share}, len(episodes["reward"]))
        self._updates.add({"explained_variance": ppo.get("explained_variance")})
        for row in rows or []:
            self._episodes.add({name: row.get(name) for name in PLAN_METRIC_NAMES})
        if self.settings["save_train_instances"] and rows:
            self.logger.save_instance_results(rows, "train", step)
        every = self.settings["log_every_updates"]
        if step // every > self._last_train_step // every:
            self.write_train(learning_rate, step, processed_instances, elapsed_seconds)

    def write_train(self, learning_rate: float, step: int, processed_instances: int, elapsed_seconds: float) -> None:
        """Schreibt, was seit dem letzten Mal aufgelaufen ist; auch vor einem Checkpoint, damit nichts verloren geht."""
        self._write_updates(step)
        if not self._episodes:
            return
        means, stds = self._episodes.means(), self._episodes.stds()
        self.logger.log_train_metrics(
            {"reward_mean": means["reward"], "reward_std": stds["reward"], "makespan_mean": means["makespan"],
             "tardiness_mean": means["tardiness"], "window_compliance": means["window_compliance"],
             "feasibility_rate": means["feasible"], "episode_length": means["episode_length"],
             "wait_share": means["wait_share"], "learning_rate": learning_rate,
             "processed_instances": processed_instances, "elapsed_seconds": elapsed_seconds,
             **{f"{name}_mean": means[name] for name in PLAN_METRIC_NAMES if name in means}}, step)
        self._episodes.reset()
        self._last_train_step = step

    def after_validation(self, results: list, step: int, final: bool = False) -> None:
        """results: Ergebnisse je Instanz mit row, schedule, waits, instance und start_times."""
        if not self.logger.enabled:
            self.validations += 1
            return
        self.validations += 1
        rows = [result.row for result in results]
        self.logger.log_validation_metrics(summarize_instances(rows), step)
        self.logger.save_instance_results(rows, "val", step)
        if self._save_schedules_for("val"):
            self.logger.save_schedules(results, "val", step)
        if final or self.validations % self.settings["temporal_every_validations"] == 0:
            self._write_temporal(self.logger.log_temporal_metrics, results, step)
        if final or (self.validations - 1) % self.settings["gantt_every_validations"] == 0:
            for result in results:
                instance_id = result.row["instance_id"]
                if instance_id in self.gantt_ids and result.row["feasible"]:
                    self.logger.log_gantt(result.schedule, instance_id, step, result.instance, result.waits)
                elif instance_id in self.gantt_ids:
                    self.logger.mark_unavailable(f"gantt/{instance_id}", f"kein zulaessiger Plan bei Schritt {step}")
        self.logger.flush()

    def _write_temporal(self, write, results: list, step: int) -> None:
        rows = [result.row for result in results]
        starts = [minute for result in results for minute in result.start_times]
        metrics = {**summarize_temporal(rows), "start_times": starts}
        if self.static_rows:
            metrics.update(gain_metrics(rows, self.static_rows))
        write(metrics, step)

    def after_test(self, results: list, step: int) -> None:
        """Volle Testauswertung nach dem Training."""
        if not self.logger.enabled:
            return
        rows = [result.row for result in results]
        self.logger.save_instance_results(rows, "test", step, replace=True)
        if self._save_schedules_for("test"):
            self.logger.save_schedules(results, "test", step, replace=True)
        self.logger.log_test_metrics(summarize_instances(rows), step)
        self.logger.log_test_metrics({f"temporal/{key}": value for key, value in summarize_temporal(rows).items()},
                                     step)
        self.logger.flush()

    def _save_schedules_for(self, split: str) -> bool:
        """Opt-in; test-only by default to avoid full plans at every validation."""
        return bool(self.settings.get("save_schedules", False)) and split in self.settings.get("schedule_splits", ["test"])
