"""PPO-Training; alle Einstellungen kommen aus problem_config.json (Abschnitte environment, reward,
model, training, monitoring). Die Optionen ueberschreiben nur die Auswahl fuer die Versuchsreihe.

Aufruf (aus dem Projektordner):
    python -m rl.train
    python -m rl.train --variant A1 --seed 1 --iterations 500
    python -m rl.train --variant A3 --no-time-information     Agent ohne Zeitinformation
    python -m rl.train --variant A3 --no-wait                 ohne freiwilliges Warten
    python -m rl.train --classes mittel_16 gross_21           nur diese Instanzklassen
    python -m rl.train --no-monitoring                        ohne Experiment-Tracking
    python -m rl.train --resume experiments/checkpoints/<Laufname>/latest.pt --iterations 1000

Ergebnisse in training.output_folder/<Laufname>/:
    config.json          die vollstaendigen Einstellungen des Laufs
    training_log.csv     eine Zeile je Iteration
    latest.pt, best.pt   Modell, Optimizer, Fortschritt, Zufallszustand, Einstellungen; best.pt hat den
                         besten Wert von training.best_metric
Ein zweiter Lauf mit gleichem Namen bekommt die Endung _r2, _r3, ...
Kurven, Kennzahlen je Instanz und Gantt-Diagramme: siehe monitoring/experiment_logger.py.

global_step zaehlt Optimizer-Schritte und ist die x-Achse aller Kurven.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from monitoring.experiment_logger import ExperimentLogger, NullLogger, reserve_run_id
from monitoring.metrics_aggregator import summarize_instances
from monitoring.training_monitor import TrainingMonitor
from rl.env import TDFJSPEnv
from rl.generators import TDFJSPFileGenerator
from rl.instance_results import InstanceResult, evaluate_class, rule_rows, train_rows
from rl.observation import ObservationBuilder
from rl.policy import VARIANTS, ActorCritic, NeuralPolicy, build_model, model_name, run_name
from rl.ppo import collect_rollout, ppo_update
from schedule_validator import BEST_METRICS, DEFAULT_CONFIG, load_config, resolve_path, selected_classes

LOG_COLUMNS = ("iteration", "global_step", "instance_class", "episode_reward", "total_tardiness", "late_share",
               "makespan", "truncated_share", "wait_share", "policy_loss", "value_loss", "entropy",
               "approx_kl", "clip_fraction", "grad_norm", "explained_variance",
               "val_cost", "val_total_tardiness", "val_late_share", "val_makespan", "val_truncated_share",
               "seconds")
# Trainingsseeds liegen so weit auseinander, dass sich ihre Instanzen nicht wiederholen
_SEEDS_PER_RUN = 10_000_000


@dataclass
class Progress:
    """Stand des Trainings; steht in jedem Checkpoint."""

    iteration: int = 0
    global_step: int = 0
    processed_instances: int = 0
    elapsed_seconds: float = 0.0
    validations: int = 0
    best_value: float | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _random_state() -> dict:
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_random_state(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_checkpoint(state: dict, path: Path) -> None:
    """Erst in eine Hilfsdatei, damit ein Abbruch nie einen halben Checkpoint hinterlaesst."""
    helper = path.with_name(path.name + ".tmp")
    torch.save(state, helper)
    os.replace(helper, path)


def validate(env: TDFJSPEnv, model: ActorCritic, files: dict[str, TDFJSPFileGenerator], count: int,
             decode: str = "greedy") -> list[InstanceResult]:
    """Auf denselben festen Instanzen je Klasse, ohne Gradienten; beruehrt den Zufallszustand des Trainings nicht."""
    policy = NeuralPolicy(env, model, decode)
    results = []
    for class_name, generator in files.items():
        results += evaluate_class(env, policy, generator, class_name, count)
    return results


def _trim_training_log(path: Path, iteration: int) -> None:
    """Fortsetzung: Zeilen nach dem Checkpoint entfernen."""
    with path.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if int(row["iteration"]) <= iteration]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _run_metadata(config: dict, seed: int, iterations: int) -> dict:
    temporal, edge_values, time_aware = VARIANTS[config["model"]["variant"]]
    settings = config["training"]
    return {"experiment_id": model_name(config), "model": model_name(config), "variant": config["model"]["variant"],
            "seed": seed, "algorithm": "ppo", "iterations": iterations,
            "time_information": config["model"]["time_information"], "allow_wait": config["environment"]["allow_wait"],
            "environment_view": config["environment"]["view"], "train_classes": config["environment"]["train_classes"],
            "hyperparameters": settings, "model_settings": config["model"],
            "encoder": {"name": "HetGNNEncoder", "temporal": temporal, "edge_values": edge_values},
            "decoder": {"name": "TimeAwareDecoder" if time_aware else "StaticPairDecoder"},
            "objective": config["reward"], "instance_generator": config["generation"],
            "instance_classes": config["instance_classes"], "time_profiles": config["time_model"]}


def _open_run(config: dict, seed: int, iterations: int, config_path: Path, checkpoint: dict | None,
              resume: Path | None) -> tuple[Path, str, ExperimentLogger | NullLogger]:
    """Ausgabeordner, Laufname und Logger; ein neuer Lauf bekommt einen freien Namen."""
    settings = config["monitoring"]
    runs = resolve_path(config["training"]["output_folder"], config_path)
    root = resolve_path(settings["folder"], config_path)
    if checkpoint is not None:
        name = checkpoint["name"]
    elif settings["enabled"]:
        name = reserve_run_id(run_name(config, seed), [root / "results", root / "tensorboard", root / "figures", runs])
    else:
        name = reserve_run_id(run_name(config, seed), [runs])
    output = runs / name if resume is None else resume.parent
    output.mkdir(parents=True, exist_ok=True)
    if not settings["enabled"]:
        return output, name, NullLogger()
    logger = ExperimentLogger(name, root, config, _run_metadata(config, seed, iterations), reserved=True,
                              resume_step=None if checkpoint is None else checkpoint["progress"]["global_step"],
                              flush_seconds=settings["flush_seconds"],
                              gantt_to_tensorboard=settings["gantt_to_tensorboard"])
    return output, name, logger


def _make_monitor(config: dict, logger, files: dict[str, TDFJSPFileGenerator], progress: Progress,
                  ) -> TrainingMonitor:
    """Feste Gantt-Instanzen und die statische Referenz auf der Validierung, einmal am Anfang."""
    settings = config["monitoring"]
    if not logger.enabled:
        return TrainingMonitor(logger, settings, progress.validations, progress.global_step)
    count = config["training"]["validation_instances"]
    static_rows = []
    if settings["static_reference"] is not None:
        for class_name, generator in files.items():
            static_rows += rule_rows(config, generator, class_name, count, settings["static_reference"])
        if progress.validations == 0:
            logger.save_instance_results(static_rows, "val_static_reference", 0)
    # Reihum je Klasse die erste, dann die zweite Instanz, ...; immer dieselben
    ordered = [f"{class_name}/{generator.files[position].stem}"
               for position in range(count) for class_name, generator in files.items()
               if position < len(generator.files)]
    return TrainingMonitor(logger, settings, progress.validations, progress.global_step, static_rows,
                           tuple(ordered[:settings["gantt_instances"]]))


def _final_test(config: dict, config_path: Path, env: TDFJSPEnv, model: ActorCritic, output: Path, logger,
                monitor: TrainingMonitor, progress: Progress) -> None:
    """Bestes Modell auf dem ganzen Testdatensatz, in der Batch-Groesse der uebrigen Auswertungen."""
    settings = config["monitoring"]
    if not (logger.enabled and settings["test_after_training"] and (output / "best.pt").exists()):
        return
    model.load_state_dict(torch.load(output / "best.pt", map_location="cpu", weights_only=False)["model"])
    policy = NeuralPolicy(env, model)
    results = []
    for class_name in selected_classes(config, config["environment"]["train_classes"]):
        try:
            generator = TDFJSPFileGenerator(config, class_name, "test", config_path)
        except FileNotFoundError as exc:
            logger.mark_unavailable(f"test/{class_name}", str(exc))
            continue
        results += evaluate_class(env, policy, generator, class_name, settings["test_instances"],
                                  config["experiments"]["evaluation_batch_size"])
    if results:
        monitor.after_test(results, progress.global_step)


def train(config: dict, seed: int, iterations: int, config_path: Path, resume: Path | None = None) -> Path:
    """Fuehrt ein Training aus und gibt den Ausgabeordner zurueck. resume: Checkpoint, an dem es weitergeht;
    Einstellungen und Seed kommen dann aus dem Checkpoint, iterations ist die neue Gesamtzahl."""
    checkpoint = None if resume is None else torch.load(resume, map_location="cpu", weights_only=False)
    if checkpoint is not None:
        config, seed = checkpoint["config"], checkpoint["seed"]
    settings = config["training"]
    set_seed(seed)
    config = {**config, "environment": {**config["environment"], "compute_features": False}}
    env = TDFJSPEnv(config, config_path=config_path)
    env.generator.next_seed += seed * _SEEDS_PER_RUN
    model = build_model(env)
    builder = ObservationBuilder(env)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    progress = Progress() if checkpoint is None else Progress(**checkpoint["progress"])
    # Vor dem Anlegen des Laufs: Fehlen die Instanzen, bleibt kein leerer Lauf zurueck
    validation_classes = selected_classes(config, config["environment"]["train_classes"])
    files = {c: TDFJSPFileGenerator(config, c, "val", config_path) for c in validation_classes}

    output, name, logger = _open_run(config, seed, iterations, config_path, checkpoint, resume)
    logger.update_metadata(iterations=iterations)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    monitor = _make_monitor(config, logger, files, progress)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        env.generator.next_seed = checkpoint["generator_seed"]
        env.generator.skipped_seeds = checkpoint["skipped_seeds"]
        _trim_training_log(output / "training_log.csv", progress.iteration)
        _restore_random_state(checkpoint["random_state"])

    def on_update(metrics: dict[str, float], samples: int) -> None:
        progress.global_step += 1
        monitor.on_update(metrics, samples, progress.global_step)

    best_metric = settings["best_metric"]
    status = "failed"
    try:
        with (output / "training_log.csv").open("a" if checkpoint else "w", newline="", encoding="utf-8") as log_file:
            log = csv.DictWriter(log_file, fieldnames=LOG_COLUMNS, restval="")
            if checkpoint is None:
                log.writeheader()
            for iteration in range(progress.iteration + 1, iterations + 1):
                start = time.perf_counter()
                decay_steps = settings.get("learning_rate_decay_iterations", 0)
                if decay_steps:
                    fraction = min((iteration - 1) / max(decay_steps - 1, 1), 1.0)
                    end_fraction = settings.get("learning_rate_end_fraction", 0.2)
                    rate = settings["learning_rate"] * (1 - fraction * (1 - end_fraction))
                    for group in optimizer.param_groups:
                        group["lr"] = rate
                class_name = env.generator.classes[(iteration - 1) % len(env.generator.classes)]
                td = env.reset(env.generator(settings["instances_per_iteration"], class_name))
                data = collect_rollout(env, model, builder, td, settings)
                losses = ppo_update(model, optimizer, data, settings, on_update)
                progress.iteration = iteration
                progress.processed_instances += settings["instances_per_iteration"]

                objectives = env.get_objectives(data.final_state)
                rewards = env.get_reward(data.final_state)
                jobs = data.final_state["due"].shape[1] * settings["instances_per_iteration"]
                row = {"iteration": iteration, "global_step": progress.global_step, "instance_class": class_name,
                       "episode_reward": float(rewards.mean()),
                       "total_tardiness": float(objectives["total_tardiness"].to(torch.float32).mean()),
                       "late_share": float(objectives["late_jobs"].sum()) / jobs,
                       "makespan": float(objectives["makespan"].to(torch.float32).mean()),
                       "truncated_share": float(objectives["truncated"].to(torch.float32).mean()),
                       "wait_share": data.wait_share, **losses}
                learning_rate = optimizer.param_groups[0]["lr"]
                monitor.after_iteration(
                    {"reward": rewards.tolist(), "makespan": objectives["makespan"].tolist(),
                     "tardiness": objectives["total_tardiness"].tolist(),
                     "feasible": (~objectives["truncated"]).tolist(), "episode_length": data.decisions.tolist()},
                    jobs, int(objectives["late_jobs"].sum()), losses, learning_rate, progress.global_step,
                    progress.processed_instances, progress.elapsed_seconds + time.perf_counter() - start,
                    data.wait_share,
                    train_rows(env, data.final_state, data.wait_actions, data.decisions, class_name)
                    if logger.enabled else None)

                if iteration % settings["validate_every"] == 0 or iteration == iterations:
                    monitor.write_train(learning_rate, progress.global_step, progress.processed_instances,
                                        progress.elapsed_seconds + time.perf_counter() - start)
                    results = validate(env, model, files, settings["validation_instances"],
                                       settings["validation_decode"])
                    summary = summarize_instances([result.row for result in results])
                    row.update(val_cost=-summary["reward_mean"], val_total_tardiness=summary["tardiness_mean"],
                               val_late_share=1 - summary["window_compliance"], val_makespan=summary["makespan_mean"],
                               val_truncated_share=sum(r.row["truncated"] for r in results) / len(results))
                    monitor.after_validation(results, progress.global_step, final=iteration == iterations)
                    progress.validations = monitor.validations

                    value = summary[best_metric.removeprefix("val/")]
                    better = (progress.best_value is None
                              or (value > progress.best_value if BEST_METRICS[best_metric] else value < progress.best_value))
                    if better:
                        progress.best_value = value
                    schedule = ({"kind": "linear_by_iteration", "iterations": settings["learning_rate_decay_iterations"],
                                 "end_fraction": settings.get("learning_rate_end_fraction", 0.2)}
                                if settings.get("learning_rate_decay_iterations", 0) else None)
                    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": schedule,
                             "config": config, "name": name, "seed": seed, "iteration": iteration,
                             "global_step": progress.global_step, "validation": summary, "best_metric": best_metric,
                             "progress": {**asdict(progress),
                                          "elapsed_seconds": progress.elapsed_seconds + time.perf_counter() - start},
                             "generator_seed": env.generator.next_seed,
                             "skipped_seeds": env.generator.skipped_seeds, "random_state": _random_state()}
                    _save_checkpoint(state, output / "latest.pt")
                    if better:
                        _save_checkpoint(state, output / "best.pt")
                        logger.update_metadata(best={"metric": best_metric, "value": value, "iteration": iteration,
                                                     "global_step": progress.global_step, "path": str(output / "best.pt"),
                                                     "validation": summary})

                row["seconds"] = time.perf_counter() - start
                progress.elapsed_seconds += row["seconds"]
                log.writerow({key: round(value, 5) if isinstance(value, float) else value
                              for key, value in row.items()})
                log_file.flush()
                print(f"{name} {iteration:>5}/{iterations} {class_name:<10} Reward {row['episode_reward']:8.4f}  "
                      f"Verspaetung {row['total_tardiness']:7.1f}  Warten {100 * row['wait_share']:4.1f} %  "
                      f"{row['seconds']:5.1f} s"
                      + (f"  | Val Kosten {row['val_cost']:.4f}  Verspaetung {row['val_total_tardiness']:.1f}"
                         if "val_cost" in row else ""), flush=True)
        _final_test(config, config_path, env, model, output, logger, monitor, progress)
        status = "finished"
    finally:
        env.generator.close()
        logger.close(status)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="PPO-Training der RL-Umgebung")
    parser.add_argument("--variant", choices=["A0", "A1", "A2", "A3"], default=None,
                        help="Modellvariante (Standard: model.variant)")
    information = parser.add_mutually_exclusive_group()
    information.add_argument("--no-time-information", action="store_true",
                        help="Agent ohne Zeitinformation (model.time_information = none)")
    information.add_argument("--mean-time-profiles", action="store_true",
                            help="Kontrollmodell mit konstanten mittleren Faktoren; Uhrzeit und Deadlines bleiben bekannt")
    parser.add_argument("--no-wait", action="store_true", help="ohne freiwilliges Warten")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Instanzklassen (Standard: environment.train_classes)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Ausgabeordner (Standard: training.output_folder)")
    parser.add_argument("--seed", type=int, default=None, help="Standard: training.seed")
    parser.add_argument("--iterations", type=int, default=None, help="Standard: training.iterations")
    parser.add_argument("--no-monitoring", action="store_true", help="ohne Experiment-Tracking")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Checkpoint, an dem es weitergeht (latest.pt); Einstellungen kommen von dort")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Einstellungen (Standard: problem_config.json)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.variant:
        config["model"]["variant"] = args.variant
    if args.no_time_information:
        config["model"]["time_information"] = "none"
    if args.mean_time_profiles:
        config["model"]["time_information"] = "mean"
    if args.no_wait:
        config["environment"]["allow_wait"] = False
    if args.classes:
        config["environment"]["train_classes"] = args.classes
    if args.output:
        config["training"]["output_folder"] = str(args.output.resolve())
    if args.no_monitoring:
        config["monitoring"]["enabled"] = False
    seed = config["training"]["seed"] if args.seed is None else args.seed
    iterations = config["training"]["iterations"] if args.iterations is None else args.iterations
    output = train(config, seed, iterations, args.config, args.resume)
    print(f"\nFertig: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
