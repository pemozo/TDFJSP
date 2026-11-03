"""Policies in der Umgebung auswerten, auf denselben Instanzen wie die Dispatching-Regeln.

Der Plan einer Policy laeuft durch dieselbe Bewertung wie die Regeln: Zuordnung und Reihenfolge
werden mit den echten Dauern ausgefuehrt (simulator.execute_plan) und dann bewertet
(evaluation.evaluate). Beide landen in derselben Tabelle; der Gap bezieht sich auf alle
Verfahren zusammen.

Ausnahme: Hat eine Policy freiwillig gewartet und plant sie mit den echten Dauern, gilt ihr Plan
unveraendert. execute_plan startet jede Operation zum fruehesten Zeitpunkt und wuerde das Warten
zuruecknehmen, obwohl der Zeitpunkt hier selbst die Entscheidung ist.

Aufruf (aus dem Projektordner):
    python -m rl.evaluate --policies edd_ect random            alle Klassen aus der Config
    python -m rl.evaluate --policies random --classes klein_11 --split val --no-rules
    python -m rl.evaluate --policies edd_ect --classes klein_11 --limit 1 --show
    python -m rl.evaluate --checkpoints runs/A3_zeit_warten_seed0/best.pt runs/A1_zeit_warten_seed0/best.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from rl.env import TDFJSPEnv
from rl.policies import POLICY_NAMES, Policy, make_policy, rollout
from rl.policy import NeuralPolicy, load_checkpoint
from run_experiments import (RESULT_COLUMNS, SUMMARY_COLUMNS, VIEW_NAMES, class_files, instance_row,
                             print_summary, run_instance, score_plan, summarize, write_csv)
from schedule_validator import (DEFAULT_CONFIG, load_config, load_json, resolve_path,
                                selected_classes, validate_instance)
from simulator import SimulationError, execute_plan, plan_from_schedule


def evaluate_policy(env: TDFJSPEnv, policy: Policy, policy_name: str, class_name: str,
                    files: list[Path]) -> list[dict]:
    """Eine Ergebniszeile je Instanz, in denselben Spalten wie die Regeln."""
    config = env.config
    method = f"Policy {policy_name} {VIEW_NAMES[env.view]}"
    size = config["experiments"]["evaluation_batch_size"]
    rows = []
    for first in range(0, len(files), size):
        paths = files[first:first + size]
        instances = [load_json(path) for path in paths]
        for path, instance in zip(paths, instances):
            errors = validate_instance(config, instance)
            if errors:
                raise ValueError(f"{path}: {errors[0]}")

        waited = torch.zeros(len(instances), dtype=torch.bool, device=env.device)

        def counting_policy(state: dict[str, torch.Tensor]) -> torch.Tensor:
            nonlocal waited
            action = policy(state)
            waited = waited | ((action == 0) & ~state["done"])
            return action

        start = time.perf_counter()
        td, _ = rollout(env, env.load_instances(instances), counting_policy)
        plans = [env.to_schedule(td, i, instance) for i, instance in enumerate(instances)]
        planning = (time.perf_counter() - start) / len(instances)

        for i, (path, instance, planned) in enumerate(zip(paths, instances, plans)):
            row = instance_row(instance, path, class_name, method, policy_name, "-", env.view)
            start = time.perf_counter()
            try:
                if td["truncated"][i]:
                    raise SimulationError("Horizont reicht nicht")
                if waited[i] and env.view == "time_aware":
                    executed = planned
                else:
                    executed = execute_plan(config, instance, plan_from_schedule(planned))
            except SimulationError as exc:
                row.update(status=f"nicht ausfuehrbar: {exc}", feasible=False, runtime_s=planning)
                rows.append(row)
                continue
            row["runtime_s"] = planning + time.perf_counter() - start
            score_plan(config, instance, planned, executed, row)
            rows.append(row)
    return rows


def show_first_plan(env: TDFJSPEnv, policy: Policy, files: list[Path]) -> None:
    """Gantt-Diagramm der ersten Instanz."""
    import matplotlib.pyplot as plt

    instance = load_json(files[0])
    td, _ = rollout(env, env.load_instances([instance]), policy)
    env.render(td, 0, instance)
    plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(description="Policies in der RL-Umgebung auswerten")
    parser.add_argument("--policies", nargs="*", default=["edd_ect"], choices=POLICY_NAMES)
    parser.add_argument("--checkpoints", nargs="*", type=Path, default=[],
                        help="trainierte Modelle (best.pt aus rl/train.py), greedy ausgewertet")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Instanzklassen oder 'all' (Standard: experiments.classes)")
    parser.add_argument("--split", choices=["val", "test"], default=None,
                        help="Datensatz (Standard: experiments.split)")
    parser.add_argument("--limit", type=int, default=None,
                        help="nur die ersten N Instanzen je Klasse (Schnelltest)")
    parser.add_argument("--no-rules", action="store_true",
                        help="Dispatching-Regeln nicht mit auswerten")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Einstellungen (Standard: problem_config.json)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Ergebnisse je Instanz (Standard: experiments.policy_results_file)")
    parser.add_argument("--summary", type=Path, default=None,
                        help="Ergebnistabelle (Standard: experiments.policy_summary_file)")
    parser.add_argument("--show", action="store_true",
                        help="Gantt-Diagramm der ersten Instanz je Klasse und Policy")
    args = parser.parse_args()

    config = load_config(args.config)
    settings = config["experiments"]
    selection = settings["classes"] if args.classes is None else args.classes
    selection = "all" if selection == ["all"] else selection
    split = settings["split"] if args.split is None else args.split
    env = TDFJSPEnv(config, config_path=args.config)

    rows = []
    for class_name in selected_classes(config, selection):
        files = class_files(config, class_name, split, args.config, args.limit)
        if files is None:
            return 1
        if not args.no_rules:
            for path in files:
                rows += run_instance(config, path, class_name)
        for name in args.policies:
            policy = make_policy(name, settings["policy_seed"], env.device)
            rows += evaluate_policy(env, policy, name, class_name, files)
            if args.show:
                show_first_plan(env, policy, files)
        for checkpoint in args.checkpoints:
            trained_env, model, name = load_checkpoint(checkpoint, args.config)
            policy = NeuralPolicy(trained_env, model)
            rows += evaluate_policy(trained_env, policy, name, class_name, files)
            if args.show:
                show_first_plan(trained_env, policy, files)
        print(f"{class_name}: {len(files)} Instanzen fertig", flush=True)

    summary = summarize(rows)
    out = resolve_path(settings["policy_results_file"], args.config) if args.out is None else args.out
    summary_file = (resolve_path(settings["policy_summary_file"], args.config)
                    if args.summary is None else args.summary)
    write_csv(out, RESULT_COLUMNS, rows)
    write_csv(summary_file, SUMMARY_COLUMNS, summary)
    print_summary(summary)
    print(f"\nErgebnisse: {out}  |  Zusammenfassung: {summary_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
