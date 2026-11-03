"""Ergebnisse je Instanz fuer Validierung, Test und Training: eine Zeile mit allen Kennzahlen, dazu Plan und
freiwillige Wartezeiten. instance_id = <Klasse>/<Dateiname>, damit Modelle dieselben Instanzen vergleichen.

Grundlage der Kennzahlen:
    reward, makespan, total_tardiness, late_jobs   Umgebung (env.get_reward, env.get_objectives)
    feasible                                       unabhaengiger Validator (evaluation.evaluate)
    Zeiten, Faktoren, Leerlauf                     fertiger Plan (monitoring.schedule_metrics)
    inference_time                                 Laufzeit der Episoden im Batch / Anzahl Instanzen
Abgebrochene Episoden (truncated) haben keinen vollstaendigen Plan; ihre Plan-Kennzahlen bleiben leer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from baselines.dispatcher import run_dispatcher
from evaluation import evaluate, makespan
from monitoring.schedule_metrics import start_shares, temporal_variability, time_metrics
from rl.env import TDFJSPEnv
from rl.generators import TDFJSPFileGenerator
from rl.policies import Policy, rollout
from schedule_validator import all_resources
from simulator import SimulationError, execution_instance, view_config


@dataclass
class InstanceResult:
    row: dict
    schedule: list[dict]
    waits: list[dict]
    instance: dict
    start_times: list[int]


class _WaitRecorder:
    """Policy-Huelle: zaehlt Entscheidungen und haelt freiwillige Wartezeiten je Ressource fest.

    Solange ein Paar starten koennte, geht die Uhr nur durch WAIT weiter; sonst springt die Umgebung selbst.
    Freiwillig wartet also genau eine Ressource, die bei WAIT ein startbares Paar hat. Das Intervall endet am
    naechsten Ereignis, denn dort beginnt ein neuer Zustand mit einer neuen Entscheidung."""

    def __init__(self, env: TDFJSPEnv, policy: Policy, batch: int):
        self.env = env
        self.policy = policy
        self.decisions = [0] * batch
        self.wait_actions = [0] * batch
        self.intervals: list[list[tuple[int, int, int]]] = [[] for _ in range(batch)]

    def __call__(self, td: dict[str, torch.Tensor]) -> torch.Tensor:
        action = self.policy(td)
        running = ~td["done"]
        for index in running.nonzero().squeeze(1).tolist():
            self.decisions[index] += 1
        waiting = (action == 0) & running
        if waiting.any():
            event = self.env.wait_target(td)
            startable = td["pair_mask"].any(1)
            for index in waiting.nonzero().squeeze(1).tolist():
                self.wait_actions[index] += 1
                now, until = int(td["time"][index]), int(event[index])
                for resource in startable[index].nonzero().squeeze(1).tolist():
                    self.intervals[index].append((resource, now, until))
        return action


def _merged_waits(intervals: list[tuple[int, int, int]], instance: dict) -> list[dict]:
    """Aufeinanderfolgende WAIT-Intervalle einer Ressource zu einem Eintrag zusammenfassen."""
    names = all_resources(instance)
    merged: list[dict] = []
    for resource, start, end in sorted(intervals):
        if merged and merged[-1]["resource"] == names[resource] and merged[-1]["end"] == start:
            merged[-1]["end"] = end
        else:
            merged.append({"resource": names[resource], "start": start, "end": end})
    return merged


def _base_row(instance: dict, instance_id: str, class_name: str, env_view: str) -> dict:
    return {"instance_id": instance_id, "instance_class": class_name, "instance_seed": instance["seed"],
            "instance_size": len(instance["operations"]), "jobs": len(instance["jobs"]),
            "resources": len(all_resources(instance)), "env_view": env_view,
            "reference_makespan": makespan(instance["reference_schedule"])}


def evaluate_instances(env: TDFJSPEnv, policy: Policy, instances: list[dict], instance_ids: list[str],
                       class_name: str) -> list[InstanceResult]:
    """Spielt die Instanzen als ein Batch und misst je Instanz; funktioniert mit jeder Policy."""
    state = env.load_instances(instances)
    recorder = _WaitRecorder(env, policy, len(instances))
    start = time.perf_counter()
    td, _ = rollout(env, state, recorder)
    elapsed = (time.perf_counter() - start) / len(instances)
    rewards = env.get_reward(td)
    objectives = env.get_objectives(td)
    rules = view_config(env.config, env.view)

    results = []
    for index, (instance, instance_id) in enumerate(zip(instances, instance_ids)):
        truncated = bool(td["truncated"][index])
        jobs = len(instance["jobs"])
        late_jobs = int(objectives["late_jobs"][index])
        schedule = [] if truncated else env.to_schedule(td, index, instance)
        waits = [] if truncated else _merged_waits(recorder.intervals[index], instance)
        row = {**_base_row(instance, instance_id, class_name, env.view),
               "temporal_variability": temporal_variability(env.config, instance),   # immer die echten Faktoren
               "reward": float(rewards[index]),
               "makespan": int(objectives["makespan"][index]),
               "total_tardiness": int(objectives["total_tardiness"][index]),
               "max_tardiness": int(objectives["max_tardiness"][index]),
               "late_jobs": late_jobs,
               "window_compliance": 1 - late_jobs / jobs,
               "truncated": truncated,
               "feasible": False,
               "wait_actions": recorder.wait_actions[index],
               "decisions": recorder.decisions[index],
               "inference_time": elapsed,
               "inference_batch_size": len(instances)}
        if not truncated:
            row["feasible"] = evaluate(rules, execution_instance(env.config, instance), schedule)["feasible"]
            voluntary = sum(end - start for _, start, end in recorder.intervals[index])
            row.update(time_metrics(rules, instance, schedule, voluntary))
            row.update(start_shares(rules, schedule))
        results.append(InstanceResult(row, schedule, waits, instance, [e["start"] for e in schedule]))
    return results


def evaluate_class(env: TDFJSPEnv, policy: Policy, generator: TDFJSPFileGenerator, class_name: str,
                   count: int | None = None, batch_size: int | None = None) -> list[InstanceResult]:
    """Die ersten `count` Instanzen einer Klasse in Dateireihenfolge, also immer dieselben."""
    instances = generator.instances[:count]
    ids = [f"{class_name}/{path.stem}" for path in generator.files[:count]]
    size = batch_size or len(instances)
    results = []
    for first in range(0, len(instances), size):
        results += evaluate_instances(env, policy, instances[first:first + size], ids[first:first + size], class_name)
    return results


def rule_rows(config: dict, generator: TDFJSPFileGenerator, class_name: str, count: int | None,
              rule: dict) -> list[dict]:
    """Dieselben Instanzen mit einer Dispatching-Regel, ausgefuehrt mit den echten Dauern."""
    rows = []
    for path, instance in zip(generator.files[:count], generator.instances[:count]):
        row = _base_row(instance, f"{class_name}/{path.stem}", class_name, "time_aware")
        row["temporal_variability"] = temporal_variability(config, instance)
        start = time.perf_counter()
        try:
            executed = run_dispatcher(config, instance, rule["sequencing"], rule["routing"], rule["view"])["executed"]
        except SimulationError:
            rows.append({**row, "truncated": True, "feasible": False})
            continue
        result = evaluate(config, execution_instance(config, instance), executed)
        row.update(makespan=result["makespan"], total_tardiness=result["total_tardiness"],
                   max_tardiness=result["max_tardiness"], late_jobs=result["late_jobs"],
                   window_compliance=1 - result["late_jobs"] / result["jobs"], truncated=False,
                   feasible=result["feasible"], inference_time=time.perf_counter() - start, inference_batch_size=1)
        row.update(time_metrics(config, instance, executed, None))
        rows.append(row)
    return rows


def train_rows(env: TDFJSPEnv, td: dict[str, torch.Tensor], wait_actions: torch.Tensor,
               decisions: torch.Tensor, class_name: str) -> list[dict]:
    """Zeilen je Trainingsinstanz, direkt aus dem Endzustand; Kennzahlen, die den Plan brauchen, bleiben leer."""
    objectives = {key: value.tolist() for key, value in env.get_objectives(td).items()}
    rewards = env.get_reward(td).tolist()
    scheduled = td["op_scheduled"]
    processing = ((td["op_end"] - td["op_start"]) * scheduled).sum(1).tolist()
    setup = ((td["op_start"] - td["op_decision"]) * scheduled).sum(1).tolist()
    jobs = td["due"].shape[1]
    operations = td["op_kind"].shape[1]
    return [{"instance_id": f"{class_name}/seed_{seed}", "instance_class": class_name, "instance_seed": seed,
             "instance_size": operations, "jobs": jobs, "env_view": env.view, "reward": rewards[i],
             "makespan": objectives["makespan"][i], "total_tardiness": objectives["total_tardiness"][i],
             "max_tardiness": objectives["max_tardiness"][i], "late_jobs": objectives["late_jobs"][i],
             "window_compliance": 1 - objectives["late_jobs"][i] / jobs, "truncated": objectives["truncated"][i],
             "feasible": not objectives["truncated"][i], "processing_time_total": processing[i],
             "setup_time_total": setup[i], "wait_actions": int(wait_actions[i]), "decisions": int(decisions[i])}
            for i, seed in enumerate(td["seed"].tolist())]
