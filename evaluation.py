"""Kennzahlen fuer Schedules: Makespan und Verspaetungen.

Bewertet einen Schedule, der die harten Regeln einhaelt (Pause, Nacht,
Ueberlappung, ...), und misst, wie stark er die Deadlines verfehlt. Bei
rules.due_dates_hard = false (Standard) sind Verspaetungen erlaubt und werden
hier gezaehlt; bei true macht schon eine Verspaetung den Schedule unzulaessig.

Haengt nur von schedule_validator.py ab und kann von Generator, Baselines,
Experimenten und RL-Umgebung gleichermassen genutzt werden.

Aufruf:
    python evaluation.py instances/test/klein_11                     (Referenz-Schedules)
    python evaluation.py instances/test/klein_11/instance_000.json --schedule mein_plan.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from schedule_validator import (DEFAULT_CONFIG, check_schedule, instance_files,
                                load_config, load_json, load_schedule, validate_instance)


# --------------------------------------------------------------------------
# Kennzahlen
# --------------------------------------------------------------------------

def job_completion(schedule: list[dict], job: dict) -> int:
    """Fertigstellungszeit eines Jobs im Schedule (Ende seiner letzten Operation)."""
    return max(e["end"] for e in schedule if e["op"] in job["operations"])


def makespan(schedule: list[dict]) -> int:
    """Ende der letzten Operation im Schedule."""
    return max(e["end"] for e in schedule)


def job_tardiness(instance: dict, schedule: list[dict]) -> dict[str, dict]:
    """Je Job: Fertigstellung, Deadline und Verspaetung (0, wenn puenktlich)."""
    result = {}
    for job in instance["jobs"]:
        completion = job_completion(schedule, job)
        result[job["id"]] = {"completion": completion, "due": job["due"],
                             "tardiness": max(0, completion - job["due"])}
    return result


def evaluate(config: dict, instance: dict, schedule: list[dict]) -> dict:
    """Zulaessigkeit und Kennzahlen eines Schedules.

    feasible   : True, wenn keine harte Regel verletzt ist
    violations : Liste der Regelverstoesse (leer = zulaessig)
    Zeiten in Arbeitsminuten ab Beginn von Tag 1."""
    violations = check_schedule(config, instance, schedule)
    per_job = job_tardiness(instance, schedule)
    tardiness = [j["tardiness"] for j in per_job.values()]
    return {
        "feasible": not violations,
        "violations": violations,
        "makespan": makespan(schedule),
        "jobs": len(per_job),
        "late_jobs": sum(t > 0 for t in tardiness),
        "total_tardiness": sum(tardiness),
        "max_tardiness": max(tardiness),
        "mean_tardiness": sum(tardiness) / len(tardiness),
        "per_job": per_job,
    }


# --------------------------------------------------------------------------
# Kommandozeile
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Schedules bewerten (Makespan, Verspaetung)")
    parser.add_argument("instances", type=Path, nargs="+",
                        help="Instanzdatei(en) oder Ordner")
    parser.add_argument("--schedule", type=Path,
                        help="Eigener Schedule (JSON-Liste); sonst reference_schedule")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Regeldatei (Standard: problem_config.json)")
    args = parser.parse_args()

    config = load_config(args.config)
    files = instance_files(args.instances)
    if args.schedule and len(files) != 1:
        parser.error("--schedule nur zusammen mit genau einer Instanz")

    print(f"{'Instanz':<40} {'zulaessig':>9} {'Makespan':>8} {'verspaetet':>10} "
          f"{'Summe':>6} {'Max':>5}")
    failed = 0
    for path in files:
        instance = load_json(path)
        errors = validate_instance(config, instance)
        if errors:
            failed += 1
            print(f"{str(path):<40} Instanz ungueltig: {errors[0]}")
            continue
        schedule = load_schedule(args.schedule) if args.schedule else instance["reference_schedule"]
        result = evaluate(config, instance, schedule)
        failed += not result["feasible"]
        print(f"{str(path):<40} {'ja' if result['feasible'] else 'NEIN':>9} "
              f"{result['makespan']:>8} {result['late_jobs']:>4} / {result['jobs']:<3} "
              f"{result['total_tardiness']:>6} {result['max_tardiness']:>5}")
        for violation in result["violations"]:
            print(f"        - {violation}")

    print("\nZeiten in Arbeitsminuten; Summe/Max = Verspaetung der Jobs.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
