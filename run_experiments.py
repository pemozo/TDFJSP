"""Experimente: Instanzklassen x Instanzen x Baselines -> results/*.csv

Klassen, Verfahren, Datensatz und Dateien stehen in problem_config.json (Abschnitte
instance_classes, data_splits, experiments).

Je Instanz und Verfahren:
    1. Plan bauen und mit echten Dauern ausfuehren (baselines.dispatcher.run_dispatcher)
    2. Ausfuehrung bewerten (evaluation.evaluate) gegen den Ausfuehrungshorizont
       (Horizont der Instanz + simulation.overflow_days)
    3. eine Zeile in results/baselines.csv
Danach je Klasse und Verfahren eine Zeile in results/baselines_summary.csv.

Kennzahlen je Klasse und Verfahren:
    verspaetete Jobs : Summe verspaeteter Jobs ueber alle Instanzen / Summe aller
                       Jobs ueber alle Instanzen (late_share). Optimum 0, weil der
                       Referenz-Plan jeder Instanz alle Deadlines einhaelt.
    max. Verspaetung : groesste Verspaetung je Instanz (Minuten), gemittelt ueber
                       die Instanzen (max_tardiness_mean)
    Makespan         : Mittel ueber die Instanzen (makespan_mean)
    Gap Makespan     : Abstand in % zum besten (kleinsten) makespan_mean dieser
                       Klasse in der Tabelle (best_makespan_mean)
    Rechenzeit       : Planen + Ausfuehren je Instanz, gemittelt (Sekunden)
Zusaetzlich: Verspaetungen und Makespan, die das Verfahren selbst erwartet
(planned_*). Beim statischen Verfahren zeigt der Abstand zur Ausfuehrung,
wie stark es sich verschaetzt.

Aufruf:
    python run_experiments.py                         alle Klassen, alle Instanzen
    python run_experiments.py --classes klein_11      nur eine Klasse
    python run_experiments.py --split val             Validierungs- statt Testinstanzen
    python run_experiments.py --limit 5               je Klasse nur 5 Instanzen (Schnelltest)
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

from baselines.dispatcher import run_dispatcher
from baselines.rules import ROUTING_RULES, SEQUENCING_RULES
from evaluation import evaluate, job_tardiness, makespan
from schedule_validator import (DEFAULT_CONFIG, instance_files, load_config, load_json,
                                resolve_path, selected_classes, split_folder, validate_instance)
from simulator import DURATION_VIEWS, SimulationError, execution_instance

VIEW_NAMES = {"static": "statisch", "time_aware": "zeitbewusst"}

RESULT_COLUMNS = [
    "class", "instance", "seed", "method", "sequencing", "routing", "view", "status",
    "feasible", "jobs", "late_jobs", "late_share", "total_tardiness", "max_tardiness",
    "makespan", "reference_makespan", "planned_late_jobs", "planned_makespan", "runtime_s",
]
SUMMARY_COLUMNS = [
    "class", "method", "instances", "feasible", "jobs_total", "late_jobs_total", "late_share",
    "instances_without_late", "max_tardiness_mean", "total_tardiness_mean", "makespan_mean",
    "best_makespan_mean", "makespan_gap_pct", "planned_late_share", "runtime_s_mean",
]


def method_name(sequencing: str, routing: str, view: str) -> str:
    """Anzeigename, z. B. 'EDD+ECT statisch'."""
    return f"{sequencing}+{routing} {VIEW_NAMES[view]}"


# --------------------------------------------------------------------------
# Eine Instanz
# --------------------------------------------------------------------------

def methods(config: dict) -> list[tuple[str, str, str]]:
    """Verfahren aus experiments.methods: (Reihenfolge-Regel, Zuordnungs-Regel, Dauer-Sicht)."""
    result = []
    for method in config["experiments"]["methods"]:
        sequencing, routing, view = method["sequencing"], method["routing"], method["view"]
        if sequencing not in SEQUENCING_RULES or routing not in ROUTING_RULES:
            raise ValueError(f"Unbekannte Regel in experiments.methods: {method}")
        if view not in DURATION_VIEWS:
            raise ValueError(f"Unbekannte Sicht in experiments.methods: {method}")
        result.append((sequencing, routing, view))
    return result


def instance_row(instance: dict, path: Path, class_name: str, method: str, sequencing: str,
                 routing: str, view: str) -> dict:
    """Beginn einer Ergebniszeile; auch fuer Policies in rl/evaluate.py."""
    return {"class": class_name, "instance": path.name, "seed": instance["seed"],
            "method": method, "sequencing": sequencing, "routing": routing, "view": view,
            "jobs": len(instance["jobs"]),
            "reference_makespan": makespan(instance["reference_schedule"])}


def score_plan(config: dict, instance: dict, planned: list[dict], executed: list[dict],
               row: dict) -> None:
    """Ausgefuehrten Plan bewerten und die Kennzahlen in die Zeile schreiben."""
    evaluation = evaluate(config, execution_instance(config, instance), executed)
    expected = job_tardiness(instance, planned)
    row.update(
        status="ok" if evaluation["feasible"] else f"Regelverstoss: {evaluation['violations'][0]}",
        feasible=evaluation["feasible"],
        late_jobs=evaluation["late_jobs"],
        late_share=evaluation["late_jobs"] / evaluation["jobs"],
        total_tardiness=evaluation["total_tardiness"],
        max_tardiness=evaluation["max_tardiness"],
        makespan=evaluation["makespan"],
        planned_late_jobs=sum(job["tardiness"] > 0 for job in expected.values()),
        planned_makespan=makespan(planned),
    )


def run_instance(config: dict, path: Path, class_name: str) -> list[dict]:
    """Alle Verfahren auf einer Instanz; eine Ergebniszeile je Verfahren."""
    instance = load_json(path)
    errors = validate_instance(config, instance)
    if errors:
        raise ValueError(f"{path}: {errors[0]}")

    rows = []
    for sequencing, routing, view in methods(config):
        row = instance_row(instance, path, class_name, method_name(sequencing, routing, view),
                           sequencing, routing, view)
        start = time.perf_counter()
        try:
            result = run_dispatcher(config, instance, sequencing, routing, view)
        except SimulationError as exc:
            row.update(status=f"nicht ausfuehrbar: {exc}", feasible=False,
                       runtime_s=time.perf_counter() - start)
            rows.append(row)
            continue
        row["runtime_s"] = time.perf_counter() - start

        score_plan(config, instance, result["planned"], result["executed"], row)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Zusammenfassung
# --------------------------------------------------------------------------

def summarize(rows: list[dict]) -> list[dict]:
    """Kennzahlen je Klasse und Verfahren, in der Reihenfolge der Zeilen.

    Gezaehlt und gemittelt wird nur ueber ausfuehrbare Loesungen (feasible).
    Der Gap bezieht sich auf den besten makespan_mean der Klasse in dieser
    Tabelle; kommt ein Verfahren hinzu, kann sich diese Referenz aendern."""
    summary = []
    classes = list(dict.fromkeys(r["class"] for r in rows))
    method_names = list(dict.fromkeys(r["method"] for r in rows))
    for class_name in classes:
        class_summary = []
        for name in method_names:
            group = [r for r in rows if r["class"] == class_name and r["method"] == name]
            ok = [r for r in group if r["feasible"]]
            if not group:
                continue
            mean = lambda key: statistics.mean(r[key] for r in ok) if ok else None
            jobs_total = sum(r["jobs"] for r in ok)
            late_total = sum(r["late_jobs"] for r in ok)
            planned_late_total = sum(r["planned_late_jobs"] for r in ok)
            class_summary.append({
                "class": class_name, "method": name, "instances": len(group),
                "feasible": len(ok),
                "jobs_total": jobs_total,
                "late_jobs_total": late_total,
                "late_share": late_total / jobs_total if ok else None,
                "instances_without_late": sum(r["late_jobs"] == 0 for r in ok),
                "max_tardiness_mean": mean("max_tardiness"),
                "total_tardiness_mean": mean("total_tardiness"),
                "makespan_mean": mean("makespan"),
                "planned_late_share": planned_late_total / jobs_total if ok else None,
                "runtime_s_mean": mean("runtime_s"),
            })

        # Gap zum besten mittleren Makespan dieser Klasse in der Tabelle
        means = [s["makespan_mean"] for s in class_summary if s["feasible"]]
        best = min(means) if means else None
        for s in class_summary:
            s["best_makespan_mean"] = best
            if s["feasible"]:
                s["makespan_gap_pct"] = 100 * (s["makespan_mean"] - best) / best
        summary += class_summary
    return summary


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    """CSV mit Komma als Trennzeichen und Punkt als Dezimalzeichen."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})


def print_summary(summary: list[dict]) -> None:
    print(f"\n{'Klasse':<10} {'Verfahren':<30} {'n':>4} {'verspaetete Jobs':>24} "
          f"{'ohne Versp.':>11} {'max. Versp.':>11} {'Makespan':>9} {'Gap %':>6} {'Zeit ms':>8}")
    for class_name in dict.fromkeys(s["class"] for s in summary):
        class_summary = [s for s in summary if s["class"] == class_name]
        for s in class_summary:
            if not s["feasible"]:
                print(f"{s['class']:<10} {s['method']:<30} {s['instances']:>4}   "
                      f"keine ausfuehrbare Loesung")
                continue
            late = f"{s['late_jobs_total']} / {s['jobs_total']} = {100 * s['late_share']:5.2f} %"
            print(f"{s['class']:<10} {s['method']:<30} {s['instances']:>4} {late:>24} "
                  f"{s['instances_without_late']:>11} {s['max_tardiness_mean']:>11.1f} "
                  f"{s['makespan_mean']:>9.1f} {s['makespan_gap_pct']:>6.2f} "
                  f"{1000 * s['runtime_s_mean']:>8.1f}")
        best = class_summary[0]["best_makespan_mean"]
        if best is not None:
            print(f"{'':<10} bester Makespan {class_name} (Referenz fuer Gap): {best:.1f}")


# --------------------------------------------------------------------------
# Kommandozeile
# --------------------------------------------------------------------------

def class_files(config: dict, class_name: str, split: str, config_path: Path,
                limit: int | None = None) -> list[Path] | None:
    """Instanzdateien einer Klasse; None mit Hinweis, wenn sie noch nicht erzeugt sind."""
    folder = split_folder(config, class_name, split, config_path)
    files = instance_files([folder]) if folder.is_dir() else []
    if not files:
        print(f"Keine Instanzen in {folder}. Erst erzeugen: "
              f"python generator.py --classes {class_name} --splits {split}")
        return None
    return files[:limit] if limit else files


def main() -> int:
    parser = argparse.ArgumentParser(description="Baselines auf allen Instanzklassen auswerten")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Instanzklassen oder 'all' (Standard: experiments.classes)")
    parser.add_argument("--split", choices=["val", "test"], default=None,
                        help="Datensatz (Standard: experiments.split)")
    parser.add_argument("--limit", type=int, default=None,
                        help="nur die ersten N Instanzen je Klasse (Schnelltest)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Einstellungen (Standard: problem_config.json)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Ergebnisse je Instanz und Verfahren (Standard: experiments.results_file)")
    parser.add_argument("--summary", type=Path, default=None,
                        help="Kennzahlen je Klasse und Verfahren (Standard: experiments.summary_file)")
    args = parser.parse_args()

    config = load_config(args.config)
    settings = config["experiments"]
    selection = settings["classes"] if args.classes is None else args.classes
    selection = "all" if selection == ["all"] else selection
    split = settings["split"] if args.split is None else args.split
    out = resolve_path(settings["results_file"], args.config) if args.out is None else args.out
    summary_file = (resolve_path(settings["summary_file"], args.config)
                    if args.summary is None else args.summary)

    rows = []
    for class_name in selected_classes(config, selection):
        files = class_files(config, class_name, split, args.config, args.limit)
        if files is None:
            return 1
        start = time.perf_counter()
        for path in files:
            rows += run_instance(config, path, class_name)
        print(f"{class_name}: {len(files)} Instanzen x {len(methods(config))} Verfahren "
              f"in {time.perf_counter() - start:.1f} s")

    summary = summarize(rows)
    write_csv(out, RESULT_COLUMNS, rows)
    write_csv(summary_file, SUMMARY_COLUMNS, summary)
    print_summary(summary)
    print(f"\nErgebnisse: {out}  |  Zusammenfassung: {summary_file}")
    print("Verspaetete Jobs = verspaetete / alle Jobs ueber alle Instanzen (Hauptkennzahl, "
          "Optimum 0); max. Versp. = groesste Verspaetung je Instanz in Minuten, gemittelt;\n"
          "Makespan = Mittel ueber die Instanzen; Gap % = Abstand zum besten Makespan der "
          "Klasse in dieser Tabelle; Zeit = Planen + Ausfuehren je Instanz.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
