"""Generate fresh profiles, train paired controls, and report observed differences.

Examples:
    python run_profile_comparison.py --stage prepare --processes 4
    python run_profile_comparison.py --stage train
    python run_profile_comparison.py --stage report
    python run_profile_comparison.py --smoke --stage all
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import statistics

from generator import generate_class_instances
from schedule_validator import (load_config, load_json, resolve_path, selected_classes,
                                split_folder, validate_instance, check_schedule, time_profile_id)

DEFAULT = Path(__file__).resolve().with_name("problem_config_generalization.json")
METRICS = ("total_tardiness", "makespan", "manual_profile_regret", "weighted_manual_factor",
           "machine_idle_time", "machine_productive_utilization", "machine_flow_time")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    helper = path.with_name(path.name + ".tmp")
    helper.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(helper, path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def prepare(config, config_path, output, processes):
    """Reuse only complete validated splits; never overwrite existing instances."""
    classes = selected_classes(config, config["environment"]["train_classes"])
    manifest_path = output / "dataset_manifest.json"
    if manifest_path.exists() and load_json(manifest_path).get("generation_settings") != config["generation"]:
        raise ValueError("Existing dataset uses different generation settings; use separate dataset/output paths")
    data_settings = {split: config["data_splits"][split] for split in ("val", "test")}
    if manifest_path.exists() and load_json(manifest_path).get("data_settings", data_settings) != data_settings:
        raise ValueError("Existing dataset uses different splits/seeds; use separate dataset/output paths")
    records = []
    pool = multiprocessing.get_context("spawn").Pool(processes) if processes > 1 else None
    try:
        for split in ("val", "test"):
            for class_name in classes:
                settings = config["data_splits"][split]
                folder = split_folder(config, class_name, split, config_path)
                paths = sorted(folder.glob("instance_*.json"))
                if paths and len(paths) != settings["count"]:
                    raise ValueError(f"Incomplete existing split {folder}: {len(paths)} files; expected {settings['count']}")
                if not paths:
                    instances, _, skipped = generate_class_instances(
                        config, class_name, settings["seed_start"], settings["count"], config_path, pool)
                    folder.mkdir(parents=True, exist_ok=True)
                    paths = [folder / f"instance_{i:03d}.json" for i in range(len(instances))]
                    for path, instance in zip(paths, instances):
                        with path.open("x", encoding="utf-8") as stream:
                            json.dump(instance, stream, ensure_ascii=False, indent=2)
                    print(f"Prepared {split}/{class_name}: {len(paths)} instances, skipped seeds {skipped}", flush=True)
                else:
                    instances = [load_json(path) for path in paths]
                    print(f"Reusing {split}/{class_name}: {len(paths)} instances", flush=True)
                for path, instance in zip(paths, instances):
                    errors = validate_instance(config, instance)
                    if not errors:
                        errors = check_schedule(config, instance, instance["reference_schedule"])
                    if errors:
                        raise ValueError(f"{path}: {errors[0]}")
                    records.append({"split": split, "instance_id": f"{class_name}/{path.stem}",
                                    "seed": instance["seed"], "time_profile_id": time_profile_id(instance["time_profiles"]),
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    finally:
        if pool:
            pool.close()
            pool.join()
    identifiers = [record["time_profile_id"] for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Repeated time profiles across validation/test instances")
    write_json(output / "dataset_manifest.json", {"instances": records, "generation_settings": config["generation"],
                                                 "data_settings": data_settings,
                                                 "unique_profiles": len(set(identifiers)),
                                                 "validation_test_profiles_disjoint": True})


def train_pairs(config, config_path, output, seeds, smoke):
    import torch
    from rl.train import train

    dataset_path = output / "dataset_manifest.json"
    if not dataset_path.exists():
        raise ValueError("Run --stage prepare first to create and verify the shared dataset")
    dataset = load_json(dataset_path)
    if dataset.get("generation_settings") != config["generation"]:
        raise ValueError("Dataset generation settings differ from the requested experiment")
    if dataset.get("data_settings") != {split: config["data_splits"][split] for split in ("val", "test")}:
        raise ValueError("Dataset splits/seeds differ; rerun --stage prepare to verify them")
    for record in dataset["instances"]:
        class_name, name = record["instance_id"].split("/")
        path = split_folder(config, class_name, record["split"], config_path) / (name + ".json")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Dataset changed after preparation: {path}")
    registry_path = output / "comparison_runs.json"
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    registry = load_json(registry_path) if registry_path.exists() else {
        "config_sha256": fingerprint, "smoke": smoke, "runs": []}
    if registry["config_sha256"] != fingerprint:
        raise ValueError("Existing comparison uses different settings; select a separate output folder/config")
    # Mean controls know deadlines, clock and average resource speed. Both are
    # trained and executed in the same real time-dependent environment.
    for seed in seeds:
        for information in ("full", "mean"):
            previous = [r for r in registry["runs"] if r["seed"] == seed and r["time_information"] == information]
            if previous:
                print(f"Already trained: seed {seed}, {information}", flush=True)
                continue
            cfg = copy.deepcopy(config)
            cfg["model"]["time_information"] = information
            checkpoint_folder = train(cfg, seed, cfg["training"]["iterations"], config_path)
            results = output / "results" / checkpoint_folder.name / "test_instances.csv"
            if not results.is_file():
                raise RuntimeError(f"Missing final test output: {results}")
            registry["runs"].append({"seed": seed, "time_information": information,
                                     "checkpoint": str(checkpoint_folder / "best.pt"), "results": str(results)})
            write_json(registry_path, registry)


def report(output):
    registry = load_json(output / "comparison_runs.json")
    summaries, paired, by_seed = [], [], {}
    for run in registry["runs"]:
        rows = read_csv(Path(run["results"]))
        by_seed.setdefault(run["seed"], {})[run["time_information"]] = {r["instance_id"]: r for r in rows}
        jobs = sum(int(row["jobs"]) for row in rows)
        row = {"seed": run["seed"], "time_information": run["time_information"], "instances": len(rows),
               "feasible": sum(r["feasible"] == "True" for r in rows),
               "on_time_job_share": 1 - sum(int(r["late_jobs"]) for r in rows) / jobs}
        for metric in METRICS:
            values = [float(r[metric]) for r in rows if r.get(metric) not in (None, "")]
            row[metric] = statistics.mean(values) if values else None
            row[metric + "_n"] = len(values)
        summaries.append(row)
    for seed, runs in by_seed.items():
        if set(runs) != {"full", "mean"}:
            raise ValueError(f"Missing partner for seed {seed}")
        if set(runs["full"]) != set(runs["mean"]):
            raise ValueError(f"Different test instances for seed {seed}")
        for identifier, full in runs["full"].items():
            mean = runs["mean"][identifier]
            if (full["instance_seed"], full["time_profile_hash"]) != (mean["instance_seed"], mean["time_profile_hash"]):
                raise ValueError(f"Different instance profiles in pair {identifier}")
            row = {"seed": seed, "instance_id": identifier,
                   "both_feasible": full["feasible"] == mean["feasible"] == "True"}
            for metric in METRICS:
                row[metric + "_full_minus_mean"] = (float(full[metric]) - float(mean[metric])
                    if full.get(metric) not in (None, "") and mean.get(metric) not in (None, "") else None)
            paired.append(row)
    write_csv(output / "profile_comparison.csv", summaries)
    write_csv(output / "paired_profile_differences.csv", paired)
    lines = ["# Vergleich instanzeigener Zeitprofile", ""]
    if registry["smoke"]:
        lines += ["**Technischer Kurztest, kein Nachweis einer Lernverbesserung.** Wenige Iterationen und kleine Netze prüfen den vollständigen Ablauf.", ""]
    lines += ["full berücksichtigt den zeitlichen Verlauf. mean verwendet zeitgewichtete konstante Stations- und Rüstmittel; Deadlines, Uhrzeit, Architektur, Reward und reale Umgebung bleiben gleich.", "",
              "| Seed | Sicht | Zulässig / Instanzen | Verspätung min | Pünktliche Jobs | Manueller Profilverlust min | Maschinenleerlauf min | Produktive Auslastung |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    def f(value, percent=False):
        return "–" if value is None else f"{value * (100 if percent else 1):.3f}" + (" %" if percent else "")
    for row in summaries:
        lines.append("| " + " | ".join([str(row["seed"]), row["time_information"], f"{row['feasible']} / {row['instances']}",
                         f(row['total_tardiness']), f(row['on_time_job_share'], True), f(row['manual_profile_regret']),
                         f(row['machine_idle_time']), f(row['machine_productive_utilization'], True)]) + " |")
    lines += ["", "Die folgenden Differenzen werden zunächst je Trainingsseed über dieselben Instanzen und anschließend über Seeds gemittelt. Negativ ist für Kosten besser; bei produktiver Auslastung ist positiv höher.", "",
              "| Kennzahl (full − mean) | Mittel über Seeds | Standardabweichung zwischen Seeds | Seeds |", "|---|---:|---:|---:|"]
    for metric in METRICS:
        seed_means = []
        for seed in sorted(by_seed):
            values = [r[metric + "_full_minus_mean"] for r in paired if r["seed"] == seed and r[metric + "_full_minus_mean"] is not None]
            if values:
                seed_means.append(statistics.mean(values))
        lines.append(f"| {metric} | {f(statistics.mean(seed_means) if seed_means else None)} | {f(statistics.stdev(seed_means) if len(seed_means)>1 else None)} | {len(seed_means)} |")
    lines += ["", "Kosten/Tardiness enthalten auch die Abbruchstrafe; Planmetriken fehlen bei abgebrochenen Episoden. Die n-Spalten in profile_comparison.csv zeigen die Stichprobengröße jeder Kennzahl. Bei unvollständigen Plänen sind kleine Mittelwerte einzelner Planmetriken kein Qualitätsbeleg.", "",
              "Alle Modelle verwenden denselben synthetischen Profilgenerator, aber Training, Validierung und Test haben getrennte Instanzseeds und Profile. Dies prüft neue Profile innerhalb dieser Verteilung, nicht automatisch unbekannte reale Profilfamilien.", "",
              "Ein Vorteil gilt nur für die gemessenen Ziele und Seeds; die Auswertung meldet auch Verschlechterungen. Gewichte dürfen anhand der Validierung angepasst werden, nicht anhand wiederholter Auswahl auf dem Testdatensatz.", "",
              "[Zusammenfassung](profile_comparison.csv) · [Gepaarte Unterschiede](paired_profile_differences.csv) · [Datensatz](dataset_manifest.json) · [Checkpoints](comparison_runs.json)", ""]
    (output / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Comparison written: {output / 'comparison_report.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--stage", choices=("prepare", "train", "report", "all"), default="all")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, help="Separater Ergebnisordner; feste Instanzen bleiben dieselben")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.processes < 1 or args.threads < 1 or (args.iterations is not None and args.iterations < 1):
        parser.error("Processes, threads and iterations must be positive")
    config_path = args.config.resolve()
    config = load_config(config_path)
    if not config["generation"].get("instance_time_profiles", {}).get("enabled"):
        parser.error("This comparison requires per-instance time profiles")
    if config["environment"]["view"] != "time_aware":
        parser.error("Both models must execute in the real time-aware environment")
    if args.smoke:
        config["environment"].update(train_classes=["klein_11", "mittel_16"], generator_workers=0, device="cpu")
        config["experiments"]["classes"] = ["klein_11", "mittel_16"]
        config["model"].update(d_model=32, hidden_dim=32, num_layers=1)
        config["training"].update(iterations=4, instances_per_iteration=4, ppo_epochs=1,
                                  validation_instances=2, validate_every=2,
                                  output_folder="experiments_generalization/smoke_verified/checkpoints")
        config["monitoring"].update(folder="experiments_generalization/smoke_verified", gantt_instances=0, static_reference=None)
        for split in ("val", "test"):
            config["data_splits"][split].update(count=2, seed_start=500000 if split == "val" else 400000,
                                               folder=f"instances_generalization/smoke_verified/{split}/{{class}}")
    if args.iterations:
        config["training"]["iterations"] = args.iterations
    if args.output:
        config["monitoring"]["folder"] = str(args.output.resolve())
        config["training"]["output_folder"] = str(args.output.resolve() / "checkpoints")
    output = resolve_path(config["monitoring"]["folder"], config_path)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(output / "plot_cache"))
    seeds = args.seeds or ([1] if args.smoke else config.get("comparison", {}).get("seeds", [1, 2, 3]))
    if args.stage in ("prepare", "all"):
        prepare(config, config_path, output, 1 if args.smoke else args.processes)
    if args.stage in ("train", "all"):
        import torch
        torch.set_num_threads(args.threads)
        train_pairs(config, config_path, output, seeds, args.smoke)
    if args.stage in ("report", "all"):
        report(output)


if __name__ == "__main__":
    main()
