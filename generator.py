"""Instanzgenerator fuer das zeitabhaengige FJSP.

Alle Scheduling-Regeln und das Zeitmodell kommen aus problem_config.json,
ausgewertet ueber schedule_validator.py. Dieser Generator enthaelt selbst
keine Regeln, nur die Erzeugung der Daten und des Referenz-Schedules.

Der Referenz-Schedule haelt alle Deadlines ein (sie werden aus ihm abgeleitet).
Jede Instanz hat also einen Plan ohne Verspaetung, auch wenn Deadlines in der
Config weich sind (rules.due_dates_hard = false).

Aufbau:
    Instanz erzeugen      generate_instance und ihre Teilschritte
    Referenz-Schedule     randomisierte serielle Listenplanung (Loesbarkeitsnachweis)
    Instanzklassen        Klassen aus der Config, Seeds ohne Loesung ueberspringen
    Anzeige               Tabelle im Terminal, Gantt-Diagramm mit matplotlib
    Kommandozeile         python generator.py ...

Aufruf:
    python generator.py --classes all --splits test val          Klassen aus der Config
    python generator.py --count 5 --jobs 6 --machines 3 --ops 3 --workers 2 --show
"""
from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
import multiprocessing
import random
from pathlib import Path

from evaluation import job_completion, makespan
from schedule_validator import (DEFAULT_CONFIG, all_resources, check_schedule,
                                _check_instance_profile_settings, config_fingerprint, horizon,
                                instance_classes, instance_config, load_config,
                                selected_classes, setup_resources, split_folder,
                                time_profile_id, validate_instance)
from simulator import first_feasible_start, make_entry, setup_lookup


# --------------------------------------------------------------------------
# Instanz erzeugen
# --------------------------------------------------------------------------

def generate_instance(seed: int = 42, n_jobs: int = 5, n_machines: int = 5,
                      n_ops: int = 5, n_workers: int = 2, n_days: int = 1,
                      due_slack: float | None = None, schedule_attempts: int | None = None,
                      config: dict | None = None,
                      config_path: Path = DEFAULT_CONFIG) -> dict:
    """Erzeugt eine loesbare Instanz.

    Ablauf: Jobs und Operationen -> Ruestzeiten -> Referenz-Schedule ->
    Deadlines aus dem Referenz-Schedule. Gleicher Seed = gleiche Instanz.

    due_slack / schedule_attempts: None = Wert aus problem_config.json
    (Abschnitt "generation"). due_slack ist der Deadline-Puffer relativ zur
    Durchlaufzeit im Referenz-Schedule (0 = sehr eng, 0.5 = bis zu 50 %).
    Wirft ValueError, wenn die Instanz nicht in den Horizont passt."""
    if config is None:
        config = load_config(config_path)
    profile_errors = _check_instance_profile_settings(config)
    if profile_errors:
        raise ValueError("; ".join(profile_errors))
    gen = config["generation"]
    due_slack = gen["due_slack"] if due_slack is None else due_slack
    schedule_attempts = gen["schedule_attempts"] if schedule_attempts is None else schedule_attempts
    _check_arguments(gen, n_jobs, n_machines, n_ops, n_workers, n_days, due_slack)

    rng = random.Random(seed)
    machines = [f"M{i + 1}" for i in range(n_machines)]
    # Jeder Worker hat genau einen festen Handarbeitsplatz: W1 <-> H1, W2 <-> H2, ...
    workers = [f"W{i + 1}" for i in range(n_workers)]
    stations = [f"H{i + 1}" for i in range(n_workers)]
    jobs, operations = _generate_jobs(rng, gen, n_jobs, n_ops, machines, stations)

    instance = {
        "schema_version": 2,
        "problem_config": {"file": Path(config_path).name,
                           "fingerprint": config_fingerprint(config)},
        "seed": seed,
        "days": n_days,
        "resources": {"manual_stations": stations, "machines": machines, "workers": workers},
        "station_worker": dict(zip(stations, workers)),  # H1 -> W1, H2 -> W2, ...
        "jobs": jobs,
        "operations": operations,
        "setups": [],
    }
    instance["setups"] = _generate_setups(rng, gen, config, instance)
    profile_settings = gen.get("instance_time_profiles") or {}
    if profile_settings.get("enabled", False):
        instance["time_profiles"] = _generate_time_profiles(config, instance, n_ops)
        instance["time_profile_id"] = time_profile_id(instance["time_profiles"])
    config = instance_config(config, instance)

    # Loesbarkeit per Konstruktion (vgl. Dumas et al. 1995 fuer TSPTW):
    # erst einen zulaessigen Referenz-Schedule bauen, dann die Deadlines
    # aus dessen Fertigstellungszeiten plus zufaelligem Puffer ableiten.
    schedule = _best_reference_schedule(config, instance, rng, schedule_attempts)
    _assign_due_dates(rng, config, instance, schedule, due_slack)
    instance["reference_schedule"] = schedule
    return instance


def _generate_time_profiles(config: dict, instance: dict, n_ops: int) -> dict:
    """Deterministic instance-owned profiles, using a separate random stream.

    Circular phase shifts preserve the template's rhythm on working windows;
    amplitude and small jitter vary the slowdown.  Breaks never move, and the
    best windows stay at min_factor.  Job durations, eligibility and setup base
    times retain their original random stream when this feature is enabled.
    """
    errors = _check_instance_profile_settings(config)
    if errors:
        raise ValueError("; ".join(errors))
    settings = config["generation"]["instance_time_profiles"]
    low, high, jitter = settings.get("min_factor", 1.0), settings.get("max_factor", 1.8), settings.get("jitter", 0.05)
    stations = instance["resources"]["manual_stations"]
    seed_data = ["instance_time_profiles_v1", instance["seed"], len(instance["jobs"]), n_ops,
                 len(stations), len(instance["resources"]["machines"]), instance["days"]]
    profile_seed = int.from_bytes(hashlib.sha256(json.dumps(seed_data).encode()).digest(), "big")
    profile_rng = random.Random(profile_seed)
    windows = config["time_model"]["day_windows"]
    working = [i for i, window in enumerate(windows) if window["factor"] is not None]
    base = [window["factor"] for window in windows]
    templates = config["time_model"].get("resource_factors", {})
    station_templates = templates.get("manual_stations", [base])

    def sample(template: list) -> list:
        values = [template[i] for i in working]
        minimum, maximum = min(values), max(values)
        if maximum == minimum:
            shape = [float(i % 3 == 2) for i in range(len(values))]
            shape[-1] = 1.0
        else:
            shape = [(value - minimum) / (maximum - minimum) for value in values]
        phase = profile_rng.randrange(len(shape))
        shape = shape[phase:] + shape[:phase]
        amplitude = profile_rng.uniform(0.65, 1.0) * (high - low)
        profile = [None] * len(windows)
        for index, fraction in zip(working, shape):
            value = low if fraction == 0 else low + amplitude * fraction + profile_rng.uniform(-jitter, jitter)
            profile[index] = min(high, max(low, round(value, 6)))
        return profile

    return {"manual_stations": [sample(station_templates[i % len(station_templates)]) for i in range(len(stations))],
            "setup": sample(templates.get("setup", base))}


def _check_arguments(gen: dict, n_jobs: int, n_machines: int, n_ops: int,
                     n_workers: int, n_days: int, due_slack: float) -> None:
    """Groessenangaben pruefen; ValueError bei ungueltigen Werten."""
    if n_days < 1:
        raise ValueError("Es muss mindestens 1 Tag geplant werden")
    if n_jobs < gen["min_jobs"]:
        raise ValueError(f"Es werden mindestens {gen['min_jobs']} Jobs benoetigt")
    if n_workers < 1:
        raise ValueError("Es wird mindestens 1 Worker benoetigt")
    if n_machines < 1:
        raise ValueError("Es wird mindestens 1 Maschine benoetigt")
    if n_ops < 1:
        raise ValueError("Jeder Job braucht mindestens 1 Operation")
    if due_slack < 0:
        raise ValueError("due_slack darf nicht negativ sein")
    if gen["kind_pattern"] != "alternating":
        raise ValueError(f"kind_pattern '{gen['kind_pattern']}' wird nicht unterstuetzt")


def _generate_jobs(rng: random.Random, gen: dict, n_jobs: int, n_ops: int,
                   machines: list[str], stations: list[str]) -> tuple[list[dict], list[dict]]:
    """Jobs und ihre Operationen mit zulaessigen Ressourcen und Basisdauern.
    Die Deadlines (`due`) werden spaeter aus dem Referenz-Schedule gesetzt."""
    jobs, operations = [], []
    num_auto = 0

    for j in range(n_jobs):
        job_id = f"J{j + 1}"
        op_ids = []

        for step in range(n_ops):
            op_id = f"{job_id}_O{step + 1}"
            # "alternating": manuell und automatisch im Schachbrettmuster
            kind = "manual" if (j + step) % 2 == 0 else "automatic"

            if kind == "manual":
                # Jeder Worker kann jede manuelle Operation; Basisdauer entweder
                # gleich auf allen Plaetzen (Worker gleich schnell) oder je Platz
                if gen["manual_same_duration_on_all_stations"]:
                    base = rng.randint(*gen["manual_processing_range"])
                    base_processing = {h: base for h in stations}
                else:
                    base_processing = {h: rng.randint(*gen["manual_processing_range"])
                                       for h in stations}
            else:
                eligible = _eligible_machines(rng, gen, machines, first=num_auto == 0)
                base_processing = _machine_times(rng, gen, eligible)
                num_auto += 1

            operations.append({
                "id": op_id,
                "job": job_id,
                "kind": kind,
                "base_processing": base_processing,
            })
            op_ids.append(op_id)

        jobs.append({"id": job_id, "operations": op_ids, "release": gen["release"],
                     "due": None})
    return jobs, operations


def _eligible_machines(rng: random.Random, gen: dict, machines: list[str],
                       first: bool) -> list[str]:
    """Maschinen, auf denen eine automatische Operation laufen darf."""
    # Erste automatische Operation darf auf allen Maschinen laufen,
    # damit jede Maschine mindestens einmal zulaessig ist.
    if first and gen["first_automatic_on_all_machines"]:
        return list(machines)
    if len(machines) >= 2 and rng.random() < gen["flexible_probability"]:
        return rng.sample(machines, rng.randint(2, len(machines)))  # flexibel
    return [rng.choice(machines)]  # fest an eine Maschine


def _machine_times(rng: random.Random, gen: dict, eligible: list[str]) -> dict[str, int]:
    """Bearbeitungszeit einer automatischen Operation je zulaessiger Maschine.

    automatic_machine_deviation = d: eine Basiszeit pro Operation, je Maschine
    um bis zu +-d abweichend (z. B. 0.2 -> Basis 45 min ergibt 36-54 min).
    automatic_machine_deviation = null: Zeit je Maschine unabhaengig gezogen."""
    deviation = gen["automatic_machine_deviation"]
    if deviation is None:
        return {m: rng.randint(*gen["automatic_processing_range"]) for m in eligible}
    base = rng.randint(*gen["automatic_processing_range"])
    return {m: max(1, round(base * rng.uniform(1 - deviation, 1 + deviation)))
            for m in eligible}


def _generate_setups(rng: random.Random, gen: dict, config: dict,
                     instance: dict) -> list[dict]:
    """Direkte, asymmetrische Ruestzeiten fuer jedes geordnete OP-Paar auf jeder
    Ressource mit Ruestung (laut Regeln), sofern beide Operationen dort laufen duerfen."""
    setups = []
    for resource in setup_resources(config, instance):
        candidates = [o["id"] for o in instance["operations"]
                      if resource in o["base_processing"]]
        for previous in candidates:
            for next_op in candidates:
                if previous == next_op:
                    continue
                setups.append({
                    "resource": resource,
                    "from_op": previous,
                    "to_op": next_op,
                    "base_minutes": rng.randint(*gen["setup_range"]),
                })
    return setups


def _assign_due_dates(rng: random.Random, config: dict, instance: dict,
                      schedule: list[dict], due_slack: float) -> None:
    """Deadline je Job = Fertigstellung im Referenz-Schedule + zufaelliger
    Puffer (0 bis due_slack * Durchlaufzeit), hoechstens das Horizontende."""
    for job in instance["jobs"]:
        completion = job_completion(schedule, job)
        slack = math.ceil(rng.uniform(0, due_slack) * (completion - job["release"]))
        job["due"] = min(horizon(config, instance), completion + slack)


def save_instance(instance: dict, path: Path) -> None:
    """Instanz als JSON speichern (Ordner wird bei Bedarf angelegt)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(instance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Referenz-Schedule
# --------------------------------------------------------------------------

def _best_reference_schedule(config: dict, instance: dict, rng: random.Random,
                             attempts: int) -> list[dict]:
    """Mehrere randomisierte Durchlaeufe; der mit dem kleinsten Makespan gewinnt.
    ValueError, wenn kein Durchlauf in den Horizont passt."""
    candidates = [s for s in (build_reference_schedule(config, instance, rng)
                              for _ in range(attempts)) if s is not None]
    if not candidates:
        raise ValueError("Kein zulaessiger Referenz-Schedule im Horizont gefunden. "
                         "Instanz zu gross: weniger Jobs/Operationen oder mehr "
                         "Maschinen/Worker/Tage waehlen.")
    return min(candidates, key=makespan)


def build_reference_schedule(config: dict, instance: dict,
                             rng: random.Random) -> list[dict] | None:
    """Randomisierte serielle Listenplanung (ein Durchlauf).

    Solange Operationen offen sind: zufaelligen Job waehlen, seine naechste
    Operation auf die zulaessige Ressource mit dem fruehesten Ende legen,
    und zwar am fruehesten zulaessigen Zeitpunkt nach dem letzten Auftrag
    dieser Ressource (Luecken werden nicht aufgefuellt).
    Liefert None, wenn nicht alles in den Horizont passt."""
    config = instance_config(config, instance)
    ops = {o["id"]: o for o in instance["operations"]}
    setup_base = setup_lookup(instance)
    with_setup = set(setup_resources(config, instance))
    initial_setup = config["rules"]["initial_setup_minutes"]
    resources = all_resources(instance)

    resource_free = {r: 0 for r in resources}  # Ende des letzten Auftrags
    last_on = {r: None for r in resources}      # letzte Operation (fuer Ruestzeit)
    next_index = {job["id"]: 0 for job in instance["jobs"]}
    job_ready = {job["id"]: job["release"] for job in instance["jobs"]}
    job_ops = {job["id"]: job["operations"] for job in instance["jobs"]}
    pending = [job["id"] for job in instance["jobs"]]
    schedule = []

    while pending:
        job_id = rng.choice(pending)
        op = ops[job_ops[job_id][next_index[job_id]]]
        best = None

        for resource in op["base_processing"]:
            previous = last_on[resource]
            if resource not in with_setup:
                setup = None
            elif previous is None:
                setup = (None, initial_setup)
            else:
                setup = (previous, setup_base[(resource, previous, op["id"])])
            earliest = max(resource_free[resource], job_ready[job_id])
            entry = _place(config, instance, op, resource, earliest, setup)
            if entry is not None and (best is None or entry["end"] < best["end"]):
                best = entry

        if best is None:
            return None
        schedule.append(best)
        resource_free[best["resource"]] = best["end"]
        last_on[best["resource"]] = best["op"]
        job_ready[job_id] = best["end"]
        next_index[job_id] += 1
        if next_index[job_id] == len(job_ops[job_id]):
            pending.remove(job_id)

    return schedule


def _place(config: dict, instance: dict, op: dict, resource: str, earliest: int,
           setup: tuple[str | None, int] | None) -> dict | None:
    """Fruehestmoegliche Einplanung von `op` auf `resource` ab `earliest`.

    setup = None (Ressource ohne Ruestung) oder (Vorgaenger, Basis-Ruestzeit).
    Ruesten startet erst, wenn der Job bereit ist, und die Bearbeitung folgt
    direkt darauf (simulator.first_feasible_start). Liefert den
    Schedule-Eintrag oder None, wenn nichts mehr in den Horizont passt."""
    found = first_feasible_start(config, instance, op["kind"], op["base_processing"][resource],
                                 setup[1] if setup else 0, earliest, resource)
    return None if found is None else make_entry(op["id"], resource, setup, *found)


# --------------------------------------------------------------------------
# Instanzklassen aus der Konfiguration
# --------------------------------------------------------------------------

def generate_class_instance(config: dict, class_name: str, seed: int,
                            config_path: Path = DEFAULT_CONFIG) -> dict:
    """Instanz einer Klasse aus problem_config.json. ValueError, wenn dieser
    Seed keinen zulaessigen Referenz-Schedule im Horizont hat."""
    c = instance_classes(config)[class_name]
    return generate_instance(seed=seed, n_jobs=c["jobs"], n_machines=c["machines"],
                             n_ops=c["operations_per_job"], n_workers=c["workers"],
                             n_days=c["days"], due_slack=c.get("due_slack"),
                             schedule_attempts=c.get("schedule_attempts"),
                             config=config, config_path=config_path)


def _generate_or_none(task: tuple) -> dict | None:
    """Fuer Prozess-Pools: Instanz oder None, wenn der Seed keine Loesung hat."""
    try:
        return generate_class_instance(*task)
    except ValueError:
        return None


def generate_class_instances(config: dict, class_name: str, seed_start: int, count: int,
                             config_path: Path = DEFAULT_CONFIG, pool=None
                             ) -> tuple[list[dict], int, int]:
    """Die ersten `count` erzeugbaren Instanzen ab `seed_start`, in Seed-Reihenfolge.
    Das Ergebnis haengt nicht davon ab, ob ein Prozess-Pool genutzt wird.
    Rueckgabe: (Instanzen, naechster freier Seed, uebersprungene Seeds)."""
    limit = config["generation"]["max_consecutive_seed_failures"]
    instances: list[dict] = []
    seed, skipped, in_a_row = seed_start, 0, 0
    while len(instances) < count:
        tasks = [(config, class_name, s, config_path)
                 for s in range(seed, seed + count - len(instances))]
        results = pool.map(_generate_or_none, tasks) if pool else map(_generate_or_none, tasks)
        for instance in results:
            if instance is None:
                skipped += 1
                in_a_row += 1
                if in_a_row >= limit:
                    raise ValueError(f"Klasse {class_name}: {limit} Seeds in Folge ohne "
                                     "zulaessigen Referenz-Plan. Weniger Jobs oder mehr "
                                     "Ressourcen/Tage in instance_classes eintragen.")
            else:
                instances.append(instance)
                in_a_row = 0
        seed += len(tasks)
    return instances, seed, skipped


# --------------------------------------------------------------------------
# Anzeige des Referenz-Schedules
# --------------------------------------------------------------------------

# Farben fuer das Gantt-Diagramm: feste Reihenfolge, eine Farbe je Job (siehe _job_color).
JOB_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, MUTED_INK = "#fcfcfb", "#0b0b0b", "#52514e"
GRID, BASELINE, BREAK_FILL = "#e1e0d9", "#c3c2b7", "#efeeea"
EXTENSION_ALPHA = 0.35  # dunkelt den Teil eines Balkens ab, den der Zeitfaktor hinzufuegt


def _job_color(index: int) -> str:
    """Eigene Farbe je Job: erst JOB_COLORS, danach Farbtoene im goldenen Winkel mit
    wechselnder Helligkeit. Ein Job behaelt seine Farbe unabhaengig von der Jobzahl."""
    if index < len(JOB_COLORS):
        return JOB_COLORS[index]
    hue = index * 0.618034 % 1
    red, green, blue = colorsys.hls_to_rgb(hue, (0.72, 0.58, 0.82)[index % 3], 0.75)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def _darker(color: str) -> str:
    """Farbe des Balkenteils, den der Zeitfaktor hinzufuegt."""
    channels = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    ink = [int(INK[i:i + 2], 16) for i in (1, 3, 5)]
    mixed = (round(c + (k - c) * EXTENSION_ALPHA) for c, k in zip(channels, ink))
    return "#" + "".join(f"{value:02x}" for value in mixed)


def _clock(config: dict, instance: dict, minute: int, end: bool = False,
           with_day: bool | None = None) -> str:
    """Arbeitsminute -> Uhrzeit, z. B. 75 -> '07:15'; bei mehreren Tagen
    mit Tag davor, z. B. 795 -> 'T2 07:15'. end=True ordnet ein Ende genau
    auf der Tagesgrenze dem alten Tag zu (720 -> 'T1 18:00' statt 'T2 06:00')."""
    day_len = config["time_model"]["day_minutes"]
    day = (minute - 1) // day_len if end and minute > 0 else minute // day_len
    hours, minutes = map(int, config["time_model"]["origin"].split(":"))
    total = hours * 60 + minutes + minute - day * day_len
    time = f"{total // 60:02d}:{total % 60:02d}"
    if with_day is None:
        with_day = instance["days"] > 1
    return f"T{day + 1} {time}" if with_day else time


def print_reference_schedule(config: dict, instance: dict) -> None:
    """Referenz-Schedule als Tabelle je Ressource und Job-Uebersicht ausgeben."""
    schedule = instance["reference_schedule"]
    clock = lambda m, end=False: _clock(config, instance, m, end)

    print("\nReferenz-Schedule")
    for resource in all_resources(instance):
        entries = sorted((e for e in schedule if e["resource"] == resource),
                         key=lambda e: e["start"])
        worker = instance["station_worker"].get(resource)
        print(f"  {resource}" + (f" ({worker})" if worker else ""))
        if not entries:
            print("    -")
        for e in entries:
            if e.get("setup_end", e["start"]) > e.get("setup_start", e["start"]):
                print(f"    {clock(e['setup_start'])}-{clock(e['setup_end'], True)}  "
                      f"Ruesten {e['setup_from']} -> {e['op']}")
            print(f"    {clock(e['start'])}-{clock(e['end'], True)}  {e['op']}")

    print("\n  Job   fertig  Deadline  Puffer")
    for job in instance["jobs"]:
        completion = job_completion(schedule, job)
        print(f"  {job['id']:<5} {clock(completion, True)}   {clock(job['due'], True)}   "
              f"{job['due'] - completion:>4} min")
    print(f"  Makespan: {clock(makespan(schedule), True)}")


def plot_reference_schedule(config: dict, instance: dict, path: Path | None = None):
    """Referenz-Schedule als Gantt-Diagramm mit matplotlib.
    Mit `path` wird das Bild gespeichert; die Figure wird zurueckgegeben."""
    return plot_schedule(config, instance, instance["reference_schedule"],
                         "Referenz-Schedule", path)


def _factor_extensions(config: dict, instance: dict, entry: dict) -> tuple[int, int]:
    """Minuten, die der Zeitfaktor zu Ruestung und Bearbeitung eines Eintrags hinzufuegt
    (tatsaechliche Dauer minus Basisdauer)."""
    op = next(o for o in instance["operations"] if o["id"] == entry["op"])
    process = entry["end"] - entry["start"] - op["base_processing"][entry["resource"]]
    if entry.get("setup_end", 0) <= entry.get("setup_start", 0):
        return 0, process
    base = (config["rules"]["initial_setup_minutes"] if entry["setup_from"] is None
            else setup_lookup(instance)[(entry["resource"], entry["setup_from"], entry["op"])])
    return entry["setup_end"] - entry["setup_start"] - base, process


def _deadline_levels(jobs: list[dict], label_minutes: float) -> tuple[list[tuple[dict, int]], int]:
    """Verteilt die Deadline-Beschriftungen auf Zeilen, damit sie sich nicht ueberlappen.
    label_minutes = Breite einer Beschriftung in Minuten auf der Zeitachse.
    Rueckgabe: (Job, Zeile) je Job und die Anzahl der Zeilen."""
    line_ends: list[float] = []
    placed = []
    for job in sorted(jobs, key=lambda j: j["due"]):
        level = next((i for i, end in enumerate(line_ends) if end <= job["due"]), len(line_ends))
        if level == len(line_ends):
            line_ends.append(0)
        line_ends[level] = job["due"] + label_minutes
        placed.append((job, level))
    return placed, len(line_ends)


def plot_schedule(config: dict, instance: dict, schedule: list[dict], title: str,
                  path: Path | None = None, waits: list[dict] | None = None, show_times: bool = False):
    """Beliebigen Schedule als Gantt-Diagramm. instance["days"] legt die Achse fest.
    Der Teil eines Balkens, den der Zeitfaktor hinzufuegt, ist abgedunkelt; bei
    Operationen steht die Verlaengerung in Minuten unter dem Namen. Die Deadlines
    stehen beschriftet in einem eigenen Streifen unter den Ressourcen.
    waits: freiwillige Wartezeiten [{"resource", "start", "end"}], als gepunktete Flaeche.
    show_times: tatsaechlicher Start und tatsaechliches Ende stehen im Balken."""
    import matplotlib.pyplot as plt  # nur hier gebraucht, daher spaet importiert
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    config = instance_config(config, instance)
    resources = all_resources(instance)
    end_of_horizon = horizon(config, instance)
    days = instance["days"]
    day_len = config["time_model"]["day_minutes"]
    job_ids = [job["id"] for job in instance["jobs"]]
    color = {job_id: _job_color(i) for i, job_id in enumerate(job_ids)}
    job_of = {o["id"]: o["job"] for o in instance["operations"]}

    width_inches = min(13 + 8 * (days - 1), 40)
    label_minutes = 0.62 * end_of_horizon / (width_inches - 2.5)
    deadlines, deadline_lines = _deadline_levels(instance["jobs"], label_minutes)
    line_height = 0.33      # in Zeilen der Ressourcen
    fig, ax = plt.subplots(figsize=(width_inches,
                                    0.55 * (len(resources) + deadline_lines * line_height) + 2.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Pausen als graues Band
    for d in range(days):
        for w in config["time_model"]["day_windows"]:
            if w["factor"] is None:
                start, end = d * day_len + w["start"], d * day_len + w["end"]
                ax.axvspan(start, end, color=BREAK_FILL, zorder=0, lw=0)
                ax.text((start + end) / 2, -0.75, "Pause", ha="center",
                        va="bottom", fontsize=8, color=MUTED_INK)

    # Ruestungen (schraffiert) und Operationen (eine Farbe je Job, beschriftet)
    bar_height = 0.6
    total_extension = 0
    for row, resource in enumerate(resources):
        for e in (e for e in schedule if e["resource"] == resource):
            setup_extra, process_extra = _factor_extensions(config, instance, e)
            total_extension += setup_extra + process_extra
            if e.get("setup_end", 0) > e.get("setup_start", 0):
                ax.barh(row, e["setup_end"] - e["setup_start"], left=e["setup_start"],
                        height=bar_height, color=SURFACE, edgecolor=MUTED_INK,
                        hatch="////", linewidth=0.6, zorder=2)
                if setup_extra:
                    ax.barh(row, setup_extra, left=e["setup_end"] - setup_extra, height=bar_height,
                            color=INK, alpha=EXTENSION_ALPHA, linewidth=0, zorder=2.5)
            ax.barh(row, e["end"] - e["start"], left=e["start"], height=bar_height,
                    color=color[job_of[e["op"]]], edgecolor=SURFACE, linewidth=1.5,
                    zorder=3)
            if process_extra:
                ax.barh(row, process_extra, left=e["end"] - process_extra, height=bar_height,
                        color=_darker(color[job_of[e["op"]]]), edgecolor=SURFACE, linewidth=1.5,
                        zorder=3.5)
            # Beschriftung im Teil mit der Basisdauer, damit sie gut lesbar bleibt
            width = e["end"] - e["start"] - process_extra
            label = f"{e['op']}\n+{process_extra}" if process_extra else e["op"]
            if show_times:
                extra = f" +{process_extra}" if process_extra else ""
                label = (f"{e['op']}{extra}\n{_clock(config, instance, e['start'], with_day=False)}\n"
                         f"{_clock(config, instance, e['end'], end=True, with_day=False)}")
            ax.text(e["start"] + width / 2, row, label, ha="center",
                    va="center", fontsize=(5 if show_times else 7) if width >= 26 else (4.5 if show_times else 5.5),
                    color=INK, rotation=0 if width >= 18 else 90, zorder=4, linespacing=1.0 if show_times else None)

    # Freiwilliges Warten: Die Ressource haette hier anfangen koennen
    for wait in waits or []:
        ax.barh(resources.index(wait["resource"]), wait["end"] - wait["start"], left=wait["start"],
                height=bar_height, facecolor="none", edgecolor=MUTED_INK, hatch="....",
                linewidth=0.6, linestyle="--", zorder=1.8)

    # Deadlines: eigener Streifen unter den Ressourcen, Linie nach oben hinter den Balken
    strip_top = len(resources) - 0.4 + (0.45 if days > 1 else 0)
    ax.axhline(strip_top, color=BASELINE, linewidth=0.8, zorder=1)
    first_line = strip_top + 0.3
    with_day = max(job["due"] for job in instance["jobs"]) > day_len
    for job, level in deadlines:
        y = first_line + level * line_height
        ax.plot([job["due"], job["due"]], [-0.5, y], color=color[job["id"]], linewidth=0.8,
                linestyle=(0, (3, 3)), zorder=1.5)
        ax.plot(job["due"], y, marker="D", markersize=4.5, color=color[job["id"]],
                markeredgecolor=INK, markeredgewidth=0.5, zorder=4, clip_on=False)
        clock = _clock(config, instance, job["due"], end=True, with_day=with_day)
        ax.text(job["due"] + label_minutes * 0.08, y, f"{job['id']} {clock}", ha="left",
                va="center", fontsize=6.5, color=INK, zorder=4)

    # Achsen: Ressourcen als Zeilen, Uhrzeit ohne Tag; an Tagesgrenzen beginnt der naechste Tag
    worker_of = instance["station_worker"]
    strip_end = first_line + (deadline_lines - 1) * line_height + 0.3
    ax.set_yticks([*range(len(resources)), (first_line + strip_end - 0.3) / 2],
                  [f"{r} ({worker_of[r]})" if r in worker_of else r for r in resources]
                  + ["Deadlines"])
    ax.set_ylim(strip_end, -0.85)
    ax.set_xlim(0, end_of_horizon)
    step = 60 if days <= 2 else 120 if days <= 4 else 240
    ticks = range(0, end_of_horizon + 1, step)
    ax.set_xticks(ticks, [_clock(config, instance, t, end=t == end_of_horizon,
                                 with_day=False) for t in ticks])
    for d in range(days):
        if d > 0:  # Nacht: alles steht still
            ax.axvline(d * day_len, color=MUTED_INK, linewidth=1.5, zorder=5)
        if days > 1:
            ax.text(d * day_len + day_len / 2, len(resources) - 0.4, f"Tag {d + 1}",
                    ha="center", va="top", fontsize=9, color=MUTED_INK)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED_INK, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)

    ax.set_title(f"{title}  |  Seed {instance['seed']}  |  "
                 f"Makespan {_clock(config, instance, makespan(schedule), end=True)}  |  "
                 f"Verlaengerung durch Zeitfaktoren {total_extension} min",
                 loc="left", fontsize=11, color=INK)

    handles = [Patch(facecolor=color[j], edgecolor=SURFACE, label=j)
               for j in job_ids]
    handles.append(Patch(facecolor=SURFACE, edgecolor=MUTED_INK, hatch="////",
                         label="Ruesten"))
    if total_extension:
        handles.append(Patch(facecolor=_darker(BASELINE), edgecolor=SURFACE,
                             label="Verlaengerung\ndurch Zeitfaktor"))
    if waits:
        handles.append(Patch(facecolor="none", edgecolor=MUTED_INK, hatch="....", linestyle="--",
                             label="Freiwilliges\nWarten"))
    handles.append(Patch(facecolor=BREAK_FILL, edgecolor=BASELINE, label="Pause"))
    if days > 1:
        handles.append(Line2D([], [], color=MUTED_INK, linewidth=1.5, label="Schichtende\n(Nacht)"))
    handles.append(Line2D([], [], color=MUTED_INK, linestyle=(0, (3, 3)), linewidth=0.8, marker="D",
                          markersize=4.5, markeredgecolor=INK, markeredgewidth=0.5, label="Deadline"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
              frameon=False, fontsize=8, labelcolor=INK, ncols=1 + len(handles) // 30)

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    return fig


# --------------------------------------------------------------------------
# Kommandozeile
# --------------------------------------------------------------------------

def _write_instance(config: dict, instance: dict, path: Path, show: bool) -> None:
    """Instanz unabhaengig pruefen, speichern und eine Zeile ausgeben."""
    # Sicherheitsnetz: jede gespeicherte Instanz ist unabhaengig geprueft
    errors = (validate_instance(config, instance)
              or check_schedule(config, instance, instance["reference_schedule"]))
    if errors:
        raise RuntimeError("Erzeugte Instanz ist ungueltig: " + "; ".join(errors))

    save_instance(instance, path)
    print(f"{path}: {len(instance['jobs'])} Jobs, "
          f"{len(instance['operations'])} Operationen, "
          f"{len(instance['setups'])} Setup-Paare, Referenz-Makespan "
          f"{makespan(instance['reference_schedule'])} min")
    if show:
        print_reference_schedule(config, instance)
        print()
        plot_reference_schedule(config, instance)


def _generate_classes(args, config: dict) -> None:
    """Modus --classes: alle Groessen und Seeds kommen aus der Konfiguration."""
    selection = "all" if args.classes == ["all"] else args.classes
    pool = multiprocessing.get_context("spawn").Pool(args.processes) if args.processes > 1 else None
    for class_name in selected_classes(config, selection):
        for split in args.splits:
            settings = config["data_splits"][split]
            count = settings["count"] if args.count is None else args.count
            seed = settings["seed_start"] if args.seed is None else args.seed
            instances, _, skipped = generate_class_instances(
                config, class_name, seed, count, args.config, pool)
            folder = split_folder(config, class_name, split, args.config)
            for i, instance in enumerate(instances):
                _write_instance(config, instance, folder / f"instance_{i:03d}.json", args.show)
            print(f"{class_name} / {split}: {count} Instanzen ab Seed {seed}, "
                  f"{skipped} Seeds uebersprungen")
    if pool:
        pool.close()
        pool.join()


def main() -> None:
    parser = argparse.ArgumentParser(description="TD-FJSP-Instanzen erzeugen")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Instanzklassen aus problem_config.json oder 'all'; Groessen, Seeds "
                             "und Ordner kommen dann aus der Config")
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["val", "test"],
                        help="Datensaetze im Klassen-Modus (Standard: test)")
    parser.add_argument("--processes", type=int, default=1,
                        help="Prozesse im Klassen-Modus (aendert das Ergebnis nicht)")
    parser.add_argument("--count", type=int, default=None,
                        help="Anzahl Instanzen (Standard: 3; im Klassen-Modus aus der Config)")
    parser.add_argument("--jobs", type=int, default=2, help="Anzahl Jobs")
    parser.add_argument("--machines", type=int, default=2, help="Anzahl Maschinen")
    parser.add_argument("--ops", type=int, default=2, help="Operationen pro Job")
    parser.add_argument("--workers", type=int, default=1, help="Worker = Handarbeitsplaetze")
    parser.add_argument("--days", type=int, default=1, help="Anzahl Arbeitstage")
    parser.add_argument("--slack", type=float, default=None,
                        help="Deadline-Puffer relativ zur Referenz-Durchlaufzeit "
                             "(Standard: Wert aus problem_config.json)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Start-Seed; Instanz i bekommt seed + i "
                             "(Standard: 42; im Klassen-Modus aus der Config)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Regeldatei (Standard: problem_config.json)")
    parser.add_argument("--out", type=Path, default=Path("instances"),
                        help="Ausgabeordner (vorhandene Dateien werden ueberschrieben)")
    parser.add_argument("--show", action="store_true",
                        help="Referenz-Schedule anzeigen (Tabelle + Gantt-Diagramm)")
    args = parser.parse_args()
    if args.count is not None and args.count <= 0:
        parser.error("--count muss positiv sein")

    config = load_config(args.config)
    if args.classes:
        _generate_classes(args, config)
    else:
        count = 3 if args.count is None else args.count
        seed = 42 if args.seed is None else args.seed
        for i in range(count):
            instance = generate_instance(seed=seed + i, n_jobs=args.jobs,
                                         n_machines=args.machines, n_ops=args.ops,
                                         n_workers=args.workers, n_days=args.days,
                                         due_slack=args.slack, config=config,
                                         config_path=args.config)
            _write_instance(config, instance, args.out / f"instance_{i:03d}.json", args.show)

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()  # alle Diagramme gleichzeitig oeffnen


if __name__ == "__main__":
    main()
