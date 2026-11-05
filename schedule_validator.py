"""Unabhaengige Pruefung von Instanzen und Schedules.

Alle Scheduling-Regeln und das Zeitmodell stehen in problem_config.json.
Dieses Modul interpretiert sie nur (Referenz-Implementierung des Zeitmodells)
und haengt weder vom Generator noch von einer RL-Environment ab. Beide koennen
die Funktionen hier importieren, z. B. `duration` fuer Action Masking.

Aufbau:
    Konfiguration   laden, auf Vollstaendigkeit pruefen, Fingerprint bilden
    Zeitmodell      Horizont, Zeitfaktor, zulaessige Intervalle, Dauern
    Pruefungen      Struktur einer Instanz, Zulaessigkeit eines Schedules
    Kommandozeile   python schedule_validator.py ...

Aufruf:
    python schedule_validator.py instances/instance_000.json
    python schedule_validator.py instances/                     (alle Dateien)
    python schedule_validator.py instances/instance_000.json --schedule mein_plan.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("problem_config.json")

# Taetigkeitsarten, fuer die die Regeln in der Konfiguration gelten
KINDS = ("manual", "automatic", "setup")
# Arten, die eine Operation haben kann (Ruesten ist keine Operation)
OPERATION_KINDS = ("manual", "automatic")
KIND_NAMES = {"manual": "Manuelle Operation", "automatic": "Automatische Operation",
              "setup": "Ruestung"}
# Regeln, die nur die Bewertung eines Schedules aendern, nicht die Daten einer
# Instanz. Sie gehen nicht in den Fingerprint ein und sind daher umschaltbar,
# ohne Instanzen neu zu erzeugen.
EVALUATION_SWITCHES = ("due_dates_hard",)
# Abschnitte, die jede Konfiguration haben muss
REQUIRED_SECTIONS = ("time_model", "rules", "generation", "simulation", "instance_classes",
                     "data_splits", "experiments", "environment", "reward", "model", "training",
                     "monitoring")
# Erlaubte Werte der Umgebungs-Einstellungen
ENVIRONMENT_VIEWS = ("time_aware", "static")
REWARD_LEVELS = ("total_tardiness", "late_jobs", "max_tardiness", "makespan")
REWARD_MODES = ("terminal", "stepwise")
MODEL_VARIANTS = ("A0", "A1", "A2", "A3")
TIME_INFORMATION = ("full", "none")
VALIDATION_DECODES = ("greedy", "sample")
# Moegliche Kennzahlen fuer das beste Modell und ob groesser besser ist
BEST_METRICS = {"val/reward_mean": True, "val/tardiness_mean": False, "val/makespan_mean": False,
                "val/window_compliance": True}


# --------------------------------------------------------------------------
# Konfiguration
# --------------------------------------------------------------------------

def load_json(path: Path | str) -> dict | list:
    """JSON-Datei (UTF-8) lesen."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    """problem_config.json laden; ValueError, wenn sie ungueltig ist."""
    config = load_json(path)
    errors = check_config(config)
    if errors:
        raise ValueError("Ungueltige Konfiguration: " + "; ".join(errors))
    return config


def _strip_comments(obj):
    """Alle Schluessel, die mit '_' beginnen (Kommentare), rekursiv entfernen."""
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def config_fingerprint(config: dict) -> str:
    """Kurzer Hash ueber Zeitmodell und Regeln (ohne Kommentare, ohne
    Generator-Parameter und ohne reine Bewertungsschalter). Eine Instanz
    speichert ihn, damit sie nicht versehentlich mit geaenderten Regeln
    geprueft wird."""
    relevant = {key: _strip_comments(config[key]) for key in ("time_model", "rules")}
    for key in EVALUATION_SWITCHES:
        relevant["rules"].pop(key, None)
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


def check_config(config: dict) -> list[str]:
    """Prueft, ob die Konfiguration vollstaendig ist und nur Varianten nutzt,
    die dieses Modul auswerten kann. Leere Liste = in Ordnung."""
    missing = [name for name in REQUIRED_SECTIONS if name not in config]
    if missing:
        return [f"Abschnitt fehlt: {name}" for name in missing]

    errors = []
    tm, rules = config["time_model"], config["rules"]

    windows = tm["day_windows"]
    if (windows[0]["start"] != 0 or windows[-1]["end"] != tm["day_minutes"]
            or any(a["end"] != b["start"] for a, b in zip(windows, windows[1:]))):
        errors.append("day_windows muessen den Tag lueckenlos von 0 bis day_minutes abdecken")
    for key in ("time_dependent", "allowed_in_break", "allowed_overnight"):
        if set(rules[key]) != set(KINDS):
            errors.append(f"rules.{key} braucht genau die Arten {KINDS}")

    # Regel-Varianten, die in der Config beschreibbar, hier aber nicht umgesetzt sind
    unsupported = {
        "time_model.duration_rounding != 'ceil'": tm["duration_rounding"] != "ceil",
        "time_model.duration_fixed_at_start = false": not tm["duration_fixed_at_start"],
        "rules.non_preemptive = false": not rules["non_preemptive"],
        "rules.setup_requires_worker = true": rules["setup_requires_worker"],
        "rules.station_worker_mapping != 'one_to_one'":
            rules["station_worker_mapping"] != "one_to_one",
        "rules.one_task_per_resource_at_a_time = false":
            not rules["one_task_per_resource_at_a_time"],
        "rules.setup_directly_before_operation_on_same_resource = false":
            not rules["setup_directly_before_operation_on_same_resource"],
        "rules.job_operations_in_order = false": not rules["job_operations_in_order"],
    }
    errors += [f"nicht unterstuetzt: {name}" for name, bad in unsupported.items() if bad]
    return errors + _check_resource_factors(config) + _check_settings(config)


def _check_resource_factors(config: dict) -> list[str]:
    """time_model.resource_factors (optional): je Handarbeitsplatz und fuer die Ruestung eine Liste mit einem
    Faktor je Zeitfenster; Pausen an denselben Stellen wie in day_windows."""
    tm = config["time_model"]
    profiles = tm.get("resource_factors")
    if profiles is None:
        return []
    stations, setup = profiles.get("manual_stations"), profiles.get("setup")
    if not stations or setup is None:
        return ["time_model.resource_factors braucht manual_stations und setup"]
    errors = []
    breaks = [w["factor"] is None for w in tm["day_windows"]]
    named = [(f"manual_stations[{i}]", values) for i, values in enumerate(stations)] + [("setup", setup)]
    for name, values in named:
        if len(values) != len(breaks) or [v is None for v in values] != breaks:
            errors.append(f"time_model.resource_factors.{name} braucht je Zeitfenster einen Wert "
                          "und null genau in den Pausen")
        elif any(v is not None and v <= 0 for v in values):
            errors.append(f"time_model.resource_factors.{name}: Faktoren muessen positiv sein")
    for name, c in instance_classes(config).items():
        if isinstance(c.get("workers"), int) and c["workers"] > len(stations):
            errors.append(f"instance_classes.{name}: {c['workers']} Handarbeitsplaetze, aber nur "
                          f"{len(stations)} Profile in time_model.resource_factors.manual_stations")
    return errors


def _check_settings(config: dict) -> list[str]:
    """Instanzklassen, Datensaetze und Umgebung pruefen (nicht Teil des Fingerprints)."""
    errors = []
    if config["simulation"]["overflow_days"] < 0:
        errors.append("simulation.overflow_days darf nicht negativ sein")

    classes = instance_classes(config)
    if not classes:
        errors.append("instance_classes braucht mindestens eine Klasse")
    for name, c in classes.items():
        for key in ("jobs", "operations_per_job", "workers", "machines", "days"):
            if not isinstance(c.get(key), int) or c[key] < 1:
                errors.append(f"instance_classes.{name}.{key} muss eine positive ganze Zahl sein")

    splits = _strip_comments(config["data_splits"])
    if set(splits) != {"train", "val", "test"}:
        errors.append("data_splits braucht genau train, val und test")
    else:
        errors += _check_seed_ranges(splits)

    for section, key in (("experiments", "classes"), ("environment", "train_classes")):
        selection = config[section][key]
        if selection != "all" and (not isinstance(selection, list) or not selection
                                   or set(selection) - set(classes)):
            errors.append(f"{section}.{key} muss \"all\" oder eine Liste vorhandener Klassen sein")
    if config["experiments"]["split"] not in ("val", "test"):
        errors.append("experiments.split muss val oder test sein")

    env = config["environment"]
    if env["view"] not in ENVIRONMENT_VIEWS:
        errors.append(f"environment.view muss eines von {ENVIRONMENT_VIEWS} sein")
    if env["device"] != "auto" and not str(env["device"]).startswith(("cpu", "cuda")):
        errors.append("environment.device muss auto, cpu oder cuda sein")

    reward = config["reward"]
    levels = reward["levels"]
    if not levels or len(set(levels)) != len(levels) or set(levels) - set(REWARD_LEVELS):
        errors.append(f"reward.levels braucht verschiedene Werte aus {REWARD_LEVELS}")
    weights = reward["level_weights"]
    if weights is not None and (len(weights) != len(levels) or min(weights) <= 0):
        errors.append("reward.level_weights muss null sein oder je Ebene ein Gewicht > 0 haben")
    if reward["mode"] not in REWARD_MODES:
        errors.append(f"reward.mode muss eines von {REWARD_MODES} sein")

    model = config["model"]
    if model["variant"] not in MODEL_VARIANTS:
        errors.append(f"model.variant muss eines von {MODEL_VARIANTS} sein")
    if model["time_information"] not in TIME_INFORMATION:
        errors.append(f"model.time_information muss eines von {TIME_INFORMATION} sein")
    if model["num_heads"] < 1 or model["d_model"] % model["num_heads"]:
        errors.append("model.d_model muss durch model.num_heads teilbar sein")
    training = config["training"]
    for key in ("iterations", "instances_per_iteration", "ppo_epochs", "minibatch_size",
                "validate_every", "validation_instances"):
        if not isinstance(training[key], int) or training[key] < 1:
            errors.append(f"training.{key} muss eine positive ganze Zahl sein")
    if training["best_metric"] not in BEST_METRICS:
        errors.append(f"training.best_metric muss eines von {tuple(BEST_METRICS)} sein")
    if training["validation_decode"] not in VALIDATION_DECODES:
        errors.append(f"training.validation_decode muss eines von {VALIDATION_DECODES} sein")

    monitoring = config["monitoring"]
    for key in ("flush_seconds", "log_every_updates", "temporal_every_validations", "gantt_every_validations"):
        if not isinstance(monitoring[key], int) or monitoring[key] < 1:
            errors.append(f"monitoring.{key} muss eine positive ganze Zahl sein")
    if not isinstance(monitoring["gantt_instances"], int) or monitoring["gantt_instances"] < 0:
        errors.append("monitoring.gantt_instances darf nicht negativ sein")
    limit = monitoring["test_instances"]
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        errors.append("monitoring.test_instances muss null oder eine positive ganze Zahl sein")
    reference = monitoring["static_reference"]
    if reference is not None and set(reference) != {"sequencing", "routing", "view"}:
        errors.append("monitoring.static_reference muss null sein oder sequencing, routing und view haben")
    return errors


def _check_seed_ranges(splits: dict) -> list[str]:
    """Die Seeds der Datensaetze duerfen sich nicht ueberschneiden. Uebersprungene Seeds
    verlaengern einen Bereich, daher der Abstand von 10 * count."""
    errors = []
    first, second = sorted(("val", "test"), key=lambda name: splits[name]["seed_start"])
    for name in (first, second):
        if splits[name]["count"] < 1:
            errors.append(f"data_splits.{name}.count muss positiv sein")
    gap = 10 * splits[first]["count"]
    if splits[second]["seed_start"] < splits[first]["seed_start"] + gap:
        errors.append("data_splits: Seed-Bereiche von val und test liegen zu dicht beieinander")
    gap = 10 * splits[second]["count"]
    if splits["train"]["seed_start"] < splits[second]["seed_start"] + gap:
        errors.append("data_splits.train.seed_start muss hinter den Seeds von val und test liegen")
    return errors


def instance_classes(config: dict) -> dict[str, dict]:
    """Alle Instanzklassen aus der Konfiguration (ohne Kommentare)."""
    return _strip_comments(config["instance_classes"])


def selected_classes(config: dict, selection: str | list[str]) -> list[str]:
    """Klassennamen zu einer Auswahl: "all" oder eine Liste von Namen."""
    classes = instance_classes(config)
    if selection == "all":
        return list(classes)
    unknown = [name for name in selection if name not in classes]
    if unknown:
        raise ValueError(f"Unbekannte Instanzklasse(n): {unknown}. Vorhanden: {list(classes)}")
    return list(selection)


def resolve_path(relative: str | Path, config_path: Path | str = DEFAULT_CONFIG) -> Path:
    """Pfad aus der Konfiguration: relativ zum Ordner der Konfigurationsdatei."""
    return Path(config_path).resolve().parent / relative


def split_folder(config: dict, class_name: str, split: str,
                 config_path: Path | str = DEFAULT_CONFIG) -> Path:
    """Ordner mit den festen Instanzen einer Klasse (split = val oder test)."""
    folder = config["data_splits"][split]["folder"].replace("{class}", class_name)
    return resolve_path(folder, config_path)


# --------------------------------------------------------------------------
# Zeitmodell (alle Werte aus der Konfiguration)
# --------------------------------------------------------------------------

def horizon(config: dict, instance: dict) -> int:
    """Planungshorizont in Arbeitsminuten (Naechte ausgeblendet)."""
    return instance["days"] * config["time_model"]["day_minutes"]


def all_resources(instance: dict) -> list[str]:
    """Alle Ressourcen der Instanz: erst Handarbeitsplaetze, dann Maschinen."""
    return instance["resources"]["manual_stations"] + instance["resources"]["machines"]


def setup_resources(config: dict, instance: dict) -> list[str]:
    """Ressourcen, auf denen laut Regeln geruestet wird."""
    resources = list(instance["resources"]["machines"])
    if config["rules"]["setups_on_manual_stations"]:
        resources = instance["resources"]["manual_stations"] + resources
    return resources


def station_index(config: dict, instance: dict, resource: str | None) -> int | None:
    """Position von `resource` unter den Handarbeitsplaetzen, wenn es Profile je Platz gibt, sonst None."""
    if resource is None or "resource_factors" not in config["time_model"]:
        return None
    stations = instance["resources"]["manual_stations"]
    return stations.index(resource) if resource in stations else None


def factor_at(config: dict, t: int, kind: str | None = None, station: int | None = None) -> float | None:
    """Zeitfaktor zum Zeitpunkt t; None = Pause. Mit time_model.resource_factors gilt fuer Ruestungen das
    Ruestprofil und fuer manuelle Operationen das Profil des Handarbeitsplatzes `station`; sonst day_windows."""
    tm = config["time_model"]
    offset = t % tm["day_minutes"]
    profiles = tm.get("resource_factors")
    for index, w in enumerate(tm["day_windows"]):
        if w["start"] <= offset < w["end"]:
            if profiles and kind == "setup":
                return profiles["setup"][index]
            if profiles and kind == "manual" and station is not None:
                return profiles["manual_stations"][station][index]
            return w["factor"]
    raise ValueError(f"Kein Zeitfenster fuer Minute {t}")


def interval_violation(config: dict, instance: dict, kind: str,
                       start: int, end: int) -> str | None:
    """Prueft ein Intervall einer Taetigkeitsart gegen Horizont, Nacht und
    Pause. None = zulaessig, sonst eine Begruendung."""
    rules, day = config["rules"], config["time_model"]["day_minutes"]
    if start < 0 or end > horizon(config, instance):
        return "liegt ausserhalb des Planungshorizonts"
    if end == start:
        return None

    first_day, last_day = start // day, (end - 1) // day
    if not rules["allowed_overnight"][kind] and first_day != last_day:
        return "laeuft ueber ein Tagesende (nachts steht alles still)"

    if not rules["allowed_in_break"][kind]:
        breaks = [w for w in config["time_model"]["day_windows"] if w["factor"] is None]
        for d in range(first_day, last_day + 1):
            for w in breaks:
                if start < d * day + w["end"] and end > d * day + w["start"]:
                    return "ueberlappt die Pause"
    return None


def duration(config: dict, instance: dict, kind: str, base_minutes: int, start: int,
             resource: str | None = None) -> int:
    """Tatsaechliche Dauer einer Taetigkeit mit Start `start`; `resource` waehlt bei Profilen je
    Handarbeitsplatz das richtige Profil. Wirft ValueError, wenn sie zu diesem Zeitpunkt nicht starten darf."""
    if config["rules"]["time_dependent"][kind]:
        if not 0 <= start < horizon(config, instance):
            raise ValueError(f"{KIND_NAMES[kind]}: Start ausserhalb des Planungshorizonts")
        factor = factor_at(config, start, kind, station_index(config, instance, resource))
        if factor is None:
            raise ValueError(f"{KIND_NAMES[kind]} darf nicht in der Pause starten")
        result = math.ceil(base_minutes * factor - 1e-9)  # Dauer beim Start festgelegt
    else:
        result = base_minutes

    violation = interval_violation(config, instance, kind, start, start + result)
    if violation:
        raise ValueError(f"{KIND_NAMES[kind]} {violation}")
    return result


def _operation(instance: dict, op_id: str) -> dict | None:
    """Operation mit dieser ID oder None."""
    return next((o for o in instance["operations"] if o["id"] == op_id), None)


def processing_duration(config: dict, instance: dict, op_id: str, resource: str,
                        start: int) -> int:
    """Bearbeitungsdauer von `op_id` auf `resource` mit Start `start`.
    KeyError bei unbekannter Operation, ValueError bei Regelverstoss."""
    op = _operation(instance, op_id)
    if op is None:
        raise KeyError(f"Unbekannte Operation: {op_id}")
    if resource not in op["base_processing"]:
        raise ValueError(f"{op_id} darf nicht auf {resource} laufen")
    return duration(config, instance, op["kind"], op["base_processing"][resource], start, resource)


def setup_duration(config: dict, instance: dict, resource: str, previous: str | None,
                   next_op: str, start: int) -> int:
    """Ruestdauer auf `resource` von `previous` auf `next_op` mit Start `start`.
    previous=None = erste Operation auf der Ressource (Anfangsruestung)."""
    operation = _operation(instance, next_op)
    if operation is None or resource not in operation["base_processing"]:
        raise ValueError("Nachfolgeoperation ist auf Ressource nicht zulaessig")
    if resource not in setup_resources(config, instance):
        return 0
    if previous is None:
        initial = config["rules"]["initial_setup_minutes"]
        return duration(config, instance, "setup", initial, start) if initial else 0

    pair = next((s for s in instance["setups"]
                 if s["resource"] == resource and s["from_op"] == previous
                 and s["to_op"] == next_op), None)
    if pair is None:
        raise ValueError("Unzulaessiges oder fehlendes Operationspaar fuer Setup")
    return duration(config, instance, "setup", pair["base_minutes"], start)


# --------------------------------------------------------------------------
# Pruefung einer Instanz (Struktur)
# --------------------------------------------------------------------------

def validate_instance(config: dict, instance: dict) -> list[str]:
    """Strukturelle Pruefung einer Instanz. Leere Liste = in Ordnung.
    Ob ein Schedule zulaessig ist, prueft `check_schedule`."""
    expected = instance.get("problem_config", {}).get("fingerprint")
    if expected != config_fingerprint(config):
        return ["Instanz wurde mit anderen Regeln erzeugt als in der aktuellen "
                "problem_config (Fingerprint passt nicht oder altes Instanzformat)"]

    operations = {o["id"]: o for o in instance["operations"]}
    return (_check_header(instance, operations)
            + _check_jobs(config, instance, operations)
            + _check_operations(instance, operations)
            + _check_setup_pairs(config, instance, operations))


def _check_header(instance: dict, operations: dict) -> list[str]:
    """Tage, Zuordnung Handarbeitsplatz -> Worker, eindeutige Operations-IDs."""
    errors = []
    resources = instance["resources"]
    if instance["days"] < 1:
        errors.append("days muss mindestens 1 sein")
    if (set(instance["station_worker"]) != set(resources["manual_stations"])
            or sorted(instance["station_worker"].values()) != sorted(resources["workers"])):
        errors.append("station_worker muss jedem Handarbeitsplatz genau einen Worker zuordnen")
    if len(operations) != len(instance["operations"]):
        errors.append("Doppelte Operations-ID")
    return errors


def _check_jobs(config: dict, instance: dict, operations: dict) -> list[str]:
    """Release/Deadline im Horizont, jede Operation gehoert genau einem Job."""
    errors = []
    all_ids = []
    for job in instance["jobs"]:
        if not 0 <= job["release"] <= job["due"] <= horizon(config, instance):
            errors.append(f"{job['id']}: release/due ausserhalb des Horizonts")
        if not job["operations"]:
            errors.append(f"{job['id']}: hat keine Operationen")
        for op_id in job["operations"]:
            if op_id not in operations or operations[op_id]["job"] != job["id"]:
                errors.append(f"{job['id']}: Operation {op_id} passt nicht zum Job")
            all_ids.append(op_id)
    if sorted(all_ids) != sorted(operations):
        errors.append("Jede Operation muss genau einem Job angehoeren")
    return errors


def _check_operations(instance: dict, operations: dict) -> list[str]:
    """Art der Operation passt zu ihren Ressourcen, Basisdauern sind positiv."""
    errors = []
    machines = set(instance["resources"]["machines"])
    stations = set(instance["resources"]["manual_stations"])
    for op in operations.values():
        allowed = op["base_processing"]
        target = stations if op["kind"] == "manual" else machines
        if op["kind"] not in OPERATION_KINDS:
            errors.append(f"{op['id']}: unbekannte Art {op['kind']}")
        elif not allowed or not set(allowed) <= target:
            errors.append(f"{op['id']}: unzulaessige Ressourcen {sorted(allowed)}")
        if any(v <= 0 for v in allowed.values()):
            errors.append(f"{op['id']}: Basisdauern muessen positiv sein")
    return errors


def _check_setup_pairs(config: dict, instance: dict, operations: dict) -> list[str]:
    """Ruestzeiten nur auf Ruest-Ressourcen und nur fuer passende Operationen."""
    errors = []
    valid_setup_resources = set(setup_resources(config, instance))
    for pair in instance["setups"]:
        where = f"Setup {pair['from_op']}->{pair['to_op']} auf {pair['resource']}"
        if pair["resource"] not in valid_setup_resources:
            errors.append(f"{where}: auf dieser Ressource wird laut Regeln nicht geruestet")
        elif pair["from_op"] == pair["to_op"] or pair["base_minutes"] <= 0:
            errors.append(f"{where}: ungueltig")
        elif any(pair["resource"] not in operations.get(o, {}).get("base_processing", {})
                 for o in (pair["from_op"], pair["to_op"])):
            errors.append(f"{where}: Operation darf nicht auf dieser Ressource laufen")
    return errors


# --------------------------------------------------------------------------
# Pruefung eines Schedules (Zulaessigkeit)
# --------------------------------------------------------------------------

def check_schedule(config: dict, instance: dict, schedule: list[dict]) -> list[str]:
    """Prueft einen Schedule gegen alle harten Regeln der Konfiguration.
    Leere Liste = zulaessig.

    Format je Eintrag: {"op", "resource", "start", "end"} und auf Ressourcen
    mit Ruestung zusaetzlich {"setup_from", "setup_start", "setup_end"}."""
    by_op = {e["op"]: e for e in schedule}
    if len(by_op) != len(schedule) or set(by_op) != {o["id"] for o in instance["operations"]}:
        return ["Jede Operation muss genau einmal eingeplant sein"]

    return (_check_processing(config, instance, schedule)
            + _check_job_order_and_due(config, instance, by_op)
            + _check_resource_sequences(config, instance, schedule))


def _check_processing(config: dict, instance: dict, schedule: list[dict]) -> list[str]:
    """Jede Operation: erlaubte Ressource, zulaessiger Zeitpunkt, richtige Dauer."""
    errors = []
    for e in schedule:
        try:
            expected = processing_duration(config, instance, e["op"], e["resource"], e["start"])
            if e["end"] - e["start"] != expected:
                errors.append(f"{e['op']}: Dauer {e['end'] - e['start']} statt {expected}")
        except (ValueError, KeyError) as exc:
            errors.append(f"{e['op']}: {exc}")
    return errors


def _check_job_order_and_due(config: dict, instance: dict, by_op: dict) -> list[str]:
    """Reihenfolge der Operationen eines Jobs, Release und - nur wenn
    rules.due_dates_hard - Deadline. Bei weichen Deadlines misst
    evaluation.py die Verspaetung."""
    errors = []
    rules = config["rules"]
    for job in instance["jobs"]:
        previous_end = job["release"]
        for op_id in job["operations"]:
            e = by_op[op_id]
            if rules["job_operations_in_order"] and e["start"] < previous_end:
                errors.append(f"{op_id}: startet vor Ende des Vorgaengers bzw. vor Release")
            previous_end = max(previous_end, e["end"])
        if rules["due_dates_hard"] and previous_end > job["due"]:
            errors.append(f"{job['id']}: Deadline verletzt ({previous_end} > {job['due']})")
    return errors


def _check_resource_sequences(config: dict, instance: dict, schedule: list[dict]) -> list[str]:
    """Je Ressource: Ruestungen korrekt und keine Ueberlappungen."""
    errors = []
    with_setup = set(setup_resources(config, instance))
    for resource in all_resources(instance):
        sequence = sorted((e for e in schedule if e["resource"] == resource),
                          key=lambda e: e["start"])
        previous = None
        for e in sequence:
            busy_from = e["start"]
            if resource in with_setup:
                setup_errors, busy_from = _check_setup(config, instance, resource, e, previous)
                errors += setup_errors
            if previous is not None and busy_from < previous["end"]:
                errors.append(f"{resource}: {previous['op']} und {e['op']} ueberlappen")
            previous = e
    return errors


def _check_setup(config: dict, instance: dict, resource: str, entry: dict,
                 previous: dict | None) -> tuple[list[str], int]:
    """Ruestung vor `entry` auf `resource`: richtiger Vorgaenger, zulaessiger
    Zeitpunkt, richtige Dauer, endet vor dem Bearbeitungsstart.
    Gibt (Verstoesse, Beginn der Belegung der Ressource) zurueck."""
    expected_from = previous["op"] if previous else None
    if "setup_start" not in entry or entry.get("setup_from") != expected_from:
        return [f"{entry['op']}: Ruestung fehlt oder falscher Vorgaenger "
                f"auf {resource} (erwartet {expected_from})"], entry["start"]

    errors = []
    try:
        expected = setup_duration(config, instance, resource, entry["setup_from"],
                                  entry["op"], entry["setup_start"])
        if entry["setup_end"] - entry["setup_start"] != expected:
            errors.append(f"{entry['op']}: Ruestdauer "
                          f"{entry['setup_end'] - entry['setup_start']} statt {expected}")
    except ValueError as exc:
        errors.append(f"{entry['op']}: {exc}")
    if entry["setup_end"] > entry["start"]:
        errors.append(f"{entry['op']}: Ruestung endet nach Bearbeitungsstart")
    return errors, entry["setup_start"]


# --------------------------------------------------------------------------
# Kommandozeile
# --------------------------------------------------------------------------

def instance_files(paths: list[Path]) -> list[Path]:
    """Dateien und Ordner (alle *.json darin, auch in Unterordnern) zu einer Dateiliste aufloesen."""
    files = []
    for path in paths:
        files += sorted(path.rglob("*.json")) if path.is_dir() else [path]
    return files


def load_schedule(path: Path) -> list[dict]:
    """Schedule-Datei lesen: JSON-Liste oder Objekt mit Schluessel "schedule"."""
    data = load_json(path)
    return data["schedule"] if isinstance(data, dict) else data


def main() -> int:
    parser = argparse.ArgumentParser(description="Instanzen und Schedules pruefen")
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

    failed = 0
    for path in files:
        instance = load_json(path)
        errors = validate_instance(config, instance)
        if not errors:
            schedule = (load_schedule(args.schedule) if args.schedule
                        else instance["reference_schedule"])
            errors = check_schedule(config, instance, schedule)
        if errors:
            failed += 1
            print(f"FEHLER  {path}")
            for error in errors:
                print(f"        - {error}")
        else:
            print(f"OK      {path}")

    print(f"\n{len(files) - failed} von {len(files)} zulaessig")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
