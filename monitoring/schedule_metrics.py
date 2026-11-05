"""Kennzahlen eines Plans, die evaluation.py nicht abdeckt: Zeiten, Zeitfaktoren, Leerlauf, Zeitabhaengigkeit.

Reine Funktionen auf Instanz und Schedule. Dauern und Faktoren gelten in der Sicht der uebergebenen
Konfiguration (simulator.view_config).

Leerlauf in Ressourcen-Minuten, jede Ressource von 0 bis zum Makespan:
    voluntary_idle_time     die Ressource haette anfangen koennen, der Agent hat WAIT gewaehlt
    unavoidable_idle_time   der Rest: In diesem Moment konnte auf der Ressource nichts starten, auch in
                            der Pause und nach ihrer letzten Operation
Die beiden werden nie zusammengezaehlt.
"""
from __future__ import annotations

import math
import hashlib
import json

from evaluation import makespan
from schedule_validator import (all_resources, factor_at, instance_config, interval_violation,
                                station_index, time_profile_id)


METRIC_DEFINITIONS = {
    "aggregation": "Arithmetic mean over completed instances; utilization is the mean of per-instance ratios. Job compliance is pooled over jobs.",
    "machine_processing_time": "Sum of automatic processing minutes on machines; excludes setup.",
    "machine_setup_time": "Sum of setup minutes on machines only.",
    "machine_idle_time": "machines * makespan - machine processing - machine setup, from time zero to each plan's makespan.",
    "machine_productive_utilization": "machine_processing_time / (machines * makespan); setup never increases productive utilization.",
    "machine_occupied_utilization": "(machine processing + machine setup) / (machines * makespan).",
    "machine_flow_time": "Sum of automatic-operation completion times; lower values indicate earlier machine output without rewarding slow routing or extra setup.",
    "manual_base_time": "Sum of assigned-resource base durations of manual operations.",
    "manual_extra_time": "Actual manual duration minus assigned base duration, including rounding.",
    "weighted_manual_factor": "Manual start-profile factors weighted by assigned base durations; excludes duration rounding.",
    "manual_profile_regret": "Actual manual duration minus sum of smallest calendar-feasible rounded durations on each assigned station. Ignores release, precedence and competition; a diagnostic lower-bound gap, not achievable schedule regret.",
    "manual_favourable_start_share": "Share of manual starts at the lowest calendar-feasible factor for that operation's base duration on its assigned station.",
    "manual_favourable_base_share": "Share of manual base work started at that station/operation's lowest calendar-feasible factor.",
    "time_profile_hash": "SHA256 of effective daily boundaries and actual per-station/setup profiles; instance profile ID is recorded separately.",
}
PLAN_METRIC_NAMES = (
    "machine_processing_time", "machine_setup_time", "machine_idle_time",
    "machine_productive_utilization", "machine_occupied_utilization",
    "machine_flow_time", "machine_flow_time_mean_operation",
    "manual_base_time", "manual_extra_time", "weighted_manual_factor", "manual_profile_regret",
    "manual_favourable_start_share", "manual_favourable_base_share",
)


def profile_metadata(config: dict, instance: dict) -> dict[str, str]:
    """Identify actual profiles independently of the agent's information ablation."""
    rules = instance_config(config, instance)
    tm = rules["time_model"]
    windows = tm["day_windows"]
    profiles = tm.get("resource_factors")
    stations = instance.get("resources", {}).get("manual_stations", [])
    common = [w["factor"] for w in windows]
    payload = {"day_minutes": tm["day_minutes"], "windows": [[w["start"], w["end"]] for w in windows],
               "manual_stations": profiles["manual_stations"][:len(stations)] if profiles else [common] * len(stations),
               "setup": profiles["setup"] if profiles else common}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    profile_id = time_profile_id({key: payload[key] for key in ("manual_stations", "setup")})
    return {"time_profile_id": instance.get("time_profile_id", profile_id), "time_profile_hash": digest}


def time_metrics(rules: dict, instance: dict, schedule: list[dict],
                 voluntary_idle: int | None) -> dict[str, float | None]:
    """voluntary_idle = None: Wartezeit ist nicht gemessen (z. B. Regeln), unavoidable bleibt dann auch offen."""
    rules = instance_config(rules, instance)
    kinds = {operation["id"]: operation["kind"] for operation in instance["operations"]}
    dependent = rules["rules"]["time_dependent"]
    setups = [e for e in schedule if e.get("setup_end", 0) > e.get("setup_start", 0)]
    processing = sum(e["end"] - e["start"] for e in schedule)
    setup = sum(e["setup_end"] - e["setup_start"] for e in setups)
    idle = len(all_resources(instance)) * makespan(schedule) - processing - setup

    process_factors = [factor_at(rules, e["start"], kinds[e["op"]], station_index(rules, instance, e["resource"]))
                       for e in schedule if dependent[kinds[e["op"]]]]
    setup_factors = [factor_at(rules, e["setup_start"], "setup") for e in setups] if dependent["setup"] else []
    return {"processing_time_total": processing,
            "setup_time_total": setup,
            "voluntary_idle_time": voluntary_idle,
            "unavoidable_idle_time": None if voluntary_idle is None else idle - voluntary_idle,
            "mean_processing_factor": sum(process_factors) / len(process_factors) if process_factors else None,
            "mean_setup_factor": sum(setup_factors) / len(setup_factors) if setup_factors else None,
            **machine_metrics(instance, schedule)}


def machine_metrics(instance: dict, schedule: list[dict]) -> dict[str, float | int | None]:
    """Machine-only utilization on [0, makespan]; distinguish processing from setup."""
    machines = set(instance.get("resources", {}).get("machines", instance.get("machines", [])))
    entries = [entry for entry in schedule if entry["resource"] in machines]
    processing = sum(e["end"] - e["start"] for e in entries)
    setup = sum(e.get("setup_end", 0) - e.get("setup_start", 0) for e in entries)
    capacity = len(machines) * makespan(schedule)
    return {"machine_processing_time": processing, "machine_setup_time": setup,
            "machine_idle_time": capacity - processing - setup,
            "machine_productive_utilization": processing / capacity if capacity else None,
            "machine_occupied_utilization": (processing + setup) / capacity if capacity else None,
            "machine_flow_time": sum(e["end"] for e in entries),
            "machine_flow_time_mean_operation": sum(e["end"] for e in entries) / len(entries) if entries else None}


def worker_metrics(rules: dict, instance: dict, schedule: list[dict]) -> dict[str, int]:
    """Worker-Zeit in Minuten: worker_busy_time = Dauer aller manuellen Operationen, manual_extra_time = der
    Anteil davon, der nur durch den Zeitfaktor entsteht, setup_extra_time = dasselbe fuer die Ruestungen.
    Sinkt worker_busy_time gegenueber einem Vergleichsplan, ist Worker-Zeit frei geworden."""
    rules = instance_config(rules, instance)
    operations = {operation["id"]: operation for operation in instance["operations"]}
    setups = {(s["resource"], s["from_op"], s["to_op"]): s["base_minutes"] for s in instance["setups"]}
    busy = manual_extra = setup_extra = 0
    for entry in schedule:
        operation = operations[entry["op"]]
        if operation["kind"] == "manual":
            length = entry["end"] - entry["start"]
            busy += length
            manual_extra += length - operation["base_processing"][entry["resource"]]
        length = entry.get("setup_end", 0) - entry.get("setup_start", 0)
        if length > 0:
            base = (rules["rules"]["initial_setup_minutes"] if entry["setup_from"] is None
                    else setups[(entry["resource"], entry["setup_from"], entry["op"])])
            setup_extra += length - base
    return {"worker_busy_time": busy, "manual_extra_time": manual_extra, "setup_extra_time": setup_extra}


def manual_profile_metrics(rules: dict, instance: dict, schedule: list[dict]) -> dict[str, float | int | None]:
    """Manual work versus a calendar-only lower bound on the assigned station.

    For each base duration, consider starts at daily window boundaries and exclude
    candidates crossing a forbidden break/day/horizon. Precedence, releases and
    resource competition are intentionally ignored; this is not optimal-schedule regret.
    """
    rules = instance_config(rules, instance)
    operations = {operation["id"]: operation for operation in instance["operations"]}
    dependent = rules["rules"]["time_dependent"]["manual"]
    windows = rules["time_model"]["day_windows"]
    total_base = weighted_factor = actual = lower_bound = favourable_base = count = favourable = 0
    choices = {}
    for entry in schedule:
        operation = operations[entry["op"]]
        if operation["kind"] != "manual":
            continue
        resource = entry["resource"]
        station = station_index(rules, instance, resource)
        base = operation["base_processing"][resource]
        key = (station, base)
        if key not in choices:
            candidates = []
            for window in windows:
                factor = factor_at(rules, window["start"], "manual", station) if dependent else 1.0
                if factor is None:
                    continue
                length = math.ceil(base * factor - 1e-9)
                if interval_violation(rules, instance, "manual", window["start"], window["start"] + length) is None:
                    candidates.append((factor, length))
            if not candidates:
                raise ValueError(f"No calendar-feasible manual duration on {resource} for base {base}")
            choices[key] = min(candidates)
        minimum_factor, minimum_duration = choices[key]
        factor = factor_at(rules, entry["start"], "manual", station) if dependent else 1.0
        at_minimum = math.isclose(factor, minimum_factor, rel_tol=1e-9, abs_tol=1e-9)
        total_base += base
        weighted_factor += base * factor
        actual += entry["end"] - entry["start"]
        lower_bound += minimum_duration
        count += 1
        favourable += int(at_minimum)
        favourable_base += base if at_minimum else 0
    return {"manual_base_time": total_base,
            "weighted_manual_factor": weighted_factor / total_base if total_base else None,
            "manual_profile_regret": actual - lower_bound,
            "manual_favourable_start_share": favourable / count if count else None,
            "manual_favourable_base_share": favourable_base / total_base if total_base else None}


def start_shares(rules: dict, schedule: list[dict]) -> dict[str, float]:
    """Anteil der gestarteten Operationen je Tagesfenster: start_share_<Fenster> -> Anteil."""
    windows = rules["time_model"]["day_windows"]
    day = rules["time_model"]["day_minutes"]
    counts = {window["label"]: 0 for window in windows}
    for entry in schedule:
        minute = entry["start"] % day
        counts[next(w["label"] for w in windows if w["start"] <= minute < w["end"])] += 1
    return {f"start_share_{label}": count / max(len(schedule), 1) for label, count in counts.items()}


def _factor_spread(windows: list[dict], factors: list[float | None]) -> float:
    """Variationskoeffizient eines Profils, zeitgewichtet; Pausen zaehlen nicht."""
    pairs = [(w["end"] - w["start"], f) for w, f in zip(windows, factors) if f is not None]
    length = sum(minutes for minutes, _ in pairs)
    mean = sum(minutes * f for minutes, f in pairs) / length
    variance = sum(minutes * (f - mean) ** 2 for minutes, f in pairs) / length
    return math.sqrt(variance) / mean


def temporal_variability(rules: dict, instance: dict) -> float:
    """Wie stark die Dauern einer Instanz rechnerisch vom Startzeitpunkt abhaengen: Anteil zeitabhaengiger
    Basisarbeit (Operationen und mittlere Ruestung) mal Variationskoeffizient der Fensterfaktoren, zeitgewichtet.
    Mit Profilen je Handarbeitsplatz zaehlt das Mittel ueber die Plaetze der Instanz, fuer Ruestungen das Ruestprofil.
    0 = der Zeitpunkt aendert keine Dauer. Die Zahl sagt nichts darueber, ob eine zeitbewusste
    Entscheidung bessere Plaene liefert; dafuer braeuchte es eine verlaessliche Referenz."""
    rules = instance_config(rules, instance)
    windows = rules["time_model"]["day_windows"]
    spread = _factor_spread(windows, [w["factor"] for w in windows])
    spreads = {"manual": spread, "automatic": spread, "setup": spread}
    profiles = rules["time_model"].get("resource_factors")
    if profiles:
        stations = profiles["manual_stations"][:len(instance["resources"]["manual_stations"])]
        spreads["manual"] = sum(_factor_spread(windows, row) for row in stations) / len(stations)
        spreads["setup"] = _factor_spread(windows, profiles["setup"])

    dependent_rules = rules["rules"]["time_dependent"]
    total = dependent = 0.0
    for operation in instance["operations"]:
        base = operation["base_processing"].values()
        work = sum(base) / len(base)
        total += work
        dependent += work * spreads[operation["kind"]] if dependent_rules[operation["kind"]] else 0.0
    by_target: dict[str, list[int]] = {}
    for setup in instance["setups"]:
        by_target.setdefault(setup["to_op"], []).append(setup["base_minutes"])
    setup_work = sum(sum(values) / len(values) for values in by_target.values())
    total += setup_work
    dependent += setup_work * spreads["setup"] if dependent_rules["setup"] else 0.0
    return dependent / total if total else 0.0


def relative_gain(static: float | None, temporal: float | None, higher_is_better: bool = False) -> float | None:
    """Verbesserung in % gegenueber static; None, wenn static fehlt oder 0 ist."""
    if static is None or temporal is None or static == 0:
        return None
    return ((temporal - static) if higher_is_better else (static - temporal)) / static * 100


def gain_metrics(rows: list[dict], static_rows: list[dict]) -> dict[str, float | None]:
    """temporal_gain und relative Verbesserungen auf den gemeinsamen, beiderseits ausfuehrbaren Instanzen.

    Bezugsgroesse ist das Mittel ueber diese Instanzen: Eine einzelne Instanz ohne Verspaetung haette sonst
    keinen definierten Wert. Beide Zeilensaetze muessen mit den echten Dauern ausgefuehrt sein."""
    static_by_id = {row["instance_id"]: row for row in static_rows if row.get("feasible")}
    pairs = [(row, static_by_id[row["instance_id"]]) for row in rows
             if row.get("feasible") and row["instance_id"] in static_by_id]
    if not pairs:
        return {"temporal_gain": None, "tardiness_gain": None, "compliance_gain": None, "gain_instances": 0}

    def mean(index: int, key: str) -> float:
        return sum(pair[index][key] for pair in pairs) / len(pairs)

    def compliance(index: int) -> float:
        return 1 - sum(pair[index]["late_jobs"] for pair in pairs) / sum(pair[index]["jobs"] for pair in pairs)

    return {"temporal_gain": relative_gain(mean(1, "makespan"), mean(0, "makespan")),
            "tardiness_gain": relative_gain(mean(1, "total_tardiness"), mean(0, "total_tardiness")),
            "compliance_gain": relative_gain(compliance(1), compliance(0), higher_is_better=True),
            "gain_instances": len(pairs)}
