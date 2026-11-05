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

from evaluation import makespan
from schedule_validator import all_resources, factor_at, station_index


def time_metrics(rules: dict, instance: dict, schedule: list[dict],
                 voluntary_idle: int | None) -> dict[str, float | None]:
    """voluntary_idle = None: Wartezeit ist nicht gemessen (z. B. Regeln), unavoidable bleibt dann auch offen."""
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
            "mean_setup_factor": sum(setup_factors) / len(setup_factors) if setup_factors else None}


def worker_metrics(rules: dict, instance: dict, schedule: list[dict]) -> dict[str, int]:
    """Worker-Zeit in Minuten: worker_busy_time = Dauer aller manuellen Operationen, manual_extra_time = der
    Anteil davon, der nur durch den Zeitfaktor entsteht, setup_extra_time = dasselbe fuer die Ruestungen.
    Sinkt worker_busy_time gegenueber einem Vergleichsplan, ist Worker-Zeit frei geworden."""
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
