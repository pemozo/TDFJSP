"""Prioritaetsregeln fuer den Dispatcher.

Sequencing-Regeln waehlen aus den Operationen, die jetzt starten koennen,
die naechste. Routing-Regeln waehlen fuer diese Operation einen der jetzt
moeglichen Schedule-Eintraege (eine freie zulaessige Ressource).

Beide arbeiten auf dem simulator.SimulationState. Die Eintraege aus
state.options(...) enthalten Ruestung und Dauer bereits in der Dauer-Sicht
des Verfahrens (statisch: Faktor 1.0; zeitbewusst: echte Dauer).

Gleichstaende werden deterministisch aufgeloest (Reihenfolge der Jobs bzw.
Ressourcen in der Instanz), damit Ergebnisse reproduzierbar sind.

Weitere Regeln (z. B. MST, CR, LWL) koennen hier ergaenzt und in
SEQUENCING_RULES bzw. ROUTING_RULES eingetragen werden.
"""
from __future__ import annotations

from simulator import plan_operation


# --------------------------------------------------------------------------
# Sequencing: welche Operation als naechste?
# --------------------------------------------------------------------------

def edd(state, candidates: list[dict]) -> dict:
    """Earliest Due Date: Operation des Jobs mit der fruehesten Deadline.
    Gleichstand: Reihenfolge der Jobs in der Instanz."""
    return min(candidates, key=lambda op: (state.jobs[op["job"]]["due"],
                                           state.job_index[op["job"]]))


# --------------------------------------------------------------------------
# Routing: welche Ressource?
# --------------------------------------------------------------------------

def ect(state, operation: dict, options: list[dict]) -> dict:
    """Earliest Completion Time: Eintrag mit dem fruehesten Ende
    (Ruestung + Bearbeitung). Gleichstand: Reihenfolge der Ressourcen."""
    return min(options, key=lambda e: (e["end"], state.resource_index[e["resource"]]))


def spt(state, operation: dict, options: list[dict]) -> dict:
    """Shortest Processing Time: Eintrag mit der kuerzesten Bearbeitungszeit
    (ohne Ruestung). Gleichstand: frueheres Ende, dann Reihenfolge der Ressourcen."""
    return min(options, key=lambda e: (e["end"] - e["start"], e["end"],
                                       state.resource_index[e["resource"]]))


def slot(state, operation: dict, options: list[dict]) -> dict | None:
    """Kuerzester Zeitslot (nur manuelle Operationen; alle anderen wie ECT).

    Betrachtet alle zulaessigen Handarbeitsplaetze zum jetzigen Zeitpunkt und zu jeder
    Fenstergrenze innerhalb von experiments.slot_lookahead_minutes (Standard 120), auf
    Plaetzen, die bis dahin frei sind. Ist die kuerzeste Dauer jetzt erreichbar, startet die
    Operation jetzt (Gleichstand: frueheres Ende). Liegt der kuerzeste Slot spaeter, wartet
    die Operation (Rueckgabe None); der Dispatcher prueft sie beim naechsten Ereignis erneut,
    andere Operationen duerfen den Platz bis dahin nutzen. Die Sicht des Verfahrens bestimmt
    die Dauern: statisch sind alle Slots gleich lang, dann verhaelt sich die Regel wie ECT."""
    if operation["kind"] != "manual":
        return ect(state, operation, options)
    def length(entry):
        return entry["end"] - entry.get("setup_start", entry["start"])
    best_now = min(options, key=lambda e: (length(e), e["end"], state.resource_index[e["resource"]]))
    lookahead = state.config.get("experiments", {}).get("slot_lookahead_minutes", 120)
    for resource, base in operation["base_processing"].items():
        setup = state.setup_of(resource, operation["id"])
        for t in state.boundaries_between(state.time, state.time + lookahead):
            if state.resource_free[resource] > t:
                continue
            try:
                setup_len, process_len = plan_operation(state.config, state.instance, operation["kind"], base,
                                                        setup[1] if setup else 0, t, resource)
            except ValueError:
                continue
            if setup_len + process_len < length(best_now):
                return None
    return best_now


SEQUENCING_RULES = {"EDD": edd}
ROUTING_RULES = {"ECT": ect, "SPT": spt, "SLOT": slot}
# Zuordnungs-Regeln, die eine Operation zurueckhalten koennen (Rueckgabe None). Ihre geplanten
# Startzeiten bleiben bei der Ausfuehrung erhalten, sonst wuerde das gewollte Warten verschwinden.
HOLDING_ROUTING_RULES = {"SLOT"}
