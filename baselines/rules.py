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


SEQUENCING_RULES = {"EDD": edd}
ROUTING_RULES = {"ECT": ect, "SPT": spt}
