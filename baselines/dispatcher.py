"""Dispatcher: Plan aus einer Reihenfolge- und einer Zuordnungs-Regel, danach Ausfuehrung.

Ablauf (siehe simulator.py fuer die festgelegten Annahmen):

1. Planen (ereignisgesteuert, non-delay) in der Dauer-Sicht des Verfahrens:
       state = SimulationState(config, instance, view)
       solange Operationen offen:
           Kandidaten = Operationen, die jetzt auf einer freien Ressource starten koennen
           keine Kandidaten -> Uhr zum naechsten Ereignis
           Operation  = Reihenfolge-Regel(state, Kandidaten)        z. B. EDD
           Eintrag    = Zuordnungs-Regel(state, Operation, Optionen) z. B. ECT
           einplanen
   view = "static":     Faktor 1.0, Pause und Arbeitsende bekannt
   view = "time_aware": echte Dauern

2. Ausfuehren: Zuordnung und Reihenfolge je Ressource werden mit den echten
   Dauern ausgefuehrt (simulator.execute_plan, mit Zusatztag als Ueberlauf).
   Erst dieser ausgefuehrte Schedule wird bewertet. Fuer "time_aware" stimmt
   er mit dem Plan ueberein oder ist hoechstens frueher.
"""
from __future__ import annotations

from baselines.rules import ROUTING_RULES, SEQUENCING_RULES
from simulator import (DURATION_VIEWS, SimulationError, SimulationState,
                       execute_plan, plan_from_schedule)


def build_plan(config: dict, instance: dict, sequencing: str, routing: str,
               view: str) -> list[dict]:
    """Plan der Regelkombination in der Dauer-Sicht `view` (Schritt 1)."""
    if view not in DURATION_VIEWS:
        raise ValueError(f"Unbekannte Dauer-Sicht '{view}', erlaubt: {DURATION_VIEWS}")
    choose_operation = SEQUENCING_RULES[sequencing]
    choose_entry = ROUTING_RULES[routing]

    state = SimulationState(config, instance, view)
    while not state.done():
        candidates = state.candidate_operations()
        if not candidates:
            if not state.advance():
                raise SimulationError(f"{sequencing}+{routing} ({view}): keine Operation "
                                      "passt mehr in den Horizont")
            continue
        operation = choose_operation(state, candidates)
        state.apply(choose_entry(state, operation, state.options(operation["id"])))
    return state.schedule


def run_dispatcher(config: dict, instance: dict, sequencing: str, routing: str,
                   view: str) -> dict:
    """Plan bauen und mit den echten Dauern ausfuehren.

    sequencing: Schluessel aus rules.SEQUENCING_RULES, z. B. "EDD"
    routing   : Schluessel aus rules.ROUTING_RULES, z. B. "ECT" oder "SPT"
    view      : "static" oder "time_aware"
    Rueckgabe : {"planned": Plan in der Sicht des Verfahrens,
                 "executed": Schedule, wie er mit echten Dauern ablaeuft}"""
    planned = build_plan(config, instance, sequencing, routing, view)
    executed = execute_plan(config, instance, plan_from_schedule(planned))
    return {"planned": planned, "executed": executed}
