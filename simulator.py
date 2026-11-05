"""Simulator: gemeinsamer Ausfuehrungskern fuer Generator, Baselines und RL-Umgebung.

Der Simulator ist die "Welt" der Fertigung. Er weiss, wann welche Ressource
frei ist, welche Operation zuletzt auf einer Maschine lief (fuer die
Ruestzeit) und wie weit jeder Job ist. Ob ein Zeitpunkt zulaessig ist,
entscheidet ausschliesslich schedule_validator.duration, also die Regeln aus
problem_config.json.

Festgelegte Annahmen fuer Planung und Ausfuehrung:

1. Dauer-Sichten (DURATION_VIEWS)
   "time_aware": echte Dauern (Zeitfaktor zum Startzeitpunkt).
   "static"    : Faktor immer 1.0; Pause und Arbeitsende sind aber bekannt.
                 Ein statisches Verfahren kennt also den Kalender, nicht aber
                 die zeitabhaengige Leistung. Der Vergleich misst damit genau
                 den Effekt der zeitabhaengigen Leistung.
2. Ueberlauf (simulation.overflow_days in problem_config.json)
   Planung und Ausfuehrung duerfen Zusatztage nach dem Horizont der
   Instanz nutzen. Passt ein Plan in der Realitaet nicht mehr in den Tag,
   laeuft er in den Folgetag ueber; die Verspaetung wird normal gemessen.
   Die Deadlines der Instanz bleiben unveraendert.
3. Ruesten
   Die Ruestung beginnt, wenn Ressource und Job bereit sind, die Bearbeitung
   folgt direkt darauf. Passen Ruestung und Bearbeitung zu diesem Zeitpunkt
   nicht (Pause, Arbeitsende), beginnen beide zum naechsten zulaessigen
   Zeitpunkt. Gilt fuer alle Verfahren gleich.
4. Ereignisgesteuerte Planung (SimulationState)
   Die Uhr springt von Ereignis zu Ereignis (Ressource wird frei, Job wird
   bereit, Zeitfenster beginnt). Zu jedem Zeitpunkt sind nur Paare aus
   (bereite Operation, freie zulaessige Ressource) waehlbar, die jetzt
   regelkonform starten koennen (non-delay). Die Regeln bzw. der Agent
   waehlen daraus; gibt es keine Wahl, springt die Uhr weiter.

Ausfuehrung eines Plans (execute_plan): Zuordnung und Reihenfolge je
Ressource werden uebernommen, die Uhrzeiten mit den echten Dauern neu
berechnet (fruehester zulaessiger Start nach Ressource und Vorgaenger).
So zeigt sich, was ein mit statischen Zeiten gebauter Plan in der
Realitaet bewirkt.

Schedule-Eintraege haben das Format von schedule_validator.check_schedule:
    {"op", "resource", "start", "end"} und auf Ruest-Ressourcen zusaetzlich
    {"setup_from", "setup_start", "setup_end"}
"""
from __future__ import annotations

import copy

from schedule_validator import all_resources, duration, horizon, setup_resources

DURATION_VIEWS = ("time_aware", "static")


class SimulationError(RuntimeError):
    """Ein Plan laesst sich auch im verlaengerten Horizont nicht ausfuehren."""


# --------------------------------------------------------------------------
# Hilfsfunktionen (auch vom Generator genutzt)
# --------------------------------------------------------------------------

def execution_instance(config: dict, instance: dict, overflow_days: int | None = None) -> dict:
    """Instanz mit Ausfuehrungshorizont: days + overflow_days (None = Wert aus der
    Konfiguration). Nur fuer Planung, Ausfuehrung und Pruefung ausgefuehrter
    Plaene; Deadlines bleiben gleich."""
    if overflow_days is None:
        overflow_days = config["simulation"]["overflow_days"]
    return {**instance, "days": instance["days"] + overflow_days}


def view_config(config: dict, view: str) -> dict:
    """Konfiguration aus Sicht eines Verfahrens.
    time_aware: unveraendert. static: alle Zeitfaktoren 1.0, Pausen bleiben."""
    if view == "time_aware":
        return config
    if view == "static":
        static = copy.deepcopy(config)
        for window in static["time_model"]["day_windows"]:
            if window["factor"] is not None:
                window["factor"] = 1.0
        profiles = static["time_model"].get("resource_factors")
        if profiles:
            for values in (*profiles["manual_stations"], profiles["setup"]):
                values[:] = [None if value is None else 1.0 for value in values]
        return static
    raise ValueError(f"Unbekannte Dauer-Sicht '{view}', erlaubt: {DURATION_VIEWS}")


def setup_lookup(instance: dict) -> dict[tuple[str, str, str], int]:
    """(Ressource, Vorgaenger, Nachfolger) -> Basis-Ruestzeit."""
    return {(s["resource"], s["from_op"], s["to_op"]): s["base_minutes"]
            for s in instance["setups"]}


def plan_operation(config: dict, instance: dict, kind: str, base: int,
                   setup_minutes: int, start: int, resource: str | None = None) -> tuple[int, int]:
    """(Ruestdauer, Bearbeitungsdauer), wenn die Ruestung bei `start` beginnt
    und die Bearbeitung direkt folgt. ValueError bei einem Regelverstoss."""
    setup_len = duration(config, instance, "setup", setup_minutes, start) if setup_minutes else 0
    return setup_len, duration(config, instance, kind, base, start + setup_len, resource)


def first_feasible_start(config: dict, instance: dict, kind: str, base: int, setup_minutes: int,
                         earliest: int, resource: str | None = None) -> tuple[int, int, int] | None:
    """Fruehester Zeitpunkt t >= earliest, zu dem Ruestung und direkt folgende
    Bearbeitung regelkonform sind. Liefert (t, Ruestdauer, Bearbeitungsdauer)
    oder None, wenn bis zum Horizontende nichts mehr passt."""
    for t in range(earliest, horizon(config, instance)):
        try:
            return (t, *plan_operation(config, instance, kind, base, setup_minutes, t, resource))
        except ValueError:
            continue
    return None


def make_entry(op_id: str, resource: str, setup: tuple[str | None, int] | None,
               t: int, setup_len: int, process_len: int) -> dict:
    """Schedule-Eintrag; setup = None (Ressource ohne Ruestung) oder
    (Vorgaenger, Basis-Ruestzeit)."""
    entry = {"op": op_id, "resource": resource}
    if setup is not None:
        entry.update(setup_from=setup[0], setup_start=t, setup_end=t + setup_len)
    entry.update(start=t + setup_len, end=t + setup_len + process_len)
    return entry


# --------------------------------------------------------------------------
# Ereignisgesteuerte Planung
# --------------------------------------------------------------------------

class SimulationState:
    """Zustand einer ereignisgesteuerten Planung (non-delay).

    Ablauf fuer Baselines und RL-Umgebung:
        state = SimulationState(config, instance, view)
        while not state.done():
            candidates = state.candidate_operations()     # jetzt startbare Operationen
            if not candidates:
                state.advance()                            # zum naechsten Ereignis
                continue
            op = ...                                       # Regel oder Agent
            entry = ...  aus state.options(op["id"])       # Regel oder Agent
            state.apply(entry)

    Alle Dauern und Zulaessigkeiten gelten in der Sicht `view`
    ("time_aware" oder "static")."""

    def __init__(self, config: dict, instance: dict, view: str = "time_aware",
                 overflow_days: int | None = None):
        self.view = view
        self.config = view_config(config, view)
        self.instance = execution_instance(config, instance, overflow_days)
        self.ops = {o["id"]: o for o in instance["operations"]}
        self.jobs = {j["id"]: j for j in instance["jobs"]}
        self.job_index = {j["id"]: i for i, j in enumerate(instance["jobs"])}
        self.resources = all_resources(instance)
        self.resource_index = {r: i for i, r in enumerate(self.resources)}
        self.with_setup = set(setup_resources(config, instance))
        self.setup_base = setup_lookup(instance)
        self.initial_setup = config["rules"]["initial_setup_minutes"]

        self.time = 0
        self.resource_free = {r: 0 for r in self.resources}  # Ende des letzten Auftrags
        self.last_on = {r: None for r in self.resources}     # letzte Operation (Ruestung)
        self.next_index = {j: 0 for j in self.jobs}          # naechste offene Operation
        self.job_ready = {j: job["release"] for j, job in self.jobs.items()}
        self.schedule: list[dict] = []
        self._options: dict[str, list[dict]] = {}            # Cache fuer self.time

        day = self.config["time_model"]["day_minutes"]
        self._boundaries = sorted({d * day + w["start"]
                                   for d in range(self.instance["days"])
                                   for w in self.config["time_model"]["day_windows"]})

    # --- Abfragen ---------------------------------------------------------

    def done(self) -> bool:
        """True, wenn alle Operationen eingeplant sind."""
        return all(self.next_index[j] == len(job["operations"]) for j, job in self.jobs.items())

    def next_operation(self, job_id: str) -> dict | None:
        """Naechste offene Operation eines Jobs oder None."""
        ops = self.jobs[job_id]["operations"]
        index = self.next_index[job_id]
        return self.ops[ops[index]] if index < len(ops) else None

    def setup_of(self, resource: str, op_id: str) -> tuple[str | None, int] | None:
        """Ruestung vor op_id auf resource: None (keine Ruest-Ressource) oder
        (Vorgaenger, Basis-Ruestzeit)."""
        if resource not in self.with_setup:
            return None
        previous = self.last_on[resource]
        if previous is None:
            return (None, self.initial_setup)
        return (previous, self.setup_base[(resource, previous, op_id)])

    def options(self, op_id: str) -> list[dict]:
        """Alle Schedule-Eintraege, mit denen op_id JETZT (self.time) starten kann:
        Job bereit, Ressource frei und Ruestung + Bearbeitung regelkonform.
        Reihenfolge der Ressourcen wie in der Instanz (erst H, dann M)."""
        if op_id not in self._options:
            self._options[op_id] = self._compute_options(op_id)
        return self._options[op_id]

    def _compute_options(self, op_id: str) -> list[dict]:
        op = self.ops[op_id]
        if self.job_ready[op["job"]] > self.time:
            return []
        result = []
        for resource in self.resources:
            if resource not in op["base_processing"] or self.resource_free[resource] > self.time:
                continue
            setup = self.setup_of(resource, op_id)
            try:
                setup_len, process_len = plan_operation(
                    self.config, self.instance, op["kind"], op["base_processing"][resource],
                    setup[1] if setup else 0, self.time, resource)
            except ValueError:
                continue  # wuerde gegen eine Regel verstossen (Pause, Arbeitsende, ...)
            result.append(make_entry(op_id, resource, setup, self.time, setup_len, process_len))
        return result

    def candidate_operations(self) -> list[dict]:
        """Naechste Operationen aller Jobs, die jetzt auf mindestens einer
        Ressource starten koennen (Grundlage fuer Regeln und Action Masking)."""
        candidates = []
        for job_id in self.jobs:
            op = self.next_operation(job_id)
            if op is not None and self.options(op["id"]):
                candidates.append(op)
        return candidates

    # --- Aenderungen ------------------------------------------------------

    def apply(self, entry: dict) -> None:
        """Eintrag (aus options) einplanen und Zustand fortschreiben."""
        op = self.ops[entry["op"]]
        if entry not in self.options(op["id"]):
            raise ValueError(f"{entry['op']} kann jetzt nicht so eingeplant werden")
        self.schedule.append(entry)
        self.resource_free[entry["resource"]] = entry["end"]
        self.last_on[entry["resource"]] = entry["op"]
        self.job_ready[op["job"]] = entry["end"]
        self.next_index[op["job"]] += 1
        self._options.clear()

    def advance(self) -> bool:
        """Uhr auf das naechste Ereignis setzen: eine Ressource wird frei, ein
        Job wird bereit oder ein Zeitfenster beginnt (z. B. Pausenende,
        naechster Tag). False, wenn es kein weiteres Ereignis gibt."""
        events = [t for t in self.resource_free.values() if t > self.time]
        events += [self.job_ready[j] for j in self.jobs
                   if self.next_operation(j) is not None and self.job_ready[j] > self.time]
        events += [b for b in self._boundaries if b > self.time]
        if not events:
            return False
        self.time = min(events)
        self._options.clear()
        return True


# --------------------------------------------------------------------------
# Plan ausfuehren
# --------------------------------------------------------------------------

def plan_from_schedule(schedule: list[dict]) -> dict:
    """Zuordnung und Reihenfolge je Ressource aus einem Schedule ablesen
    (Eingabe fuer execute_plan)."""
    ordered = sorted(schedule, key=lambda e: (e.get("setup_start", e["start"]), e["start"]))
    sequence: dict[str, list[str]] = {}
    for e in ordered:
        sequence.setdefault(e["resource"], []).append(e["op"])
    return {"assignment": {e["op"]: e["resource"] for e in schedule}, "sequence": sequence}


def execute_plan(config: dict, instance: dict, plan: dict,
                 overflow_days: int | None = None) -> list[dict]:
    """Fuehrt die Entscheidungen eines Plans mit den echten Dauern aus.

    plan = {"assignment": {op_id: resource}, "sequence": {resource: [op_id, ...]}}

    Uebernommen werden nur Zuordnung und Reihenfolge je Ressource. Jede
    Operation startet zum fruehesten zulaessigen Zeitpunkt nach dem Ende
    ihres Vorgaengers im Job und des vorherigen Auftrags auf ihrer Ressource
    (Ruestung zuerst, Bearbeitung direkt danach). SimulationError, wenn der
    Plan auch mit Zusatztag nicht ausfuehrbar ist."""
    exec_instance = execution_instance(config, instance, overflow_days)
    ops = {o["id"]: o for o in instance["operations"]}
    assignment, sequence = plan["assignment"], plan["sequence"]
    _check_plan(ops, assignment, sequence)

    with_setup = set(setup_resources(config, instance))
    setup_base = setup_lookup(instance)
    initial_setup = config["rules"]["initial_setup_minutes"]
    position = {r: 0 for r in sequence}
    resource_free = {r: 0 for r in sequence}
    last_on: dict[str, str | None] = {r: None for r in sequence}
    next_index = {job["id"]: 0 for job in instance["jobs"]}
    job_ready = {job["id"]: job["release"] for job in instance["jobs"]}
    remaining = len(ops)
    schedule = []

    while remaining:
        progressed = False
        for job in instance["jobs"]:
            if next_index[job["id"]] == len(job["operations"]):
                continue
            op = ops[job["operations"][next_index[job["id"]]]]
            resource = assignment[op["id"]]
            if sequence[resource][position[resource]] != op["id"]:
                continue  # auf der Ressource ist vorher noch eine andere Operation dran

            if resource not in with_setup:
                setup = None
            elif last_on[resource] is None:
                setup = (None, initial_setup)
            else:
                setup = (last_on[resource], setup_base[(resource, last_on[resource], op["id"])])
            found = first_feasible_start(config, exec_instance, op["kind"],
                                         op["base_processing"][resource],
                                         setup[1] if setup else 0,
                                         max(resource_free[resource], job_ready[job["id"]]), resource)
            if found is None:
                raise SimulationError(f"{op['id']} passt auch mit den Zusatztagen "
                                      f"nicht mehr in den Horizont")
            entry = make_entry(op["id"], resource, setup, *found)
            schedule.append(entry)
            resource_free[resource] = entry["end"]
            last_on[resource] = op["id"]
            position[resource] += 1
            job_ready[job["id"]] = entry["end"]
            next_index[job["id"]] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            raise SimulationError("Plan ist widerspruechlich: Reihenfolgen auf den "
                                  "Ressourcen blockieren die Reihenfolge der Jobs")

    return sorted(schedule, key=lambda e: (e["start"], e["resource"]))


def _check_plan(ops: dict, assignment: dict, sequence: dict) -> None:
    """Plan vollstaendig und zur Instanz passend? Sonst ValueError."""
    if set(assignment) != set(ops):
        raise ValueError("Plan muss jeder Operation genau eine Ressource zuordnen")
    for op_id, resource in assignment.items():
        if resource not in ops[op_id]["base_processing"]:
            raise ValueError(f"{op_id} darf nicht auf {resource} laufen")
    planned = sorted(op_id for ops_on in sequence.values() for op_id in ops_on)
    if planned != sorted(ops) or any(assignment[o] != r for r, ops_on in sequence.items()
                                     for o in ops_on):
        raise ValueError("Reihenfolgen je Ressource passen nicht zur Zuordnung")
