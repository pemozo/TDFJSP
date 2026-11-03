"""Eingaben des neuronalen Netzes aus dem Zustand der Umgebung.

Formen: B = Batch, O = Operationen (einschliesslich Padding), J = Jobs, R = Ressourcen (erst
Handarbeitsplaetze, dann Maschinen; im Modell "Maschinenknoten"), K = Zeitfenster eines Tages.

static_inputs (je Instanz einmal, fuer den Encoder):
    proc_profiles     [B, O, R, K]  Dauer bei Start im Fenster k / Basisdauer; 0 = nicht zulaessig
                                    oder in diesem Fenster nicht startbar (Pause)
    proc_ref          [B, O, R]     Basisdauer / time_scale_minutes; damit bleibt die absolute Dauer
                                    rekonstruierbar: Dauer = proc_profiles * proc_ref * time_scale
    profile_lengths   [K]           Laenge jedes Fensters / Tageslaenge (Fenster sind ungleich lang)
    profile_starts    [K]           Beginn jedes Fensters in Minuten ab Tagesbeginn
    eligible          [B, O, R]     Operation darf auf der Ressource laufen
    precedence        [B, O, O]     Eintrag [i, j]: i ist direkter Vorgaenger von j
    pad_mask          [B, O]        Padding-Operation
    op_features       [B, O, F_op]  siehe OP_FEATURES
    machine_features  [B, R, F_ma]  siehe MACHINE_FEATURES

dynamic_inputs (je Schritt, fuer Decoder und Critic):
    next_op           [B, J]        naechste Operation je Job
    candidates        [B, J, R, F_c]  siehe CANDIDATE_FEATURES, je Paar aus Job und Ressource
    candidate_valid   [B, J, R]     Ressource darf die naechste Operation bearbeiten, Job offen
    time_features     [B, 7]        siehe rl.time_model.TIME_FEATURES
    state_features    [B, F_s]      siehe STATE_FEATURES
    action_mask       [B, 1 + J * R]

model.time_information = "none": Der Agent bekommt keine Zeitfaktoren und keine Uhrzeit. Alle
Dauern rechnet er mit Faktor 1.0 (Sicht "static": Pause und Arbeitsende sind bekannt), die
Zeitmerkmale sind 0 und die absolute Deadline fehlt in den Operationsmerkmalen. Die Umgebung selbst
bleibt unveraendert: Maske, Reward und Ablauf nutzen weiter die echten Dauern.
"""
from __future__ import annotations

import torch

from rl.env import TDFJSPEnv
from rl.reward import estimate_completion
from rl.time_model import KIND_IDS, SETUP, TimeModel

OP_FEATURES = ("mean_base_time", "min_base_time", "max_base_time", "is_manual", "eligible_share",
               "position_in_job", "remaining_share", "job_work", "due", "min_setup")
MACHINE_FEATURES = ("is_manual_station", "eligible_share", "mean_base_time", "has_setup", "mean_setup")
CANDIDATE_FEATURES = ("setup_start_offset", "setup_time", "processing_start_offset", "processing_time",
                      "finish_offset", "job_slack", "due_in", "feasible_now", "wait_gain")
STATE_FEATURES = ("progress", "free_resources", "mean_busy_for", "open_jobs", "min_slack",
                  "mean_slack", "estimated_late_jobs")
_FEASIBILITY_STEPS = 64     # mehr Fenstergrenzen liegen nie vor einem zulaessigen Start


def _plan_at(time_model: TimeModel, kinds: torch.Tensor, base: torch.Tensor, setup_base: torch.Tensor,
             time: torch.Tensor, limit: torch.Tensor):
    """Ruestdauer, Bearbeitungsdauer und Zulaessigkeit bei Ruestbeginn `time`, wie
    simulator.plan_operation: Die Bearbeitung beginnt direkt nach der Ruestung, mit dem dort
    geltenden Faktor."""
    setup, setup_ok = time_model.duration(SETUP, setup_base, time, limit)
    processing, processing_ok = time_model.duration(kinds, base, time + setup, limit)
    return setup, processing, setup_ok & processing_ok


def first_feasible(time_model: TimeModel, kinds: torch.Tensor, base: torch.Tensor,
                   setup_base: torch.Tensor, earliest: torch.Tensor, limit: torch.Tensor):
    """Fruehester zulaessiger Ruestbeginn >= earliest je Element, wie simulator.first_feasible_start.

    Solange Ruest- und Bearbeitungsbeginn im selben Fenster bleiben, sind beide Dauern fest. Wer in
    so einem Abschnitt nicht passt, passt dort auch spaeter nicht mehr. Deshalb reicht der Sprung
    zum Ende des Abschnitts statt einer Suche ueber jede Minute.
    Rueckgabe: Beginn, Ruestdauer, Bearbeitungsdauer, gefunden."""
    time = earliest.clone()
    setup, processing, allowed = _plan_at(time_model, kinds, base, setup_base, time, limit)
    for _ in range(_FEASIBILITY_STEPS):
        open_ = ~allowed & (time < limit)
        if not open_.any():
            break
        index = open_.nonzero(as_tuple=True)
        now, length = time[index], setup[index]
        window, _ = time_model.next_boundary(now, limit[index])
        setup_window, _ = time_model.next_boundary(now + length, limit[index])
        time[index] = torch.minimum(window, setup_window - length)
        new_setup, new_processing, new_allowed = _plan_at(
            time_model, kinds[index], base[index], setup_base[index], time[index], limit[index])
        setup[index], processing[index], allowed[index] = new_setup, new_processing, new_allowed
    return time, setup, processing, allowed


class ObservationBuilder:
    """Baut die Eingaben des Netzes; die Zeitinformation steuert model.time_information."""

    def __init__(self, env: TDFJSPEnv):
        self.env = env
        self.scale = float(env.time_scale)
        self.full_information = env.config["model"]["time_information"] == "full"
        self.time_model = (env.time_model if self.full_information
                           else TimeModel(env.config, "static", env.device))

    def static_inputs(self, td: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        largest = max(int(td["proc_times"].max().item()),
                      int(td["setup_times"].max().item()) if td["setup_times"].numel() else 0,
                      self.env.initial_setup)
        self.time_model.ensure_finish_table(largest)

        proc = td["proc_times"].transpose(1, 2)
        eligible = proc > 0
        kinds = td["op_kind"].unsqueeze(2).expand_as(proc)
        lengths = self.time_model.window_profile(kinds, proc).to(torch.float32)
        profiles = lengths / proc.clamp(min=1).unsqueeze(-1).to(torch.float32)

        job, order = td["ops_job_map"], td["ops_sequence_order"]
        same_job = job.unsqueeze(2) == job.unsqueeze(1)
        precedence = same_job & (order.unsqueeze(2) + 1 == order.unsqueeze(1))
        pad_mask = torch.zeros_like(job, dtype=torch.bool)
        return {"proc_profiles": profiles,
                "proc_ref": proc.to(torch.float32) / self.scale,
                "profile_lengths": self.time_model.window_lengths.to(torch.float32) / self.time_model.day,
                "profile_starts": self.time_model.window_starts,
                "eligible": eligible,
                "precedence": precedence,
                "pad_mask": pad_mask,
                "op_features": self._op_features(td, proc, eligible),
                "machine_features": self._machine_features(td, proc, eligible)}

    def _op_features(self, td, proc, eligible) -> torch.Tensor:
        scale = self.scale
        resources = proc.shape[2]
        job = td["ops_job_map"]
        last = (td["end_op_per_job"] - td["start_op_per_job"]).gather(1, job).clamp(min=1).to(torch.float32)
        position = td["ops_sequence_order"].to(torch.float32) / last
        largest = proc.amax(2).to(torch.float32)
        work = (td["min_proc_times"] + td["min_setup_times"]).to(torch.float32)
        job_work = torch.zeros_like(td["due"], dtype=torch.float32).scatter_add_(1, job, work)
        due = td["due"].to(torch.float32).gather(1, job) / self.time_model.day
        if not self.full_information:
            due = torch.zeros_like(due)     # zusammen mit dem Puffer ergaebe sie die Uhrzeit
        return torch.stack((td["mean_proc_times"] / scale,
                            td["min_proc_times"].to(torch.float32) / scale,
                            largest / scale,
                            (td["op_kind"] == KIND_IDS["manual"]).to(torch.float32),
                            eligible.sum(2).to(torch.float32) / resources,
                            position,
                            1 - position,
                            job_work.gather(1, job) / scale,
                            due,
                            td["min_setup_times"].to(torch.float32) / scale), dim=2)

    def _machine_features(self, td, proc, eligible) -> torch.Tensor:
        batch, ops, resources = proc.shape
        count = eligible.sum(1).to(torch.float32)
        mean_base = proc.sum(1).to(torch.float32) / count.clamp(min=1)
        has_setup = torch.zeros(batch, resources, dtype=torch.float32, device=proc.device)
        mean_setup = torch.zeros(batch, resources, dtype=torch.float32, device=proc.device)
        if td["setup_times"].numel():
            positive = td["setup_times"] > 0
            per_resource = (td["setup_times"].to(torch.float32).sum(dim=(2, 3))
                            / positive.sum(dim=(2, 3)).clamp(min=1).to(torch.float32))
            has_setup.scatter_(1, td["setup_resources"], 1.0)
            mean_setup.scatter_(1, td["setup_resources"], per_resource)
        return torch.stack((td["is_station"].to(torch.float32), count / ops,
                            mean_base / self.scale, has_setup, mean_setup / self.scale), dim=2)

    def dynamic_inputs(self, td: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        model, scale = self.time_model, self.scale
        now = td["time"]
        batch_now = now.view(-1, 1, 1)
        base, setup_base = td["opt_base"], td["opt_setup_base"]
        limit = td["horizon"].view(-1, 1, 1).expand_as(base)
        kinds = td["op_kind"].gather(1, td["next_op"]).unsqueeze(2).expand_as(base)
        valid = (base > 0) & ~td["job_complete"].unsqueeze(2)

        earliest = torch.maximum(torch.maximum(td["job_ready"].unsqueeze(2), td["busy_until"].unsqueeze(1)),
                                 batch_now)
        start, setup, processing, found = first_feasible(model, kinds, base, setup_base, earliest, limit)
        finish = start + setup + processing

        # Geschaetzte Restarbeit des Jobs nach dieser Operation, aus der Sicht des Agenten
        completion = estimate_completion(td, model)
        usable = valid & found
        best = finish.masked_fill(~usable, torch.iinfo(torch.long).max).amin(2)
        after = torch.where(usable.any(2), (completion - best).clamp(min=0), torch.zeros_like(best))
        due = td["due"].unsqueeze(2)

        # Gewinn durch Warten: Ende bei Start ab der naechsten Fenstergrenze minus Ende bei Start jetzt.
        # Negativ heisst: Ein spaeterer Start waere frueher fertig (der Faktor faellt).
        boundary, _ = model.next_boundary(now, td["horizon"])
        waited = torch.maximum(earliest, boundary.view(-1, 1, 1))
        wait_start, wait_setup, wait_processing, wait_found = first_feasible(
            model, kinds, base, setup_base, waited, limit)
        wait_finish = wait_start + wait_setup + wait_processing
        gain = torch.where(wait_found & found, wait_finish - finish, torch.zeros_like(finish))

        def scaled(values: torch.Tensor) -> torch.Tensor:
            return values.to(torch.float32).masked_fill(~usable, 0.0) / scale

        candidates = torch.stack((scaled(start - batch_now), scaled(setup),
                                  scaled(start + setup - batch_now), scaled(processing),
                                  scaled(finish - batch_now),
                                  scaled(due - finish - after.unsqueeze(2)),
                                  scaled(due - batch_now),
                                  td["pair_mask"].to(torch.float32),
                                  scaled(gain)), dim=3)

        time_features = model.time_features(now, td["horizon"], scale)
        if not self.full_information:
            time_features = torch.zeros_like(time_features)
        return {"next_op": td["next_op"], "candidates": candidates, "candidate_valid": valid,
                "time_features": time_features,
                "state_features": self._state_features(td, completion),
                "action_mask": td["action_mask"]}

    def _state_features(self, td, completion) -> torch.Tensor:
        scale = self.scale
        ops = td["op_kind"].shape[1]
        open_jobs = ~td["job_complete"]
        slack = (td["due"] - completion).to(torch.float32) / scale
        count = open_jobs.sum(1).clamp(min=1).to(torch.float32)
        busy = (td["busy_until"] - td["time"].unsqueeze(1)).clamp(min=0).to(torch.float32)
        smallest = slack.masked_fill(~open_jobs, float("inf")).amin(1)
        smallest = torch.where(open_jobs.any(1), smallest, torch.zeros_like(smallest))
        return torch.stack((td["num_scheduled"].to(torch.float32) / ops,
                            (busy == 0).to(torch.float32).mean(1),
                            busy.mean(1) / scale,
                            open_jobs.to(torch.float32).mean(1),
                            smallest,
                            (slack * open_jobs).sum(1) / count,
                            ((slack < 0) & open_jobs).sum(1).to(torch.float32) / open_jobs.shape[1]), dim=1)
