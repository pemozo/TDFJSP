"""RL-Umgebung fuer das zeitabhaengige FJSP (Aufbau angelehnt an RL4CO, nur Torch noetig).

Alle Einstellungen stehen in problem_config.json. Der Zustand ist ein dict aus Tensoren
(Formen siehe rl/parser.py); die festen Instanzdaten werden je Schritt nur referenziert, nicht
kopiert. Die Logik entspricht simulator.SimulationState, aber fuer ganze Batches.

Ablauf:
    env = TDFJSPEnv()
    td = env.reset(batch_size=4)                    # neue Instanzen on the fly
    while not td["done"].all():
        td["action"] = policy(td)                   # (B), Auswahl aus td["action_mask"]
        td = env.step(td)["next"]
    env.get_objectives(td), env.get_reward(td)

Aktionen (B): 0 = warten bis zum naechsten Ereignis (nur mit environment.allow_wait; bei
fertigen Instanzen immer 0), sonst 1 + Job * R + Ressource: Die naechste Operation des Jobs
startet jetzt auf der Ressource. Die Uhr springt selbst weiter, wenn nichts starten kann.
Die Episode endet, wenn alles eingeplant ist oder der Horizont nicht reicht (truncated).
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from generator import plot_schedule
from rl.generators import TDFJSPFileGenerator, TDFJSPGenerator
from rl.parser import instances_to_td, td_to_schedule
from rl.reward import HierarchicalCost, estimate_completion
from rl.time_model import KIND_IDS, SETUP, TIME_FEATURES, TimeModel
from schedule_validator import DEFAULT_CONFIG, check_config, check_schedule, load_config
from simulator import execution_instance, view_config

OPERATION_FEATURES = ("mean_base_time", "duration_if_started_now", "is_manual", "is_next",
                      "is_ready", "is_scheduled", "eligible_share", "position_in_job",
                      "job_slack", "job_remaining_work")
RESOURCE_FEATURES = ("busy_for", "is_manual_station", "open_operation_share", "has_predecessor")
PAIR_FEATURES = ("setup_if_started_now", "processing_if_started_now", "total_if_started_now",
                 "eligible", "feasible_now")
FEATURE_NAMES = {"time_features": TIME_FEATURES, "op_features": OPERATION_FEATURES,
                 "resource_features": RESOURCE_FEATURES, "pair_features": PAIR_FEATURES}

# Feste Instanzdaten: werden von Schritt zu Schritt nur weitergereicht
STATIC_KEYS = ("proc_times", "setup_times", "setup_resources", "is_station", "op_kind",
               "ops_job_map", "ops_sequence_order", "start_op_per_job", "end_op_per_job",
               "release", "due", "horizon", "seed", "num_eligible", "mean_proc_times",
               "min_proc_times", "min_setup_times")
# Zustand, der sich aendert: wird je Schritt kopiert, fruehere Zustaende bleiben also gueltig
_CHANGING_KEYS = ("time", "next_op", "job_ready", "job_complete", "busy_until", "last_op",
                  "op_scheduled", "op_resource", "op_predecessor", "op_decision", "op_start",
                  "op_end", "op_order", "num_scheduled", "done", "truncated")
_NEVER = torch.iinfo(torch.long).max


def resolve_device(setting: str) -> torch.device:
    """environment.device: auto nimmt die GPU, wenn es eine gibt."""
    if setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(setting)


class InstanceDataset(Dataset):
    """Feste Instanzdaten als Dataset; ein torch-DataLoader stapelt sie zu Batches."""

    def __init__(self, td: dict[str, torch.Tensor]):
        self.td = td

    def __len__(self) -> int:
        return self.td["due"].shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.td.items()}


class TDFJSPEnv:
    """Umgebung; config = None laedt die Datei config_path."""

    name = "tdfjsp"

    def __init__(self, config: dict | None = None, generator: TDFJSPGenerator | None = None,
                 config_path: Path | str = DEFAULT_CONFIG):
        if config is None:
            config = load_config(config_path)
        errors = check_config(config)
        if errors:
            raise ValueError("Ungueltige Konfiguration: " + "; ".join(errors))
        self.config = config
        self.config_path = config_path
        settings = config["environment"]
        self.device = resolve_device(settings["device"])
        self.view = settings["view"]
        self.allow_wait = settings["allow_wait"]
        self.batch_size = settings["batch_size"]
        self.compute_features = settings["compute_features"]
        self.time_scale = settings["time_scale_minutes"]
        self.reward_mode = config["reward"]["mode"]
        self.initial_setup = config["rules"]["initial_setup_minutes"]
        self.time_model = TimeModel(config, self.view, self.device)
        self.cost = HierarchicalCost(config)
        self.generator = TDFJSPGenerator(config, config_path=config_path) if generator is None else generator

    # --- Hauptfunktionen ------------------------------------------------------

    def reset(self, td: dict[str, torch.Tensor] | None = None,
              batch_size: int | None = None) -> dict[str, torch.Tensor]:
        """Startzustand. td = feste Instanzdaten (z. B. aus einem DataLoader oder ein alter
        Zustand); ohne td erzeugt der Generator neue Instanzen."""
        if td is None:
            td = self.generator(self.batch_size if batch_size is None else batch_size)
        kept = [key for key in (*STATIC_KEYS, "instance_index") if key in td]
        return self._reset({key: td[key].to(self.device) for key in kept})

    def step(self, td: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
        """Fuehrt td["action"] aus. Der Eingabezustand bleibt unveraendert."""
        return {"next": self._step(td)}

    def get_action_mask(self, td: dict[str, torch.Tensor]) -> torch.Tensor:
        """Erlaubte Aktionen (B, 1 + J * R). Position 0 = warten."""
        finished = td["done"]
        wait = finished
        if self.allow_wait:
            wait = finished | self._next_event(td)[1]
        return torch.cat((wait.unsqueeze(1), td["pair_mask"].flatten(1)), dim=1)

    def get_reward(self, td: dict[str, torch.Tensor]) -> torch.Tensor:
        """Endreward fertiger Episoden (B) als float64: -Kosten / Teiler."""
        if not td["done"].all():
            raise ValueError("get_reward braucht fertige Episoden; je Schritt steht der "
                             "Reward in td[\"reward\"]")
        return -self.cost.cost(td, self._completion(td)) / self.cost.normalizer(td)

    def get_objectives(self, td: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Zielgroessen je Instanz. Fertige Episoden: echte Werte; laufende: Schaetzung."""
        return {**self.cost.level_values(td, self._completion(td)), "truncated": td["truncated"]}

    def wait_target(self, td: dict[str, torch.Tensor]) -> torch.Tensor:
        """Zeitpunkt, bis zu dem WAIT die Uhr stellt (B); nur gueltig, wo WAIT erlaubt ist."""
        return self._next_event(td)[0]

    def to_schedule(self, td: dict[str, torch.Tensor], index: int, instance: dict) -> list[dict]:
        """Plan einer Batch-Position im Schedule-Format des Projekts."""
        return td_to_schedule(self.config, td, index, instance)

    def check_solution_validity(self, td: dict[str, torch.Tensor], instances: list[dict]) -> None:
        """Prueft fertige Plaene mit dem unabhaengigen Validator, in der Sicht der Umgebung.
        ValueError beim ersten Regelverstoss; abgebrochene Episoden werden uebersprungen."""
        rules = view_config(self.config, self.view)
        for index, instance in enumerate(instances):
            if td["truncated"][index]:
                continue
            violations = check_schedule(rules, execution_instance(self.config, instance),
                                        self.to_schedule(td, index, instance))
            if violations:
                raise ValueError(f"Batch-Position {index}: {violations[0]}")

    def render(self, td: dict[str, torch.Tensor], index: int, instance: dict,
               path: Path | None = None):
        """Plan als Gantt-Diagramm (matplotlib); gibt die Figure zurueck."""
        return plot_schedule(self.config, execution_instance(self.config, instance),
                             self.to_schedule(td, index, instance),
                             f"Umgebung ({self.view})", path)

    def dataset(self, batch_size: int | None = None, phase: str = "train",
                class_name: str | None = None) -> InstanceDataset:
        """Datensatz einer Phase: train neu erzeugt, val und test aus den festen Dateien."""
        if phase == "train":
            count = self.batch_size if batch_size is None else batch_size
            return InstanceDataset(self.generator(count, class_name))
        if class_name is None:
            raise ValueError(f"Phase {phase} braucht class_name")
        files = TDFJSPFileGenerator(self.config, class_name, phase, self.config_path)
        return InstanceDataset(files(batch_size))

    def load_instances(self, instances: list[dict]) -> dict[str, torch.Tensor]:
        """Instanz-Dicts zu einem Startzustand."""
        return self.reset(instances_to_td(self.config, instances, self.device))

    # --- Zustand --------------------------------------------------------------

    def _reset(self, td: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch, jobs = td["due"].shape
        resources, ops = td["proc_times"].shape[1:]
        device = td["due"].device
        largest = max(int(td["proc_times"].max().item()), int(td["setup_times"].max().item())
                      if td["setup_times"].numel() else 0, self.initial_setup)
        self.time_model.ensure_finish_table(largest)

        def filled(*shape, value=-1):
            return torch.full(shape, value, dtype=torch.long, device=device)

        def flags(*shape):
            return torch.zeros(shape, dtype=torch.bool, device=device)

        td.update(time=filled(batch, value=0), next_op=td["start_op_per_job"].clone(),
                  job_ready=td["release"].clone(), job_complete=flags(batch, jobs),
                  busy_until=filled(batch, resources, value=0), last_op=filled(batch, resources),
                  op_scheduled=flags(batch, ops), op_resource=filled(batch, ops),
                  op_predecessor=filled(batch, ops), op_decision=filled(batch, ops),
                  op_start=filled(batch, ops), op_end=filled(batch, ops),
                  op_order=filled(batch, ops), num_scheduled=filled(batch, value=0),
                  done=flags(batch), truncated=flags(batch),
                  reward=torch.zeros(batch, dtype=torch.float64, device=device))
        self._advance_until_decision(td)
        self._refresh(td, active=torch.zeros(batch, dtype=torch.bool, device=device))
        return td

    def _step(self, td: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        action = td["action"]
        if not td["action_mask"].gather(1, action.unsqueeze(1)).all():
            raise ValueError("Unzulaessige Aktion: Sie ist laut td[\"action_mask\"] gesperrt")
        state = {key: value for key, value in td.items() if key != "action"}
        for key in _CHANGING_KEYS:
            state[key] = td[key].clone()

        active = ~state["done"]
        waiting = active & (action == 0)
        if waiting.any():
            state["time"] = torch.where(waiting, self._next_event(state)[0], state["time"])
        planned = (active & (action > 0)).nonzero().squeeze(1)
        if planned.numel():
            self._make_step(state, planned, action[planned] - 1)

        state["done"] = state["done"] | state["job_complete"].all(1)
        self._advance_until_decision(state)
        self._refresh(state, active)
        return state

    def _make_step(self, td: dict[str, torch.Tensor], index: torch.Tensor,
                   action: torch.Tensor) -> None:
        """Plant fuer die Batch-Positionen `index` die gewaehlten Paare ein."""
        resources = td["proc_times"].shape[1]
        job, resource = action // resources, action % resources
        op = td["next_op"][index, job]
        now = td["time"][index]
        start = now + td["opt_setup"][index, job, resource]
        end = start + td["opt_proc"][index, job, resource]

        td["op_scheduled"][index, op] = True
        td["op_resource"][index, op] = resource
        td["op_predecessor"][index, op] = td["last_op"][index, resource]
        td["op_decision"][index, op] = now
        td["op_start"][index, op] = start
        td["op_end"][index, op] = end
        td["op_order"][index, op] = td["num_scheduled"][index]
        td["num_scheduled"][index] += 1

        td["busy_until"][index, resource] = end
        td["last_op"][index, resource] = op
        td["job_ready"][index, job] = end
        last = td["end_op_per_job"][index, job]
        td["job_complete"][index, job] = op == last
        td["next_op"][index, job] = torch.minimum(op + 1, last)

    def _advance_until_decision(self, td: dict[str, torch.Tensor]) -> None:
        """Uhr weiterstellen, bis in jeder laufenden Instanz ein Paar starten kann."""
        while True:
            self._update_options(td)
            stuck = ~td["done"] & ~td["pair_mask"].any(dim=(1, 2))
            if not stuck.any():
                return
            event, exists = self._next_event(td)
            td["time"] = torch.where(stuck & exists, event, td["time"])
            ended = stuck & ~exists
            td["truncated"] = td["truncated"] | ended
            td["done"] = td["done"] | ended

    def _next_event(self, td: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Naechstes Ereignis wie SimulationState.advance: Ressource frei, Job bereit,
        Zeitfenster beginnt. Rueckgabe: Zeitpunkt und ob es eines gibt."""
        time = td["time"]
        busy = td["busy_until"].masked_fill(td["busy_until"] <= time.unsqueeze(1), _NEVER).amin(1)
        waiting = (td["job_ready"] <= time.unsqueeze(1)) | td["job_complete"]
        ready = td["job_ready"].masked_fill(waiting, _NEVER).amin(1)
        boundary, in_horizon = self.time_model.next_boundary(time, td["horizon"])
        event = torch.minimum(torch.minimum(busy, ready), boundary.masked_fill(~in_horizon, _NEVER))
        return event, event < _NEVER

    def _update_options(self, td: dict[str, torch.Tensor]) -> None:
        """Dauern und Zulaessigkeit aller Paare aus naechster Operation und Ressource, wenn sie
        jetzt starten wuerden (wie SimulationState.options). Formen (B, J, R)."""
        time = td["time"].unsqueeze(1)
        next_op = td["next_op"]
        resources = td["proc_times"].shape[1]
        base = td["proc_times"].gather(2, next_op.unsqueeze(1).expand(-1, resources, -1)).transpose(1, 2)
        setup_base = self._setup_base_now(td)
        now = time.unsqueeze(2).expand_as(base)
        limit = td["horizon"].view(-1, 1, 1)

        setup, setup_allowed = self.time_model.duration(SETUP, setup_base, now, limit)
        kinds = td["op_kind"].gather(1, next_op).unsqueeze(2).expand_as(base)
        processing, processing_allowed = self.time_model.duration(kinds, base, now + setup, limit)

        free = td["busy_until"].unsqueeze(1) <= time.unsqueeze(2)
        ready = (td["job_ready"] <= time) & ~td["job_complete"]
        running = ~td["done"].view(-1, 1, 1)
        td["pair_mask"] = ((base > 0) & free & ready.unsqueeze(2) & setup_allowed
                           & processing_allowed & running)
        td["opt_base"], td["opt_setup_base"] = base, setup_base
        td["opt_setup"], td["opt_proc"] = setup, processing

    def _setup_base_now(self, td: dict[str, torch.Tensor]) -> torch.Tensor:
        """Basis-Ruestzeit je Paar (B, J, R) vom aktuellen Vorgaenger auf der Ressource."""
        batch, jobs = td["due"].shape
        resources = td["proc_times"].shape[1]
        result = torch.zeros(batch, jobs, resources, dtype=torch.long, device=td["due"].device)
        with_setup = td["setup_resources"]
        count = with_setup.shape[1]
        if count == 0:
            return result
        batch_index = torch.arange(batch, device=result.device).unsqueeze(1)
        rows = torch.arange(count, device=result.device).unsqueeze(0)
        previous = td["last_op"].gather(1, with_setup)
        by_next_op = td["setup_times"][batch_index, rows, previous.clamp(min=0)]
        values = by_next_op.gather(2, td["next_op"].unsqueeze(1).expand(-1, count, -1)).to(torch.long)
        values = torch.where((previous < 0).unsqueeze(2), self.initial_setup, values)
        return result.scatter_(2, with_setup.unsqueeze(1).expand(-1, jobs, -1), values.transpose(1, 2))

    def _completion(self, td: dict[str, torch.Tensor]) -> torch.Tensor:
        return estimate_completion(td, self.time_model)

    def _refresh(self, td: dict[str, torch.Tensor], active: torch.Tensor) -> None:
        """Maske, Reward und Merkmale passend zum neuen Zustand setzen."""
        td["action_mask"] = self.get_action_mask(td)
        need_estimate = (self.compute_features or self.reward_mode == "stepwise"
                         or bool(td["done"].any()))
        completion = self._completion(td) if need_estimate else None

        reward = torch.zeros_like(td["reward"])
        if self.reward_mode == "stepwise":
            cost = self.cost.cost(td, completion)
            if "cost_estimate" in td:
                reward = (td["cost_estimate"] - cost) / self.cost.normalizer(td)
                reward = torch.where(active, reward, torch.zeros_like(reward))
            else:
                td["initial_cost_estimate"] = cost
            td["cost_estimate"] = cost
        elif td["done"].any():
            final = -self.cost.cost(td, completion) / self.cost.normalizer(td)
            reward = torch.where(active & td["done"], final, reward)
        td["reward"] = reward

        if self.compute_features:
            self._get_features(td, completion)

    # --- Merkmale -------------------------------------------------------------

    def _get_features(self, td: dict[str, torch.Tensor], completion: torch.Tensor) -> None:
        """Beobachtung fuer den Agenten als float32; Spalten siehe FEATURE_NAMES."""
        scale = self.time_scale
        time = td["time"]
        batch, ops = td["op_kind"].shape
        resources = td["proc_times"].shape[1]
        job_of_op = td["ops_job_map"]

        scheduled = td["op_scheduled"]
        is_next = torch.zeros_like(scheduled).scatter_(1, td["next_op"], ~td["job_complete"])
        job_ready = td["job_ready"].gather(1, job_of_op) <= time.unsqueeze(1)
        slack = (td["due"] - completion).to(torch.float32) / scale
        open_work = ((td["min_proc_times"] + td["min_setup_times"]).masked_fill(scheduled, 0)
                     .to(torch.float32))
        remaining = torch.zeros_like(slack).scatter_add_(1, job_of_op, open_work) / scale
        now_factor = self.time_model.factor_for_kind(td["op_kind"], time).to(torch.float32)
        last_position = (td["end_op_per_job"] - td["start_op_per_job"]).gather(1, job_of_op)

        td["op_features"] = torch.stack((
            td["mean_proc_times"] / scale,
            td["mean_proc_times"] * now_factor / scale,
            (td["op_kind"] == KIND_IDS["manual"]).to(torch.float32),
            is_next.to(torch.float32),
            (is_next & job_ready).to(torch.float32),
            scheduled.to(torch.float32),
            td["num_eligible"].to(torch.float32) / resources,
            td["ops_sequence_order"].to(torch.float32) / last_position.clamp(min=1).to(torch.float32),
            slack.gather(1, job_of_op),
            remaining.gather(1, job_of_op),
        ), dim=2)

        open_edges = (td["proc_times"] > 0) & ~scheduled.unsqueeze(1)
        td["ops_ma_adj"] = open_edges
        td["resource_features"] = torch.stack((
            (td["busy_until"] - time.unsqueeze(1)).clamp(min=0).to(torch.float32) / scale,
            td["is_station"].to(torch.float32),
            open_edges.sum(2).to(torch.float32) / ops,
            (td["last_op"] >= 0).to(torch.float32),
        ), dim=2)

        eligible = (td["opt_base"] > 0) & ~td["job_complete"].unsqueeze(2)
        setup = td["opt_setup"].masked_fill(~eligible, 0).to(torch.float32) / scale
        processing = td["opt_proc"].masked_fill(~eligible, 0).to(torch.float32) / scale
        td["pair_features"] = torch.stack((setup, processing, setup + processing,
                                           eligible.to(torch.float32),
                                           td["pair_mask"].to(torch.float32)), dim=3)
        td["time_features"] = self.time_model.time_features(time, td["horizon"], scale)
