"""Hierarchisches Ziel der Umgebung (Abschnitt reward in problem_config.json).

Kosten eines fertigen Plans: Summe aus Ebene * Gewicht. Ohne feste Gewichte
(level_weights = null) sind die Gewichte so gewaehlt, dass die Ordnung exakt
lexikografisch ist: Eine Einheit einer Ebene wiegt mehr als jeder moegliche Wert
aller tieferen Ebenen. Standard: erst Summe der Verspaetungen, dann Makespan.

Stepwise-Reward: Aenderung der geschaetzten Kosten. Die Schaetzung nutzt
estimate_completion; am Episodenende gilt die echte Fertigstellung. In der Summe bleibt
also (Startschaetzung - echte Kosten), Schaetzfehler verfaelschen das Ziel nicht.
"""
from __future__ import annotations

import torch

from rl.time_model import KIND_IDS, SETUP, TimeModel


def estimate_completion(td: dict[str, torch.Tensor], time_model: TimeModel) -> torch.Tensor:
    """Geschaetzte Fertigstellung je Job (B, J), mit Zeitfaktoren, Pause und Arbeitsende.

    Fertige Jobs: echte Fertigstellung. Naechste Operation: beste Ressource mit Belegung
    und Ruestung vom aktuellen Vorgaenger. Weitere Operationen: kleinste Basisdauer und
    kleinste Ruestung, bei Profilen je Handarbeitsplatz mit dem Bezugsprofil, denn der Platz steht
    noch nicht fest. Andere Jobs, die spaeter um Ressourcen konkurrieren, zaehlen nicht."""
    time = td["time"].unsqueeze(1)
    ready = torch.maximum(td["job_ready"], time)
    next_op, complete = td["next_op"], td["job_complete"]

    base = td["opt_base"]
    kinds = td["op_kind"].gather(1, next_op).unsqueeze(2).expand_as(base)
    earliest = torch.maximum(ready.unsqueeze(2), td["busy_until"].unsqueeze(1))
    after_setup = time_model.finish_after(SETUP, td["opt_setup_base"], earliest)
    resource = torch.arange(base.shape[2], device=base.device).view(1, 1, -1).expand_as(base)
    finish = time_model.finish_after(kinds, base, after_setup, resource)
    finish = finish.masked_fill(base == 0, torch.iinfo(torch.long).max)
    estimate = torch.where(complete, td["job_ready"], finish.amin(2))

    last_op = td["end_op_per_job"]
    longest = int((last_op - td["start_op_per_job"]).max().item())
    num_ops = td["op_kind"].shape[1]
    for offset in range(1, longest + 1):
        op = next_op + offset
        pending = ~complete & (op <= last_op)
        if not pending.any():
            break
        op = op.clamp(max=num_ops - 1)
        after_setup = time_model.finish_after(SETUP, td["min_setup_times"].gather(1, op), estimate)
        following = time_model.finish_after(td["op_kind"].gather(1, op),
                                            td["min_proc_times"].gather(1, op), after_setup)
        estimate = torch.where(pending, following, estimate)
    return estimate


class HierarchicalCost:
    """Kosten und Reward aus den Einstellungen in der Konfiguration."""

    def __init__(self, config: dict):
        reward = config["reward"]
        self.levels = tuple(reward["levels"])
        self.level_weights = reward["level_weights"]
        self.scale = reward["scale_minutes"]
        self.penalty = config["environment"]["truncation_penalty_minutes"]
        self.normalization = reward.get("normalization", "legacy")
        self.config = config

    def _manual_regret(self, td: dict[str, torch.Tensor]) -> torch.Tensor:
        """Realized manual minutes above the assigned station's best feasible window.

        This is processing efficiency, not a reward for waiting. Waiting can still
        increase tardiness, makespan and automatic completion times.
        """
        resource = td["op_resource"].clamp(min=0)
        base = td["proc_times"].gather(1, resource.unsqueeze(1)).squeeze(1)
        profiles = td["time_profile_factors"][:, :-1]
        station = resource.clamp(max=profiles.shape[1] - 1)
        factors = profiles.gather(1, station.unsqueeze(-1).expand(-1, -1, profiles.shape[-1]))
        durations = torch.ceil(base.unsqueeze(-1).to(torch.float64) * factors - 1e-9).to(torch.long)
        if not self.config["rules"]["time_dependent"]["manual"]:
            durations = base.unsqueeze(-1).expand_as(durations)
        windows = self.config["time_model"]["day_windows"]
        day = self.config["time_model"]["day_minutes"]
        rules = self.config["rules"]
        room = []
        for w in windows:
            ends = [day] if not rules["allowed_overnight"]["manual"] else [10 ** 12]
            if not rules["allowed_in_break"]["manual"]:
                ends += [b["start"] + offset for offset in (0, day) for b in windows
                         if b["factor"] is None and b["start"] + offset >= w["start"]]
            room.append(min(ends) - w["start"])
        room = torch.tensor(room, device=base.device)
        valid = (factors > 0) & (durations <= room)
        lower = durations.masked_fill(~valid, torch.iinfo(torch.long).max).amin(-1)
        actual = td["op_end"] - td["op_start"]
        selected = td["op_scheduled"] & (td["op_kind"] == KIND_IDS["manual"])
        return torch.where(selected, (actual - lower).clamp(min=0), 0).sum(1)

    def level_values(self, td: dict[str, torch.Tensor],
                     completion: torch.Tensor) -> dict[str, torch.Tensor]:
        """Alle Zielgroessen je Instanz. Reicht der Horizont nicht (truncated), kostet jede
        offene Operation zusaetzliche Verspaetung; der Makespan bleibt auf den Horizont begrenzt."""
        tardiness = (completion - td["due"]).clamp(min=0)
        open_ops = (~td["op_scheduled"]).sum(1)
        penalty = td["truncated"].to(torch.long) * open_ops * self.penalty
        values = {"total_tardiness": tardiness.sum(1) + penalty,
                "late_jobs": (tardiness > 0).sum(1),
                "max_tardiness": tardiness.amax(1) + penalty,
                "makespan": torch.minimum(completion.amax(1), td["horizon"])}
        if "manual_profile_regret" in self.levels:
            values["manual_profile_regret"] = self._manual_regret(td)
        if "machine_flow_time" in self.levels:
            automatic = td["op_scheduled"] & (td["op_kind"] == KIND_IDS["automatic"])
            values["machine_flow_time"] = torch.where(automatic, td["op_end"], 0).sum(1)
        return values

    def _upper_bounds(self, td: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Groesste moegliche Werte fertiger Plaene, fuer die exakte Hierarchie."""
        jobs, ops = td["due"].shape[1], td["op_kind"].shape[1]
        limit, extra = td["horizon"], ops * self.penalty
        return {"total_tardiness": jobs * limit + extra, "late_jobs": torch.full_like(limit, jobs),
                "max_tardiness": limit + extra, "makespan": limit,
                "manual_profile_regret": ops * limit, "machine_flow_time": ops * limit}

    def weights(self, td: dict[str, torch.Tensor]) -> torch.Tensor:
        """Gewicht je Instanz und Ebene (B, Ebenen) als float64."""
        batch = td["due"].shape[0]
        if self.level_weights is not None:
            fixed = torch.tensor(self.level_weights, dtype=torch.float64, device=td["due"].device)
            return fixed.expand(batch, -1)
        bounds = self._upper_bounds(td)
        weights = [torch.ones(batch, dtype=torch.float64, device=td["due"].device)]
        for level in reversed(self.levels[1:]):
            weights.insert(0, weights[0] * (bounds[level].to(torch.float64) + 1))
        return torch.stack(weights, dim=1)

    def cost(self, td: dict[str, torch.Tensor], completion: torch.Tensor) -> torch.Tensor:
        """Kosten je Instanz (B) als float64: kleiner ist besser."""
        values = self.level_values(td, completion)
        stacked = torch.stack([values[level].to(torch.float64) for level in self.levels], dim=1)
        if self.normalization == "per_entity":
            ones = torch.ones_like(td["horizon"])
            counts = {"total_tardiness": ones * td["due"].shape[1],
                      "late_jobs": ones * td["due"].shape[1], "max_tardiness": ones, "makespan": ones,
                      "manual_profile_regret": (td["op_kind"] == KIND_IDS["manual"]).sum(1).clamp(min=1),
                      "machine_flow_time": (td["op_kind"] == KIND_IDS["automatic"]).sum(1).clamp(min=1)}
            stacked = stacked / torch.stack([counts[level] for level in self.levels], dim=1)
        return (stacked * self.weights(td)).sum(1)

    def normalizer(self, td: dict[str, torch.Tensor]) -> torch.Tensor:
        """Teiler fuer den Reward, damit Instanzgroessen vergleichbar bleiben."""
        jobs = td["due"].shape[1] if self.normalization == "legacy" else 1
        return self.weights(td)[:, 0] * jobs * self.scale
