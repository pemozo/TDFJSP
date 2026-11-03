"""Einfache Policies ohne neuronales Netz, zum Pruefen der Umgebung und als Vergleich.

Eine Policy bekommt den Zustand und gibt je Instanz eine Aktion zurueck. Ein spaeterer
RL-Agent nutzt dieselbe Schnittstelle und laeuft dann unveraendert durch rl/evaluate.py.

edd_ect und edd_spt entsprechen baselines.rules, einschliesslich der Gleichstand-Regeln.
"""
from __future__ import annotations

from typing import Callable

import torch

Policy = Callable[[dict[str, torch.Tensor]], torch.Tensor]

# groesser als jedes moegliche Ende (Minuten), trennt Sortierstufen in einem Schluessel
_MINUTE_RANGE = 10 ** 6
_NEVER = torch.iinfo(torch.long).max


def _dispatch(td: dict[str, torch.Tensor], use_spt: bool) -> torch.Tensor:
    """EDD waehlt den Job, ECT oder SPT die Ressource; Gleichstand: Reihenfolge."""
    pairs = td["pair_mask"]
    batch, jobs, resources = pairs.shape
    device = pairs.device

    job_order = torch.arange(jobs, device=device)
    job_key = (td["due"] * (jobs + 1) + job_order).masked_fill(~pairs.any(2), _NEVER)
    job = job_key.argmin(1)

    rows = torch.arange(batch, device=device)
    setup, processing, allowed = td["opt_setup"][rows, job], td["opt_proc"][rows, job], pairs[rows, job]
    end = setup + processing    # alle Optionen starten jetzt
    resource_order = torch.arange(resources, device=device)
    key = end * (resources + 1) + resource_order
    if use_spt:
        key = key + processing * _MINUTE_RANGE * (resources + 1)
    resource = key.masked_fill(~allowed, _NEVER).argmin(1)

    return torch.where(td["done"], 0, 1 + job * resources + resource)


def edd_ect(td: dict[str, torch.Tensor]) -> torch.Tensor:
    return _dispatch(td, use_spt=False)


def edd_spt(td: dict[str, torch.Tensor]) -> torch.Tensor:
    return _dispatch(td, use_spt=True)


def make_random_policy(seed: int, device: str | torch.device = "cpu") -> Policy:
    """Gleichverteilt ueber die erlaubten Aktionen, reproduzierbar ueber den Seed."""
    generator = torch.Generator(device=device).manual_seed(seed)

    def random_policy(td: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.multinomial(td["action_mask"].to(torch.float32), 1,
                                 generator=generator).squeeze(1)

    return random_policy


POLICY_NAMES = ("edd_ect", "edd_spt", "random")


def make_policy(name: str, seed: int = 0, device: str | torch.device = "cpu") -> Policy:
    """Policy zu einem Namen aus POLICY_NAMES."""
    if name == "random":
        return make_random_policy(seed, device)
    if name in ("edd_ect", "edd_spt"):
        return edd_ect if name == "edd_ect" else edd_spt
    raise ValueError(f"Unbekannte Policy '{name}', erlaubt: {POLICY_NAMES}")


def rollout(env, td: dict[str, torch.Tensor],
            policy: Policy) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Episode bis zum Ende spielen. Rueckgabe: Endzustand und Summe der Rewards (B)."""
    total = torch.zeros_like(td["reward"])
    while not td["done"].all():
        td["action"] = policy(td)
        td = env.step(td)["next"]
        total = total + td["reward"]
    return td, total
