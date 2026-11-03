"""Decoder: bewertet je Job und Ressource die naechste Operation des Jobs, dazu die Aktion WAIT.

Eingaben: h_op [B, O, d], h_ma [B, R, d], next_op [B, J], action_mask [B, 1 + J * R].
Ausgabe: logits [B, 1 + J * R], Position 0 = WAIT, danach 1 + Job * R + Ressource (wie in rl/env.py).
Gesperrte Aktionen bekommen einen sehr kleinen Wert, damit sie nie gewaehlt werden.

TimeAwareDecoder: Jeder Kandidat besteht aus Operation, Ressource, Zeiteinbettung und den Zeitmerkmalen
des Paars (candidates [B, J, R, F_c]). Ruestung und Bearbeitung kommen aus dem Zustand am
tatsaechlichen Beginn, nicht aus dem Encoder. Zum Wert des Pairwise-MLP kommt ein Produkt aus
Knoten und Zeit, wie in NCO4TDTSP (Yang & Fan 2025, Gl. 8): Nur so kann die Uhrzeit bestimmen, welcher
Teil des Zeitprofils einer Ressource gerade zaehlt. Ein MLP allein findet diese Kombination kaum, weil
weder Profil noch Uhrzeit fuer sich etwas ueber die bessere Ressource sagen.

StaticPairDecoder (Variante A0, Aufbau wie der L2D-Decoder von RL4CO): nur Operation und Ressource;
WAIT ist ein gelernter fester Wert ohne Bezug zum Zustand.
"""
from __future__ import annotations

import math

import torch
from torch import nn

MASKED_LOGIT = -1e9
_CLOCK_FREQUENCIES = (1.0, 2.0, 4.0, 8.0, 16.0)     # Schwingungen je Tag; 16 trennt noch 30 Minuten


def _next_op_embedding(h_op: torch.Tensor, next_op: torch.Tensor) -> torch.Tensor:
    return h_op.gather(1, next_op.unsqueeze(-1).expand(-1, -1, h_op.shape[-1]))


def _apply_mask(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~action_mask, MASKED_LOGIT)


class TimeEmbedding(nn.Module):
    """time_features [B, F_t] -> [B, hidden]; die Uhrzeit (Spalte 0, Anteil am Tag) zusaetzlich sinusfoermig."""

    def __init__(self, time_dim: int, hidden: int):
        super().__init__()
        self.register_buffer("frequencies", torch.tensor(_CLOCK_FREQUENCIES))
        self.mlp = nn.Sequential(nn.Linear(time_dim + 2 * len(_CLOCK_FREQUENCIES), hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden))

    def forward(self, time_features: torch.Tensor) -> torch.Tensor:
        angles = 2 * math.pi * time_features[:, :1] * self.frequencies
        return self.mlp(torch.cat((time_features, torch.sin(angles), torch.cos(angles)), dim=-1))


class TimeAwareDecoder(nn.Module):
    def __init__(self, d_model: int, candidate_dim: int, time_dim: int, state_dim: int, hidden: int):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim, hidden)
        self.w_time = nn.Linear(hidden, d_model, bias=False)
        self.w_op_time = nn.Linear(d_model, d_model, bias=False)
        self.w_machine_time = nn.Linear(d_model, d_model, bias=False)
        self.d_model = d_model
        self.pair_encoder = nn.Sequential(nn.Linear(2 * d_model + hidden + candidate_dim, hidden), nn.ReLU())
        self.pair_score = nn.Linear(hidden, 1)
        self.wait_score = nn.Sequential(nn.Linear(2 * hidden + 2 * d_model + state_dim, hidden), nn.ReLU(),
                                        nn.Linear(hidden, 1))

    def forward(self, h_op: torch.Tensor, h_ma: torch.Tensor, op_pooled: torch.Tensor,
                machine_pooled: torch.Tensor, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        jobs, resources = inputs["next_op"].shape[1], h_ma.shape[1]
        ops = _next_op_embedding(h_op, inputs["next_op"]).unsqueeze(2).expand(-1, -1, resources, -1)
        machines = h_ma.unsqueeze(1).expand(-1, jobs, -1, -1)
        time = self.time_embedding(inputs["time_features"])
        times = time[:, None, None, :].expand(-1, jobs, resources, -1)

        pairs = self.pair_encoder(torch.cat((ops, machines, times, inputs["candidates"]), dim=-1))
        node_time = ((self.w_op_time(ops) + self.w_machine_time(machines)) * self.w_time(times)).sum(-1)
        pair_logits = self.pair_score(pairs).squeeze(-1) + node_time / math.sqrt(self.d_model)

        # WAIT sieht alle zulaessigen Paare, auch solche, deren Ressource gerade belegt ist
        valid = inputs["candidate_valid"].unsqueeze(-1)
        pooled = (pairs * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp(min=1)
        wait_input = torch.cat((pooled, time, op_pooled, machine_pooled, inputs["state_features"]), dim=-1)
        wait_logit = self.wait_score(wait_input)
        return _apply_mask(torch.cat((wait_logit, pair_logits.flatten(1)), dim=1), inputs["action_mask"])


class StaticPairDecoder(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.pair_score = nn.Sequential(nn.Linear(2 * d_model, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.wait_logit = nn.Parameter(torch.zeros(1))

    def forward(self, h_op: torch.Tensor, h_ma: torch.Tensor, op_pooled: torch.Tensor,
                machine_pooled: torch.Tensor, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        jobs, resources = inputs["next_op"].shape[1], h_ma.shape[1]
        ops = _next_op_embedding(h_op, inputs["next_op"]).unsqueeze(2).expand(-1, -1, resources, -1)
        machines = h_ma.unsqueeze(1).expand(-1, jobs, -1, -1)
        pair_logits = self.pair_score(torch.cat((ops, machines), dim=-1)).squeeze(-1)
        wait_logit = self.wait_logit.expand(h_op.shape[0], 1)
        return _apply_mask(torch.cat((wait_logit, pair_logits.flatten(1)), dim=1), inputs["action_mask"])
