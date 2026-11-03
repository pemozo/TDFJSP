"""Heterogener Encoder: Operationen und Ressourcen tauschen ueber zulaessige Kanten Nachrichten aus.

Eingaben (siehe rl/observation.py): op_features [B, O, F_op], machine_features [B, R, F_ma],
Kantenmerkmale aus proc_profiles [B, O, R, K], proc_ref [B, O, R] und profile_lengths [K],
eligible [B, O, R], precedence [B, O, O], pad_mask [B, O].
Ausgabe: h_op [B, O, d_model], h_ma [B, R, d_model].

Eine Schicht berechnet je Richtung und Kopf:
    e        = dot(W_Q * h_ziel, W_K * h_quelle) / sqrt(d_head)              [B, O, R, H]
    e_temporal = AttentionMLP(concat(e, Kantenmerkmale))
    alpha    = masked_softmax(e_temporal, ueber die Nachbarn)
    message  = W_V * h_quelle + EdgeValueMLP(Kantenmerkmale)
    aggregiert = sum(alpha * message)

Die beiden Richtungen haben eigene Parameter und werden getrennt normalisiert. Danach folgt die
Aktualisierung mit Residual-Verbindung, MLP und LayerNorm; Operationen bekommen zusaetzlich die
Einbettungen ihrer Vorgaenger und Nachfolger im Job (wie bei HGNN nach Song et al.).

Kantenmerkmale: statischer Encoder nur [proc_ref] (Abmessung 1), zeitabhaengiger Encoder
[proc_ref, proc_profiles, profile_lengths] (Abmessung 1 + 2K). Die Fensterlaengen stehen dabei, weil
die Fenster ungleich lang sind und K Werte allein das nicht zeigen wuerden.
"""
from __future__ import annotations

import math

import torch
from torch import nn

_MASKED = -1e9


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Softmax nur ueber gueltige Eintraege. Ohne gueltigen Eintrag ergibt sich 0 statt NaN."""
    has_valid = mask.any(dim=dim, keepdim=True)
    filled = scores.masked_fill(~mask, _MASKED).masked_fill(~has_valid, 0.0)
    return torch.softmax(filled, dim=dim) * mask


def _mlp(input_dim: int, hidden: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, output_dim))


class _DirectedAttention(nn.Module):
    """Nachrichten von den Quellknoten zu den Zielknoten; Kantenformen [B, Ziel, Quelle, ...]."""

    def __init__(self, d_model: int, num_heads: int, edge_dim: int, hidden: int, use_edge_values: bool):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.w_query = nn.Linear(d_model, d_model, bias=False)
        self.w_key = nn.Linear(d_model, d_model, bias=False)
        self.w_value = nn.Linear(d_model, d_model, bias=False)
        self.attention_mlp = _mlp(1 + edge_dim, hidden, 1)
        self.edge_value_mlp = _mlp(edge_dim, hidden, d_model) if use_edge_values else None

    def _heads(self, values: torch.Tensor) -> torch.Tensor:
        return values.view(*values.shape[:-1], self.num_heads, self.d_head)

    def forward(self, target: torch.Tensor, source: torch.Tensor, edges: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        query = self._heads(self.w_query(target))
        key = self._heads(self.w_key(source))
        scores = torch.einsum("btha,bsha->btsh", query, key) / math.sqrt(self.d_head)
        repeated = edges.unsqueeze(3).expand(-1, -1, -1, self.num_heads, -1)
        temporal = self.attention_mlp(torch.cat((scores.unsqueeze(-1), repeated), dim=-1)).squeeze(-1)
        alpha = masked_softmax(temporal, mask.unsqueeze(-1), dim=2)

        message = self._heads(self.w_value(source)).unsqueeze(1)
        if self.edge_value_mlp is not None:
            message = message + self._heads(self.edge_value_mlp(edges))
        aggregated = (alpha.unsqueeze(-1) * message).sum(dim=2)
        return aggregated.flatten(2)


class TemporalHetGNNLayer(nn.Module):
    """Eine Schicht des Encoders; h_op [B, O, d_model], h_ma [B, R, d_model]."""

    def __init__(self, d_model: int, num_heads: int, edge_dim: int, hidden: int, use_edge_values: bool):
        super().__init__()
        self.machines_to_ops = _DirectedAttention(d_model, num_heads, edge_dim, hidden, use_edge_values)
        self.ops_to_machines = _DirectedAttention(d_model, num_heads, edge_dim, hidden, use_edge_values)
        self.w_predecessor = nn.Linear(d_model, d_model, bias=False)
        self.w_successor = nn.Linear(d_model, d_model, bias=False)
        self.op_update = _mlp(4 * d_model, 2 * d_model, d_model)
        self.machine_update = _mlp(2 * d_model, 2 * d_model, d_model)
        self.op_norm = nn.LayerNorm(d_model)
        self.machine_norm = nn.LayerNorm(d_model)

    def forward(self, h_op: torch.Tensor, h_ma: torch.Tensor, edges: torch.Tensor, eligible: torch.Tensor,
                precedence: torch.Tensor, pad_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from_machines = self.machines_to_ops(h_op, h_ma, edges, eligible)
        from_ops = self.ops_to_machines(h_ma, h_op, edges.transpose(1, 2), eligible.transpose(1, 2))

        links = precedence.to(h_op.dtype)
        predecessors = torch.bmm(links.transpose(1, 2), h_op) / links.sum(1, keepdim=True).transpose(1, 2).clamp(min=1)
        successors = torch.bmm(links, h_op) / links.sum(2, keepdim=True).clamp(min=1)
        op_input = torch.cat((h_op, from_machines, self.w_predecessor(predecessors),
                              self.w_successor(successors)), dim=-1)
        new_op = self.op_norm(h_op + self.op_update(op_input))
        new_ma = self.machine_norm(h_ma + self.machine_update(torch.cat((h_ma, from_ops), dim=-1)))
        return new_op.masked_fill(pad_mask.unsqueeze(-1), 0.0), new_ma


class HetGNNEncoder(nn.Module):
    """temporal=True: zeitabhaengige Kanten; temporal=False: nur die Basisdauer als Kante.
    use_edge_values=False: Kanten wirken nur auf die Attention, nicht auf die Nachrichten."""

    def __init__(self, op_dim: int, machine_dim: int, num_windows: int, d_model: int, num_layers: int,
                 num_heads: int, hidden: int, temporal: bool, use_edge_values: bool):
        super().__init__()
        self.temporal = temporal
        edge_dim = 1 + 2 * num_windows if temporal else 1
        self.op_embedding = nn.Linear(op_dim, d_model)
        self.machine_embedding = nn.Linear(machine_dim, d_model)
        self.layers = nn.ModuleList(
            TemporalHetGNNLayer(d_model, num_heads, edge_dim, hidden, use_edge_values)
            for _ in range(num_layers))

    def edge_features(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Kantenmerkmale [B, O, R, E]; unzulaessige Kanten sind 0."""
        reference = inputs["proc_ref"].unsqueeze(-1)
        if not self.temporal:
            return reference
        profiles = inputs["proc_profiles"]
        lengths = inputs["profile_lengths"].expand(*profiles.shape)
        lengths = lengths * inputs["eligible"].unsqueeze(-1)
        return torch.cat((reference, profiles, lengths), dim=-1)

    def forward(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pad_mask, eligible = inputs["pad_mask"], inputs["eligible"]
        edges = self.edge_features(inputs)
        h_op = self.op_embedding(inputs["op_features"]).masked_fill(pad_mask.unsqueeze(-1), 0.0)
        h_ma = self.machine_embedding(inputs["machine_features"])
        for layer in self.layers:
            h_op, h_ma = layer(h_op, h_ma, edges, eligible, inputs["precedence"], pad_mask)
        return h_op, h_ma
