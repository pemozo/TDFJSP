"""Actor-Critic aus Encoder, Decoder und Critic; die Variante steht in model.variant.

| Variante | Encoder                                  | Decoder            |
|----------|------------------------------------------|--------------------|
| A0       | statisch (nur Basisdauer)                | StaticPairDecoder  |
| A1       | statisch (nur Basisdauer)                | TimeAwareDecoder   |
| A2       | zeitabhaengige Attention, ohne EdgeValue | TimeAwareDecoder   |
| A3       | zeitabhaengige Attention und EdgeValue   | TimeAwareDecoder   |

Unabhaengig davon: model.time_information (full, none) und environment.allow_wait.

Der Encoder laeuft je Instanz einmal; Decoder und Critic bekommen den aktuellen Zustand je Schritt.
A0 ist deshalb nicht das HGNN aus Song et al. (2022): Dort laeuft der Encoder je Schritt auf dem aktuellen
Graphen, A0 sieht gar keinen dynamischen Zustand.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

from rl.decoder import StaticPairDecoder, TimeAwareDecoder
from rl.encoder import HetGNNEncoder
from rl.env import TDFJSPEnv
from rl.observation import (CANDIDATE_FEATURES, MACHINE_FEATURES, OP_FEATURES, STATE_FEATURES,
                            ObservationBuilder)
from rl.time_model import TIME_FEATURES

# Variante -> (zeitabhaengiger Encoder, Edge-Values, zeitbewusster Decoder)
VARIANTS = {"A0": (False, True, False), "A1": (False, True, True),
            "A2": (True, False, True), "A3": (True, True, True)}


def pooled(h_op: torch.Tensor, h_ma: torch.Tensor,
           pad_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mittel der Einbettungen ohne Padding: [B, d] und [B, d]."""
    keep = (~pad_mask).unsqueeze(-1).to(h_op.dtype)
    return (h_op * keep).sum(1) / keep.sum(1).clamp(min=1), h_ma.mean(1)


class ActorCritic(nn.Module):
    def __init__(self, model_settings: dict, num_windows: int):
        super().__init__()
        temporal, edge_values, time_aware = VARIANTS[model_settings["variant"]]
        d_model, hidden = model_settings["d_model"], model_settings["hidden_dim"]
        self.encoder = HetGNNEncoder(len(OP_FEATURES), len(MACHINE_FEATURES), num_windows, d_model,
                                     model_settings["num_layers"], model_settings["num_heads"], hidden,
                                     temporal, edge_values)
        self.decoder = (TimeAwareDecoder(d_model, len(CANDIDATE_FEATURES), len(TIME_FEATURES),
                                         len(STATE_FEATURES), hidden)
                        if time_aware else StaticPairDecoder(d_model, hidden))
        self.critic = nn.Sequential(nn.Linear(2 * d_model + len(TIME_FEATURES) + len(STATE_FEATURES), hidden),
                                    nn.ReLU(), nn.Linear(hidden, 1))

    def encode(self, static: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(static)

    def forward(self, h_op: torch.Tensor, h_ma: torch.Tensor, pad_mask: torch.Tensor,
                dynamic: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits [B, 1 + J * R] und Zustandswert [B]."""
        op_pooled, machine_pooled = pooled(h_op, h_ma, pad_mask)
        logits = self.decoder(h_op, h_ma, op_pooled, machine_pooled, dynamic)
        value = self.critic(torch.cat((op_pooled, machine_pooled, dynamic["time_features"],
                                       dynamic["state_features"]), dim=-1)).squeeze(-1)
        return logits, value


def build_model(env: TDFJSPEnv) -> ActorCritic:
    return ActorCritic(env.config["model"], len(env.time_model.window_starts)).to(env.device)


class NeuralPolicy:
    """Policy-Funktion fuer rollout und rl/evaluate.py. Die Einbettungen werden je Batch einmal
    berechnet; die festen Instanzdaten bleiben im Zustand dasselbe Objekt und dienen als Schluessel."""

    def __init__(self, env: TDFJSPEnv, model: ActorCritic, decode: str = "greedy", seed: int = 0):
        self.builder = ObservationBuilder(env)
        self.model = model.eval()
        self.decode = decode
        self._generator = torch.Generator(device=env.device).manual_seed(seed)
        self._key = None
        self._hidden = None
        self._pad_mask = None

    @torch.no_grad()
    def __call__(self, td: dict[str, torch.Tensor]) -> torch.Tensor:
        if self._key is not td["proc_times"]:
            static = self.builder.static_inputs(td)
            self._hidden = self.model.encode(static)
            self._pad_mask = static["pad_mask"]
            self._key = td["proc_times"]
        logits, _ = self.model(*self._hidden, self._pad_mask, self.builder.dynamic_inputs(td))
        if self.decode == "greedy":
            return logits.argmax(1)
        probabilities = Categorical(logits=logits).probs
        return torch.multinomial(probabilities, 1, generator=self._generator).squeeze(1)


def model_name(config: dict) -> str:
    """Name der Modellkonfiguration ohne Seed, z. B. A3_zeit_warten; gleich fuer alle Seeds."""
    information = "zeit" if config["model"]["time_information"] == "full" else "ohnezeit"
    waiting = "warten" if config["environment"]["allow_wait"] else "nondelay"
    return f"{config['model']['variant']}_{information}_{waiting}"


def run_name(config: dict, seed: int) -> str:
    """Name eines Laufs aus Variante, Algorithmus und Seed, z. B. A3_zeit_warten_ppo_seed0."""
    return f"{model_name(config)}_ppo_seed{seed}"


def load_checkpoint(path: Path | str, config_path: Path | str | None = None) -> tuple[TDFJSPEnv, ActorCritic, str]:
    """Umgebung und Modell genau mit den Einstellungen des Trainings; Rueckgabe mit Laufnamen."""
    from schedule_validator import DEFAULT_CONFIG

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    env = TDFJSPEnv(checkpoint["config"], config_path=DEFAULT_CONFIG if config_path is None else config_path)
    model = build_model(env)
    model.load_state_dict(checkpoint["model"])
    return env, model, checkpoint["name"]
