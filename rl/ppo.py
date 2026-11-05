"""PPO mit Generalized Advantage Estimation; alle Parameter stehen im Abschnitt training der Config.

Eine Iteration:
    collect_rollout   ganze Episoden eines Batches spielen; der Encoder laeuft dabei je Instanz einmal
    compute_returns   Vorteil und Return aus den Rewards der Umgebung
    ppo_update        mehrere Epochen ueber zufaellige Minibatches; hier laeuft der Encoder mit Gradienten
                      fuer die Instanzen des Minibatches
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from rl.env import TDFJSPEnv
from rl.observation import ObservationBuilder
from rl.policy import ActorCritic

_DYNAMIC_KEYS = ("next_op", "candidates", "candidate_valid", "time_features", "state_features", "action_mask")


@dataclass
class Rollout:
    """Uebergaenge laufender Episoden; alle Tensoren haben N Eintraege in erster Dimension."""

    static: dict[str, torch.Tensor]
    dynamic: dict[str, torch.Tensor]
    instance: torch.Tensor
    action: torch.Tensor
    log_prob: torch.Tensor
    advantage: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    final_state: dict[str, torch.Tensor]
    wait_share: float
    wait_actions: torch.Tensor      # je Instanz [B]
    decisions: torch.Tensor         # je Instanz [B]


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float | None:
    """1 - Var(Return - Wert) / Var(Return); None, wenn die Returns nicht streuen."""
    variance = returns.var()
    if not variance > 0:
        return None
    return float(1 - (returns - values).var() / variance)


def compute_returns(rewards: torch.Tensor, values: torch.Tensor, alive: torch.Tensor, gamma: float,
                    gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Formen [T, B]. alive[t, b]: Instanz b hat in Schritt t noch entschieden. Nach dem letzten
    lebenden Schritt ist der Wert des Folgezustands 0."""
    steps, batch = rewards.shape
    advantage = torch.zeros_like(rewards)
    running = torch.zeros(batch, dtype=rewards.dtype, device=rewards.device)
    next_value = torch.zeros(batch, dtype=rewards.dtype, device=rewards.device)
    next_alive = torch.zeros(batch, dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(steps)):
        delta = rewards[t] + gamma * next_value * next_alive - values[t]
        running = delta + gamma * gae_lambda * next_alive * running
        advantage[t] = running * alive[t]
        next_value, next_alive = values[t], alive[t].to(rewards.dtype)
    return advantage, advantage + values * alive


@torch.no_grad()
def collect_rollout(env: TDFJSPEnv, model: ActorCritic, builder: ObservationBuilder,
                    td: dict[str, torch.Tensor], settings: dict) -> Rollout:
    """Spielt alle Instanzen im Zustand td mit Stichproben aus der Policy bis zum Ende."""
    model.eval()
    static = builder.static_inputs(td)
    h_op, h_ma = model.encode(static)
    stored = {key: [] for key in _DYNAMIC_KEYS}
    rewards, values, alive, instances, actions, log_probs = [], [], [], [], [], []
    while not td["done"].all():
        dynamic = builder.dynamic_inputs(td)
        logits, value = model(h_op, h_ma, static["pad_mask"], dynamic)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        running = ~td["done"]
        index = running.nonzero().squeeze(1)
        for key in _DYNAMIC_KEYS:
            stored[key].append(dynamic[key][index])
        instances.append(index)
        actions.append(action[index])
        log_probs.append(distribution.log_prob(action)[index])
        td["action"] = action
        td = env.step(td)["next"]
        rewards.append(td["reward"].to(torch.float32) * settings["reward_scale"])
        values.append(value)
        alive.append(running)

    reward_matrix, value_matrix = torch.stack(rewards), torch.stack(values)
    alive_matrix = torch.stack(alive)
    advantage, returns = compute_returns(reward_matrix, value_matrix, alive_matrix,
                                         settings["gamma"], settings["gae_lambda"])
    action_all, instance_all = torch.cat(actions), torch.cat(instances)
    return Rollout(static=static, dynamic={key: torch.cat(stored[key]) for key in _DYNAMIC_KEYS},
                   instance=instance_all, action=action_all, log_prob=torch.cat(log_probs),
                   advantage=advantage[alive_matrix], returns=returns[alive_matrix], values=value_matrix[alive_matrix],
                   final_state=td, wait_share=float((action_all == 0).to(torch.float32).mean().item()),
                   wait_actions=torch.bincount(instance_all[action_all == 0], minlength=alive_matrix.shape[1]),
                   decisions=alive_matrix.sum(0))


def ppo_update(model: ActorCritic, optimizer: torch.optim.Optimizer, rollout: Rollout, settings: dict,
               on_update: Callable[[dict[str, float], int], None] | None = None) -> dict[str, float]:
    """Mehrere Epochen ueber zufaellige Minibatches. on_update(Kennzahlen, Groesse) laeuft nach jedem
    Optimizer-Schritt. Rueckgabe: Mittel ueber die Minibatches, gewichtet mit ihrer Groesse, und
    explained_variance aus den Werten der Datensammlung (fehlt, wenn die Returns nicht streuen)."""
    model.train()
    count = rollout.action.shape[0]
    advantage = (rollout.advantage - rollout.advantage.mean()) / (rollout.advantage.std(unbiased=count > 1) + 1e-8)
    totals = {key: 0.0 for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm")}
    seen = 0
    stopped = False
    for _ in range(settings["ppo_epochs"]):
        order = torch.randperm(count, device=rollout.action.device)
        for first in range(0, count, settings["minibatch_size"]):
            chosen = order[first:first + settings["minibatch_size"]]
            used, position = torch.unique(rollout.instance[chosen], return_inverse=True)
            static = {key: (value[used] if key not in ("profile_lengths", "profile_starts") else value)
                      for key, value in rollout.static.items()}
            h_op, h_ma = model.encode(static)
            dynamic = {key: value[chosen] for key, value in rollout.dynamic.items()}
            logits, value = model(h_op[position], h_ma[position], static["pad_mask"][position], dynamic)
            distribution = Categorical(logits=logits)
            log_ratio = distribution.log_prob(rollout.action[chosen]) - rollout.log_prob[chosen]
            ratio = torch.exp(log_ratio)
            approximate_kl = ((ratio - 1) - log_ratio).mean()
            target_kl = settings.get("target_kl")
            if target_kl is not None and approximate_kl.detach().item() > 1.5 * target_kl:
                stopped = True
                break
            gain = advantage[chosen]
            clipped = torch.clamp(ratio, 1 - settings["clip_range"], 1 + settings["clip_range"])
            policy_loss = -torch.min(ratio * gain, clipped * gain).mean()
            value_loss = torch.nn.functional.mse_loss(value, rollout.returns[chosen])
            entropy = distribution.entropy().mean()
            loss = (policy_loss + settings["value_coef"] * value_loss
                    - settings["entropy_coef"] * entropy)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), settings["max_grad_norm"])
            optimizer.step()

            with torch.no_grad():
                metrics = {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
                           "entropy": entropy.item(), "approx_kl": ((ratio - 1) - log_ratio).mean().item(),
                           "clip_fraction": ((ratio - 1).abs() > settings["clip_range"]).to(torch.float32).mean().item(),
                           "grad_norm": grad_norm.item()}
            samples = chosen.shape[0]
            for key, value in metrics.items():
                totals[key] += value * samples
            seen += samples
            if on_update is not None:
                on_update(metrics, samples)
        if stopped:
            break
    result = {key: total / max(seen, 1) for key, total in totals.items()}
    variance = explained_variance(rollout.values, rollout.returns)
    if variance is not None:
        result["explained_variance"] = variance
    return result
