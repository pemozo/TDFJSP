"""Ueberwachter Lerntest: Kann die Architektur zeitabhaengig die guenstigere Maschine waehlen?

Jede Instanz hat eine Operation und zwei Maschinen mit gleicher mittlerer Dauer. Welche Maschine
guenstiger ist, haengt vom Zeitfenster ab. Das Netz sieht die Uhrzeit und die Profile; das Ziel ist die
Maschine mit der kleineren Dauer im aktuellen Fenster. Ohne PPO, damit nur die Architektur geprueft wird.

Muster:
    two_phase        eine Maschine ist vormittags guenstiger, die andere nachmittags
    random_windows   je Fenster zufaellig, welche Maschine guenstiger ist (schwerer)
candidates_visible:
    False  Der Decoder sieht die Dauer am Start nicht; die Information muss durch den Encoder kommen.
    True   Der Decoder sieht die Dauer am tatsaechlichen Start (so wie in der echten Umgebung).

Aufruf: python -m tests.synthetic_learning
"""
from __future__ import annotations

from multiprocessing import Pool

import torch

from rl.observation import CANDIDATE_FEATURES, MACHINE_FEATURES, OP_FEATURES, STATE_FEATURES
from rl.policy import ActorCritic
from rl.time_model import TIME_FEATURES

WINDOW_LENGTHS = torch.tensor([180.0, 60.0, 120.0, 30.0, 90.0, 120.0, 120.0]) / 720
NUM_WINDOWS = len(WINDOW_LENGTHS)
_PROCESSING = CANDIDATE_FEATURES.index("processing_time")


def make_batch(size: int, pattern: str, candidates_visible: bool, generator: torch.Generator):
    """Eingaben fuer eine Operation, einen Job und zwei Maschinen; Ziel: Index der guenstigeren Maschine."""
    if pattern == "two_phase":
        sign = torch.randint(0, 2, (size, 1), generator=generator).float() * 2 - 1
        shape = torch.tensor([1.0, 1.0, 1.0, 0.0, -1.0, -1.0, -1.0]).expand(size, -1)
        delta = 0.3 * sign * shape
    elif pattern == "random_windows":
        delta = 0.3 * (torch.randint(0, 2, (size, NUM_WINDOWS), generator=generator).float() * 2 - 1)
        delta = delta - delta.mean(1, keepdim=True)     # beide Maschinen: gleiches Mittel
    else:
        raise ValueError(pattern)
    profiles = torch.stack((1 + delta, 1 - delta), dim=1).unsqueeze(1)      # [B, O=1, R=2, K]

    usable = torch.tensor([0, 1, 2, 4, 5, 6])
    window = usable[torch.randint(0, len(usable), (size,), generator=generator)]
    current = profiles[torch.arange(size), 0, :, window]                    # [B, 2]
    target = current.argmin(1)

    static = {"proc_profiles": profiles,
              "proc_ref": torch.ones(size, 1, 2),
              "profile_lengths": WINDOW_LENGTHS,
              "eligible": torch.ones(size, 1, 2, dtype=torch.bool),
              "precedence": torch.zeros(size, 1, 1, dtype=torch.bool),
              "pad_mask": torch.zeros(size, 1, dtype=torch.bool),
              "op_features": torch.zeros(size, 1, len(OP_FEATURES)),
              "machine_features": torch.zeros(size, 2, len(MACHINE_FEATURES))}

    time_features = torch.zeros(size, len(TIME_FEATURES))
    starts = torch.cumsum(torch.cat((torch.zeros(1), WINDOW_LENGTHS[:-1])), 0)
    time_features[:, 0] = starts[window] + WINDOW_LENGTHS[window] / 2       # Uhrzeit im Tag
    candidates = torch.zeros(size, 1, 2, len(CANDIDATE_FEATURES))
    if candidates_visible:
        candidates[:, 0, :, _PROCESSING] = current
    mask = torch.ones(size, 3, dtype=torch.bool)
    mask[:, 0] = False      # WAIT ist hier keine Antwort
    dynamic = {"next_op": torch.zeros(size, 1, dtype=torch.long), "candidates": candidates,
               "candidate_valid": torch.ones(size, 1, 2, dtype=torch.bool),
               "time_features": time_features,
               "state_features": torch.zeros(size, len(STATE_FEATURES)),
               "action_mask": mask}
    return static, dynamic, target


def train_and_measure(variant: str, pattern: str, candidates_visible: bool, steps: int = 1500,
                      seed: int = 0) -> float:
    """Trainiert ueberwacht und gibt die Trefferquote auf neuen Beispielen zurueck."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)    # die Batches sind klein, mehr Threads bremsen nur
    try:
        return _train_and_measure(variant, pattern, candidates_visible, steps, seed)
    finally:
        torch.set_num_threads(threads)


def _train_and_measure(variant: str, pattern: str, candidates_visible: bool, steps: int, seed: int) -> float:
    torch.manual_seed(seed)
    settings = {"variant": variant, "d_model": 64, "num_layers": 2, "num_heads": 4, "hidden_dim": 128}
    model = ActorCritic(settings, NUM_WINDOWS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        static, dynamic, target = make_batch(128, pattern, candidates_visible, generator)
        logits, _ = model(*model.encode(static), static["pad_mask"], dynamic)
        loss = torch.nn.functional.cross_entropy(logits, target + 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        static, dynamic, target = make_batch(4000, pattern, candidates_visible, generator)
        logits, _ = model(*model.encode(static), static["pad_mask"], dynamic)
        return float((logits.argmax(1) == target + 1).to(torch.float32).mean())


def _run(task: tuple) -> float:
    return train_and_measure(*task)


if __name__ == "__main__":
    # A0 sieht die Kandidatenmerkmale nicht, deshalb dort nur eine Zeile
    tasks = [(variant, pattern, visible) for pattern in ("two_phase", "random_windows")
             for visible in (False, True) for variant in ("A0", "A1", "A2", "A3")
             if not (variant == "A0" and visible)]
    with Pool(8) as pool:
        results = pool.map(_run, tasks)
    print(f"{'Muster':<16} {'Variante':<9} {'Dauer im Decoder':<17} Trefferquote")
    for (variant, pattern, visible), accuracy in zip(tasks, results):
        print(f"{pattern:<16} {variant:<9} {'ja' if visible else 'nein':<17} {accuracy:.3f}")
