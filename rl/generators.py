"""Instanzen fuer die Umgebung: on the fly fuers Training, aus Dateien fuer val und test.

Beide Generatoren liefern die festen Instanzdaten als Batch (rl.parser). Die Logik der
Instanzerzeugung liegt nur in generator.py, damit alle Datensaetze derselben Verteilung folgen.
"""
from __future__ import annotations

import multiprocessing
from pathlib import Path

import torch

from generator import generate_class_instances
from rl.parser import instances_to_td
from schedule_validator import (DEFAULT_CONFIG, instance_files, load_json, selected_classes,
                                split_folder, validate_instance)


class TDFJSPGenerator:
    """Neue Instanzen je Aufruf. Die Klassen wechseln je Batch in fester Reihenfolge; die Seeds
    laufen ab data_splits.train.seed_start weiter. Das Ergebnis haengt nur von den Einstellungen
    ab, nicht von der Anzahl der Prozesse (environment.generator_workers)."""

    def __init__(self, config: dict, classes: str | list[str] | None = None,
                 config_path: Path | str = DEFAULT_CONFIG):
        selection = config["environment"]["train_classes"] if classes is None else classes
        self.config = config
        self.config_path = config_path
        self.classes = selected_classes(config, selection)
        self.next_seed = config["data_splits"]["train"]["seed_start"]
        self.skipped_seeds = 0
        self._workers = config["environment"]["generator_workers"]
        self._pool = None
        self._cursor = 0

    def sample_instances(self, count: int, class_name: str | None = None) -> list[dict]:
        """Die naechsten `count` Instanzen. Ohne class_name: naechste Klasse der Reihe."""
        if class_name is None:
            class_name = self.classes[self._cursor % len(self.classes)]
            self._cursor += 1
        if self._workers > 0 and self._pool is None:
            # spawn: gleiches Verhalten unter Windows und Linux, kein Fork nach Torch-Threads
            self._pool = multiprocessing.get_context("spawn").Pool(self._workers)
        instances, self.next_seed, skipped = generate_class_instances(
            self.config, class_name, self.next_seed, count, self.config_path, self._pool)
        self.skipped_seeds += skipped
        return instances

    def __call__(self, batch_size: int, class_name: str | None = None) -> dict[str, torch.Tensor]:
        return instances_to_td(self.config, self.sample_instances(batch_size, class_name))

    def close(self) -> None:
        """Prozesse beenden (nur noetig mit generator_workers > 0)."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None


class TDFJSPFileGenerator:
    """Feste Instanzen einer Klasse aus data_splits.<split>.folder, in Dateireihenfolge.
    Jede Datei wird gegen die aktuelle Konfiguration geprueft (Fingerprint, Struktur)."""

    def __init__(self, config: dict, class_name: str, split: str,
                 config_path: Path | str = DEFAULT_CONFIG):
        folder = split_folder(config, class_name, split, config_path)
        self.config = config
        self.files = instance_files([folder]) if folder.is_dir() else []
        if not self.files:
            raise FileNotFoundError(f"Keine Instanzen in {folder}. Erzeugen mit: "
                                    f"python generator.py --classes {class_name} --splits {split}")
        self.instances = [load_json(path) for path in self.files]
        for path, instance in zip(self.files, self.instances):
            errors = validate_instance(config, instance)
            if errors:
                raise ValueError(f"{path}: {errors[0]}")
        self._cursor = 0

    def __len__(self) -> int:
        return len(self.instances)

    def __call__(self, batch_size: int | None = None) -> dict[str, torch.Tensor]:
        """Naechste Instanzen der Reihe nach (von vorn, wenn die Dateien zu Ende sind);
        ohne batch_size alle in Dateireihenfolge. instance_index sagt, welche Datei auf
        welcher Position liegt."""
        if batch_size is None:
            indices = list(range(len(self)))
        else:
            indices = [(self._cursor + i) % len(self) for i in range(batch_size)]
            self._cursor = (indices[-1] + 1) % len(self)
        td = instances_to_td(self.config, [self.instances[i] for i in indices])
        td["instance_index"] = torch.tensor(indices, dtype=torch.long)
        return td
