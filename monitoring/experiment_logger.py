"""Experiment-Tracking je Lauf: TensorBoard-Kurven, Kennzahlen als CSV, Gantt-Diagramme, Metadaten.

Unabhaengig vom Trainingscode, von RL4CO und Lightning. Alle log_*-Methoden nehmen ein dict, neue
Kennzahlen brauchen also keine Aenderung hier. None = Kennzahl nicht verfuegbar: Sie wird in run.json
vermerkt und nicht geschaetzt. NaN und Inf loesen eine Warnung aus und werden nicht geschrieben.

Ablage unter <root>:
    tensorboard/<run_id>/                    Ereignisdateien
    results/<run_id>/run.json                Metadaten: Variante, Seed, Ziel, Laufzeit, bestes Modell, ...
    results/<run_id>/config.json             vollstaendige Einstellungen
    results/<run_id>/scalars.csv             step, tag, value, elapsed_seconds
    results/<run_id>/<split>_instances.csv   eine Zeile je Instanz, getrennt nach train, val und test
    figures/<run_id>/gantt/                  Gantt-Diagramme als PNG und PDF
"""
from __future__ import annotations

import csv
import json
import math
import numbers
import os
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

SCALAR_COLUMNS = ("step", "tag", "value", "elapsed_seconds")
# Immer vorhanden und in dieser Reihenfolge; leer = nicht verfuegbar. Weitere Spalten kommen dahinter.
INSTANCE_COLUMNS = ("run_id", "model", "seed", "step", "instance_id", "instance_size", "temporal_variability",
                    "reward", "makespan", "total_tardiness", "window_compliance", "feasible",
                    "processing_time_total", "setup_time_total", "voluntary_idle_time", "wait_actions",
                    "inference_time")


def reserve_run_id(name: str, folders: list[Path]) -> str:
    """Erster freier Name aus name, name_r2, name_r3, ... mkdir im ersten Ordner ist atomar: Auch gleichzeitig
    gestartete Laeufe bekommen verschiedene Namen."""
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        run_id = name if attempt == 1 else f"{name}_r{attempt}"
        attempt += 1
        if any((folder / run_id).exists() for folder in folders[1:]):
            continue
        try:
            (folders[0] / run_id).mkdir()
        except FileExistsError:
            continue
        for folder in folders[1:]:
            (folder / run_id).mkdir(exist_ok=True)
        return run_id


def git_commit(folder: Path) -> str | None:
    """Aktueller Commit; None, wenn der Ordner kein Git-Repository ist oder Git fehlt."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=folder, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _write_json(path: Path, data: dict) -> None:
    """Erst in eine Hilfsdatei, damit ein Abbruch keine halbe Datei hinterlaesst."""
    helper = path.with_name(path.name + ".tmp")
    helper.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(helper, path)


def _read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def _write_rows(path: Path, columns: list[str], rows: list[dict]) -> None:
    helper = path.with_name(path.name + ".tmp")
    with helper.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(helper, path)


class NullLogger:
    """Steht fuer den Logger, wenn das Monitoring ausgeschaltet ist; jeder Aufruf tut nichts."""

    enabled = False

    def __getattr__(self, name: str):
        return lambda *args, **kwargs: None


class ExperimentLogger:
    """name: gewuenschter Laufname. Mit reserved=True ist er schon vergeben (reserve_run_id, Fortsetzung).
    resume_step setzt einen Lauf nach einem Checkpoint fort: Alles Spaetere wird verworfen, die Schritte
    laufen konsistent weiter. config ist nur fuer die Gantt-Diagramme noetig."""

    enabled = True

    def __init__(self, name: str, root: Path | str, config: dict | None = None, metadata: dict | None = None,
                 reserved: bool = False, resume_step: int | None = None, flush_seconds: int = 30,
                 gantt_to_tensorboard: bool = True):
        root = Path(root)
        roots = [root / "results", root / "tensorboard", root / "figures"]
        self.run_id = name if reserved or resume_step is not None else reserve_run_id(name, roots)
        self.results_dir, self.tensorboard_dir, self.figures_dir = (folder / self.run_id for folder in roots)
        for folder in (self.results_dir, self.tensorboard_dir, self.figures_dir):
            folder.mkdir(parents=True, exist_ok=True)
        self.config = config
        self._gantt_to_tensorboard = gantt_to_tensorboard
        self._last_step: dict[str, int] = {}
        self._closed = False
        self._metadata_path = self.results_dir / "run.json"
        self._scalars_path = self.results_dir / "scalars.csv"

        self.metadata = self._initial_metadata(metadata or {}, resume_step)
        self._runtime_before = self.metadata.get("runtime_seconds", 0.0) if resume_step is not None else 0.0
        self._started = time.perf_counter()
        if resume_step is not None:
            self._discard_after(resume_step)
        if config is not None:
            _write_json(self.results_dir / "config.json", config)
        self._write_metadata()

        new_file = not self._scalars_path.exists()
        self._scalars_file = self._scalars_path.open("a", newline="", encoding="utf-8")
        self._scalars = csv.writer(self._scalars_file)
        if new_file:
            self._scalars.writerow(SCALAR_COLUMNS)
            self._scalars_file.flush()
        self._writer = self._open_writer(flush_seconds, resume_step)

    def _initial_metadata(self, metadata: dict, resume_step: int | None) -> dict:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if resume_step is not None and self._metadata_path.exists():
            previous = json.loads(self._metadata_path.read_text(encoding="utf-8"))
            previous.setdefault("resumed", []).append({"time": now, "step": resume_step})
            previous["status"] = "running"
            return previous
        return {"run_id": self.run_id, "status": "running", "start_time": now, "runtime_seconds": 0.0,
                "git_commit": git_commit(Path(__file__).parent), "best": None, "unavailable_metrics": {},
                "non_finite_metrics": {}, **metadata}

    def _open_writer(self, flush_seconds: int, resume_step: int | None):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            warnings.warn(f"TensorBoard fehlt, nur CSV wird geschrieben: {exc}", RuntimeWarning, stacklevel=3)
            self.mark_unavailable("tensorboard", "Paket tensorboard ist nicht installiert")
            return None
        # purge_step: Ereignisse nach dem Checkpoint verschwinden aus den Kurven
        purge = None if resume_step is None else resume_step + 1
        return SummaryWriter(str(self.tensorboard_dir), purge_step=purge, flush_secs=flush_seconds)

    def _discard_after(self, step: int) -> None:
        """Fortsetzung: Zeilen nach dem Checkpoint entfernen, sie gehoeren zum abgebrochenen Verlauf."""
        for path in (self._scalars_path, *self.results_dir.glob("*_instances.csv")):
            if path.exists():
                columns, rows = _read_rows(path)
                kept = [row for row in rows if not row.get("step") or int(row["step"]) <= step]
                _write_rows(path, columns, kept)
        self._last_step = {group: step for group in ("train", "ppo", "val", "temporal", "test")}

    # --- Kennzahlen ------------------------------------------------------------

    def log_train_metrics(self, metrics: dict, step: int) -> None:
        self._log("train", metrics, step)

    def log_ppo_metrics(self, metrics: dict, step: int) -> None:
        self._log("ppo", metrics, step)

    def log_validation_metrics(self, metrics: dict, step: int) -> None:
        self._log("val", metrics, step)

    def log_temporal_metrics(self, metrics: dict, step: int) -> None:
        self._log("temporal", metrics, step)

    def log_test_metrics(self, metrics: dict, step: int) -> None:
        self._log("test", metrics, step)

    def _log(self, group: str, metrics: dict, step: int) -> None:
        if step < self._last_step.get(group, step):
            warnings.warn(f"{group}: Schritt {step} liegt vor {self._last_step[group]}, nicht geschrieben",
                          RuntimeWarning, stacklevel=3)
            return
        self._last_step[group] = step
        elapsed = self.elapsed_seconds()
        for key, value in metrics.items():
            tag = key if key.startswith(f"{group}/") else f"{group}/{key}"
            if value is None:
                self.mark_unavailable(tag, "kein Wert geliefert")
            elif isinstance(value, (list, tuple)):      # Verteilung statt einzelner Zahl
                if value and self._writer is not None:
                    self._writer.add_histogram(tag, _as_array(value), step)
            elif isinstance(value, numbers.Real) or hasattr(value, "item"):
                self._log_scalar(tag, float(value), step, elapsed)
            else:
                self.mark_unavailable(tag, f"kein Zahlenwert: {type(value).__name__}")
        self._scalars_file.flush()

    def _log_scalar(self, tag: str, value: float, step: int, elapsed: float) -> None:
        if not math.isfinite(value):
            warnings.warn(f"{tag} = {value} bei Schritt {step}", RuntimeWarning, stacklevel=4)
            counts = self.metadata["non_finite_metrics"]
            counts[tag] = counts.get(tag, 0) + 1
            return
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)
        self._scalars.writerow((step, tag, value, round(elapsed, 3)))

    def mark_unavailable(self, name: str, reason: str) -> None:
        """Haelt fest, dass eine Kennzahl fehlt und warum."""
        self.metadata["unavailable_metrics"][name] = reason

    # --- Ergebnisse je Instanz --------------------------------------------------

    def save_instance_results(self, results: list[dict], split: str = "test", step: int | None = None,
                              replace: bool = False) -> Path:
        """Haengt Zeilen an <split>_instances.csv an. Kommen neue Spalten dazu, waechst die Datei mit.
        replace: Die Zeilen ersetzen den bisherigen Inhalt, z. B. die Testauswertung eines fortgesetzten Laufs."""
        path = self.results_dir / f"{split}_instances.csv"
        rows = [{"run_id": self.run_id, "model": self.metadata.get("model"), "seed": self.metadata.get("seed"),
                 "step": step, **row} for row in results]
        if replace:
            path.unlink(missing_ok=True)
        columns = list(INSTANCE_COLUMNS)
        old_rows: list[dict] = []
        if path.exists():
            columns, old_rows = _read_rows(path)
        known = set(columns)
        added = [key for row in rows for key in row if key not in known and not (known.add(key))]
        if added or not path.exists():
            _write_rows(path, columns + added, old_rows + rows)
        else:
            with path.open("a", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=columns, restval="").writerows(rows)
        return path

    # --- Gantt-Diagramme --------------------------------------------------------

    def log_gantt(self, schedule: list[dict], instance_id: str, step: int, instance: dict | None = None,
                  waits: list[dict] | None = None) -> Path | None:
        """Speichert PNG und PDF und schreibt das Bild nach TensorBoard. Gibt den Pfad ohne Endung zurueck."""
        tag = f"gantt/{instance_id}"
        if self.config is None or instance is None or not schedule:
            self.mark_unavailable(tag, "Konfiguration, Instanz oder Plan fehlt")
            return None
        import matplotlib.pyplot as plt

        from monitoring.gantt_plotter import save_gantt

        stem = self.figures_dir / "gantt" / f"{instance_id.replace('/', '_')}_step{step}"
        figure = save_gantt(self.config, instance, schedule, f"{self.run_id}, Schritt {step}", stem, waits)
        if self._writer is not None and self._gantt_to_tensorboard:
            self._writer.add_figure(tag, figure, step, close=False)
        plt.close(figure)
        return stem

    # --- Verwaltung -------------------------------------------------------------

    def elapsed_seconds(self) -> float:
        return self._runtime_before + time.perf_counter() - self._started

    def update_metadata(self, **fields) -> None:
        self.metadata.update(fields)
        self._write_metadata()

    def _write_metadata(self) -> None:
        self.metadata["runtime_seconds"] = round(self.elapsed_seconds(), 3)
        _write_json(self._metadata_path, self.metadata)

    def flush(self) -> None:
        self._scalars_file.flush()
        if self._writer is not None:
            self._writer.flush()
        self._write_metadata()

    def close(self, status: str = "finished") -> None:
        if self._closed:
            return
        self.metadata["status"] = status
        self.flush()
        self._scalars_file.close()
        if self._writer is not None:
            self._writer.close()
        self._closed = True

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, kind, value, traceback) -> None:
        self.close("failed" if kind is not None else "finished")


def _as_array(values):
    import numpy

    return numpy.asarray(values, dtype=float)
