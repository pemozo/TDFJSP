"""Kennzahlen zusammenfassen: laufend ueber ein Logging-Intervall und je Datensatz aus den Instanzzeilen.

Mittel von Mitteln waeren hier falsch: Das letzte Minibatch einer Epoche ist kleiner, und die Klassen haben
verschieden viele Jobs. Deshalb tragen Minibatches ihre Groesse als Gewicht, und Anteile ueber Jobs
werden aus den Summen gebildet.
"""
from __future__ import annotations

import math
from typing import Iterable

from monitoring.schedule_metrics import PLAN_METRIC_NAMES


class MetricsAggregator:
    """Gewichtetes Mittel und Standardabweichung je Kennzahl seit dem letzten reset."""

    def __init__(self):
        self._weight: dict[str, float] = {}
        self._sum: dict[str, float] = {}
        self._squares: dict[str, float] = {}

    def add(self, metrics: dict[str, float | None], weight: float = 1.0) -> None:
        """None bedeutet: Kennzahl nicht verfuegbar, sie zaehlt nicht mit."""
        for name, value in metrics.items():
            if value is None:
                continue
            value = float(value)
            self._weight[name] = self._weight.get(name, 0.0) + weight
            self._sum[name] = self._sum.get(name, 0.0) + weight * value
            self._squares[name] = self._squares.get(name, 0.0) + weight * value * value

    def add_values(self, name: str, values: Iterable[float]) -> None:
        """Jeder Wert zaehlt einzeln, z. B. eine Episode."""
        for value in values:
            self.add({name: value})

    def means(self) -> dict[str, float]:
        return {name: self._sum[name] / self._weight[name] for name in self._sum}

    def stds(self) -> dict[str, float]:
        """Standardabweichung der Grundgesamtheit, also ueber alle Episoden des Intervalls."""
        result = {}
        for name, mean in self.means().items():
            variance = self._squares[name] / self._weight[name] - mean * mean
            result[name] = math.sqrt(max(variance, 0.0)) if math.isfinite(variance) else variance
        return result

    def reset(self) -> None:
        self._weight.clear()
        self._sum.clear()
        self._squares.clear()

    def __bool__(self) -> bool:
        return bool(self._sum)


def _mean(rows: list[dict], key: str) -> float | None:
    """Mittel ueber die Zeilen mit einem Wert; None, wenn keine einen hat."""
    values = [row[key] for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _total(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return sum(values) if values else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def summarize_instances(rows: list[dict]) -> dict[str, float | None]:
    """Kennzahlen eines Datensatzes aus den Zeilen je Instanz. window_compliance = Anteil puenktlicher
    Jobs ueber alle Jobs aller Instanzen (das Modell hat Deadlines, keine Abholfenster)."""
    if not rows:
        return {}
    late_share = _ratio(_total(rows, "late_jobs"), _total(rows, "jobs"))
    return {"reward_mean": _mean(rows, "reward"),
            "makespan_mean": _mean(rows, "makespan"),
            "tardiness_mean": _mean(rows, "total_tardiness"),
            "window_compliance": None if late_share is None else 1.0 - late_share,
            "setup_time_total_mean": _mean(rows, "setup_time_total"),
            "processing_time_total_mean": _mean(rows, "processing_time_total"),
            "feasibility_rate": sum(bool(row.get("feasible")) for row in rows) / len(rows),
            "inference_time_mean": _mean(rows, "inference_time"),
            **{f"{name}_mean": _mean(rows, name) for name in PLAN_METRIC_NAMES}}


def summarize_temporal(rows: list[dict]) -> dict[str, float | None]:
    """Zeitliche Kennzahlen eines Datensatzes. Freiwilliger und unvermeidbarer Leerlauf bleiben getrennt."""
    if not rows:
        return {}
    summary = {"wait_actions_mean": _mean(rows, "wait_actions"),
               "wait_share": _ratio(_total(rows, "wait_actions"), _total(rows, "decisions")),
               "voluntary_idle_time_mean": _mean(rows, "voluntary_idle_time"),
               "unavoidable_idle_time_mean": _mean(rows, "unavoidable_idle_time"),
               "processing_time_total_mean": _mean(rows, "processing_time_total"),
               "setup_time_total_mean": _mean(rows, "setup_time_total"),
               "mean_processing_factor": _mean(rows, "mean_processing_factor"),
               "mean_setup_factor": _mean(rows, "mean_setup_factor"),
               "temporal_variability_mean": _mean(rows, "temporal_variability")}
    summary.update({f"{name}_mean": _mean(rows, name) for name in PLAN_METRIC_NAMES})
    for key in dict.fromkeys(key for row in rows for key in row if key.startswith("start_share_")):
        summary[f"start_share/{key.removeprefix('start_share_')}"] = _mean(rows, key)
    return summary
