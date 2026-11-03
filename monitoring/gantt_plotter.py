"""Gantt-Diagramme fuer das Monitoring: Zeichnung aus generator.plot_schedule, gespeichert als PNG und PDF."""
from __future__ import annotations

from pathlib import Path

from evaluation import makespan
from generator import plot_schedule
from schedule_validator import horizon
from simulator import execution_instance


def save_gantt(config: dict, instance: dict, schedule: list[dict], title: str, stem: Path,
               waits: list[dict] | None = None):
    """Speichert stem.png und stem.pdf; gibt die Figure zurueck (der Aufrufer schliesst sie).
    Die Achse zeigt nur dann den Zusatztag, wenn der Plan ihn wirklich braucht."""
    fits = makespan(schedule) <= horizon(config, instance)
    shown = instance if fits else execution_instance(config, instance)
    figure = plot_schedule(config, shown, schedule, title, waits=waits, show_times=True)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        figure.savefig(stem.with_suffix(suffix), dpi=150, facecolor=figure.get_facecolor())
    return figure
