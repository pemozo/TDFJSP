"""Vergleichsdiagramme aus den gespeicherten Rohdaten; unabhaengig vom laufenden Training.

Aufruf (aus dem Projektordner):
    python -m monitoring.comparison_plotter
    python -m monitoring.comparison_plotter --runs experiments/results/A3_zeit_warten_ppo_seed0 experiments/results/A1_zeit_warten_ppo_seed0
    python -m monitoring.comparison_plotter --baselines results/baselines.csv --static "EDD+ECT statisch"

Jedes Diagramm wird nur aus vorhandenen Werten gezeichnet. Fehlt die Grundlage, entsteht kein Bild, sondern eine
Zeile mit dem Grund in status.csv; nichts wird geschaetzt.

    01 Validierungs-Reward ueber die Schritte        06 Gewinn gegen numerische Zeitabhaengigkeit
    02 Makespan ueber die Schritte                   07 WAIT-Anteil und freiwilliger Leerlauf
    03 Verspaetung und Puenktlichkeit ueber die      08 Rechenzeit gegen Instanzgroesse
       Schritte                                      09 Verteilungen ueber die Testinstanzen
    04 Varianten auf denselben Testinstanzen         10 Mittel und Streuung ueber die Seeds
    05 Reference Gap Makespan

Reference Gap: Abstand zum besten bekannten Makespan je Instanz, aus allen verglichenen Verfahren und dem
Referenz-Plan der Instanz. Dieser Wert ist nicht als optimal bewiesen.
Gewinne vergleichen nur Plaene, die mit den echten Dauern entstanden sind; Laeufe in einer statischen Umgebung
zaehlen nicht.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from generator import BASELINE, GRID, INK, JOB_COLORS, MUTED_INK, SURFACE
from schedule_validator import DEFAULT_CONFIG, load_config, resolve_path

_TIME_AWARE = "time_aware"
_MAX_SERIES = len(JOB_COLORS)
_SERIES_NOTE = "Hoechstens {} Modelle lassen sich sicher unterscheiden, weggelassen: {}"
# Fehlt eine Spalte, etwa wenn nur Regeln vorliegen, ist sie leer statt ein Fehler
_TEST_COLUMNS = ("instance_size", "temporal_variability", "wait_actions", "decisions", "voluntary_idle_time",
                 "reference_makespan", "inference_batch_size", "inference_time")


class Unavailable(Exception):
    """Fuer dieses Diagramm fehlt die Grundlage."""


# --- Daten -----------------------------------------------------------------------


def load_runs(folders: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Kurven (scalars.csv), Testzeilen (test_instances.csv) und Metadaten (run.json) mehrerer Laeufe."""
    scalars, tests, metadata = [], [], []
    for folder in folders:
        info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        metadata.append(info)
        if (folder / "scalars.csv").exists():
            frame = pd.read_csv(folder / "scalars.csv")
            scalars.append(frame.assign(run_id=info["run_id"], model=info.get("model"), seed=info.get("seed")))
        if (folder / "test_instances.csv").exists():
            tests.append(pd.read_csv(folder / "test_instances.csv"))
    return (pd.concat(scalars, ignore_index=True) if scalars else pd.DataFrame(),
            pd.concat(tests, ignore_index=True) if tests else pd.DataFrame(), metadata)


def load_baselines(path: Path, tests: pd.DataFrame) -> pd.DataFrame:
    """results/baselines.csv im Schema der Testzeilen. Die Regeln laufen mit den echten Dauern; statisch
    heisst dort nur, dass der Plan ohne Zeitfaktoren entsteht. Kennzahlen, die die Tabelle nicht hat, bleiben leer."""
    rules = pd.read_csv(path)
    frame = pd.DataFrame({"run_id": rules["method"], "model": rules["method"], "seed": 0,
                          "instance_id": rules["class"] + "/" + rules["instance"].str.removesuffix(".json"),
                          "instance_class": rules["class"], "instance_seed": rules["seed"], "jobs": rules["jobs"],
                          "makespan": rules["makespan"], "total_tardiness": rules["total_tardiness"],
                          "late_jobs": rules["late_jobs"], "window_compliance": 1 - rules["late_share"],
                          "feasible": rules["feasible"].astype(bool), "reference_makespan": rules["reference_makespan"],
                          "inference_time": rules["runtime_s"], "inference_batch_size": 1, "env_view": _TIME_AWARE})
    if not tests.empty:
        known = tests.drop_duplicates("instance_id").set_index("instance_id")
        shared = frame["instance_id"].isin(known.index)
        different = frame.loc[shared, "instance_seed"].to_numpy() != known.loc[frame.loc[shared, "instance_id"],
                                                                               "instance_seed"].to_numpy()
        if different.any():
            raise ValueError(f"{path} passt nicht zu den Testinstanzen der Laeufe (andere Seeds); neu berechnen mit "
                             "python run_experiments.py")
        for column in ("instance_size", "temporal_variability"):
            frame[column] = frame["instance_id"].map(known[column])
    return frame


def common_instances(tests: pd.DataFrame) -> pd.DataFrame:
    """Nur Instanzen, die alle Modelle haben; sonst vergleicht ein Mittel verschiedene Instanzen."""
    if tests.empty:
        raise Unavailable("keine Testergebnisse (test_instances.csv)")
    sets = [set(group["instance_id"]) for _, group in tests.groupby("model")]
    common = set.intersection(*sets)
    if not common:
        raise Unavailable("die Modelle haben keine gemeinsamen Testinstanzen")
    return tests[tests["instance_id"].isin(common)]


def real_duration_rows(tests: pd.DataFrame) -> pd.DataFrame:
    """Nur Zeilen aus der Umgebung mit den echten Dauern."""
    if tests.empty:
        raise Unavailable("keine Testergebnisse (test_instances.csv)")
    kept = tests[tests["env_view"] == _TIME_AWARE]
    if kept.empty:
        raise Unavailable("alle Laeufe stammen aus einer statischen Umgebung")
    return kept


def reference_gap(tests: pd.DataFrame) -> pd.DataFrame:
    """Reference Gap in % je Zeile, nur auf ausfuehrbaren Plaenen."""
    rows = tests[tests["feasible"].astype(bool)].copy()
    if rows.empty:
        raise Unavailable("keine ausfuehrbaren Plaene")
    best = rows.groupby("instance_id")["makespan"].transform("min")
    if "reference_makespan" in rows:
        best = best.where(rows["reference_makespan"].isna() | (best <= rows["reference_makespan"]),
                          rows["reference_makespan"])
    rows["reference_gap"] = (rows["makespan"] - best) / best * 100
    return rows


def temporal_gains(tests: pd.DataFrame, static_model: str) -> pd.DataFrame:
    """temporal_gain je Instanz und Modell gegen static_model, auf den gemeinsamen ausfuehrbaren Plaenen."""
    rows = real_duration_rows(tests)
    rows = rows[rows["feasible"].astype(bool)]
    static = rows[rows["model"] == static_model]
    if static.empty:
        raise Unavailable(f"statische Referenz '{static_model}' fehlt")
    if not static["seed"].nunique() == 1:
        static = static.groupby("instance_id", as_index=False)["makespan"].mean()
    static = static.set_index("instance_id")["makespan"]
    others = rows[rows["model"] != static_model].copy()
    others = others[others["instance_id"].isin(static.index)]
    if others.empty:
        raise Unavailable("kein Modell teilt Instanzen mit der statischen Referenz")
    reference = others["instance_id"].map(static)
    others["temporal_gain"] = (reference - others["makespan"]) / reference * 100
    return others


# --- Zeichnen ---------------------------------------------------------------------


class Canvas:
    """Gemeinsamer Stil, feste Farbe je Modell und Speichern als PNG und PDF."""

    def __init__(self, models: list[str], out: Path):
        self.colors = {model: JOB_COLORS[i] for i, model in enumerate(models[:_MAX_SERIES])}
        self.out = out
        out.mkdir(parents=True, exist_ok=True)

    def figure(self, panels: int = 1, width: float = 7.5, height: float = 4.2):
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, panels, figsize=(width * panels if panels == 1 else 5.2 * panels, height),
                                    squeeze=False)
        figure.patch.set_facecolor(SURFACE)
        for ax in axes[0]:
            ax.set_facecolor(SURFACE)
            ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(colors=MUTED_INK, labelsize=8, length=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(BASELINE)
        return figure, axes[0]

    def save(self, figure, name: str, title: str) -> Path:
        import matplotlib.pyplot as plt

        figure.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
        figure.tight_layout()
        for suffix in (".png", ".pdf"):
            figure.savefig(self.out / f"{name}{suffix}", dpi=150, facecolor=SURFACE)
        plt.close(figure)
        return self.out / f"{name}.png"

    def known(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame[frame["model"].isin(self.colors)]

    def order(self, frame: pd.DataFrame) -> list[str]:
        return [model for model in self.colors if model in set(frame["model"])]


def _label(ax, title: str, y_label: str | None = None, x_label: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    if y_label:
        ax.set_ylabel(y_label, fontsize=8, color=MUTED_INK)
    if x_label:
        ax.set_xlabel(x_label, fontsize=8, color=MUTED_INK)


def _legend(ax) -> None:
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)


def _curves(ax, canvas: Canvas, scalars: pd.DataFrame, tag: str) -> None:
    """Mittel je Modell ueber die Seeds; Band = Standardabweichung. Laeufe werden je Validierung ausgerichtet,
    denn die Schrittzahl einer Iteration haengt von den Episoden ab. x = mittlerer Schritt."""
    rows = canvas.known(scalars[scalars["tag"] == tag]).copy()
    if rows.empty:
        raise Unavailable(f"{tag} fehlt in allen Laeufen")
    rows["position"] = rows.groupby("run_id").cumcount()
    for model in canvas.order(rows):
        grouped = rows[rows["model"] == model].groupby("position").agg(
            step=("step", "mean"), mean=("value", "mean"), std=("value", "std"), seeds=("value", "count"))
        color = canvas.colors[model]
        ax.plot(grouped["step"], grouped["mean"], color=color, linewidth=1.6, marker="o", markersize=4, label=model)
        if (grouped["seeds"] > 1).any():
            ax.fill_between(grouped["step"], grouped["mean"] - grouped["std"].fillna(0),
                            grouped["mean"] + grouped["std"].fillna(0), color=color, alpha=0.18, linewidth=0)
    _legend(ax)


def plot_curve(canvas: Canvas, scalars: pd.DataFrame, name: str, tag: str, title: str, y_label: str) -> Path:
    if scalars.empty:
        raise Unavailable("keine Kurven (scalars.csv)")
    figure, axes = canvas.figure()
    _curves(axes[0], canvas, scalars, tag)
    _label(axes[0], "Mittel ueber die Seeds, Band: Standardabweichung", y_label, "Optimizer-Schritte")
    return canvas.save(figure, name, title)


def plot_tardiness_and_compliance(canvas: Canvas, scalars: pd.DataFrame) -> Path:
    if scalars.empty:
        raise Unavailable("keine Kurven (scalars.csv)")
    figure, axes = canvas.figure(2)
    _curves(axes[0], canvas, scalars, "val/tardiness_mean")
    _label(axes[0], "Summe der Verspaetungen je Instanz", "Minuten", "Optimizer-Schritte")
    _curves(axes[1], canvas, scalars, "val/window_compliance")
    _label(axes[1], "Anteil puenktlicher Jobs", "Anteil", "Optimizer-Schritte")
    return canvas.save(figure, "03_val_tardiness_compliance", "Validierung: Verspaetung und Puenktlichkeit")


def _horizontal(ax, canvas: Canvas, values: pd.Series, formatter: str = "{:.1f}") -> None:
    """Ein Balken je Modell mit dem Wert daneben."""
    models = [model for model in canvas.colors if model in values.index][::-1]
    ax.barh(models, [values[model] for model in models], color=[canvas.colors[model] for model in models],
            height=0.55, zorder=2)
    for position, model in enumerate(models):
        ax.text(values[model], position, " " + formatter.format(values[model]), va="center", fontsize=8, color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.18)


def plot_variants(canvas: Canvas, tests: pd.DataFrame) -> Path:
    rows = canvas.known(common_instances(real_duration_rows(tests)))
    if rows.empty:
        raise Unavailable("keine Testergebnisse")
    grouped = rows.groupby("model")
    table = pd.DataFrame({"instances": grouped["instance_id"].nunique(), "seeds": grouped["seed"].nunique(),
                          "tardiness_mean": grouped["total_tardiness"].mean(),
                          "window_compliance": 1 - grouped["late_jobs"].sum() / grouped["jobs"].sum(),
                          "makespan_mean": grouped["makespan"].mean(),
                          "feasibility_rate": grouped["feasible"].mean()})
    table.to_csv(canvas.out / "04_variants.csv")
    figure, axes = canvas.figure(3, height=max(2.6, 0.5 * len(table) + 1.4))
    _horizontal(axes[0], canvas, table["tardiness_mean"])
    _label(axes[0], "Summe der Verspaetungen (Minuten)")
    _horizontal(axes[1], canvas, 100 * table["window_compliance"])
    axes[1].set_yticklabels([])
    _label(axes[1], "Puenktliche Jobs (%)")
    _horizontal(axes[2], canvas, table["makespan_mean"], "{:.0f}")
    axes[2].set_yticklabels([])
    _label(axes[2], "Makespan (Minuten)")
    return canvas.save(figure, "04_variants", f"Varianten auf denselben {int(table['instances'].iloc[0])} Testinstanzen")


def plot_reference_gap(canvas: Canvas, tests: pd.DataFrame) -> Path:
    rows = canvas.known(reference_gap(common_instances(real_duration_rows(tests))))
    if rows.empty:
        raise Unavailable("keine ausfuehrbaren Plaene")
    values = rows.groupby("model")["reference_gap"].mean()
    values.to_csv(canvas.out / "05_reference_gap.csv")
    figure, axes = canvas.figure(height=max(2.6, 0.5 * len(values) + 1.4))
    _horizontal(axes[0], canvas, values)
    _label(axes[0], "Bezug: bester bekannter Makespan je Instanz, nicht als optimal bewiesen", x_label="Gap (%)")
    return canvas.save(figure, "05_reference_gap", "Reference Gap Makespan")


def plot_gain_vs_variability(canvas: Canvas, tests: pd.DataFrame, static_model: str) -> Path:
    gains = temporal_gains(tests, static_model)
    gains = canvas.known(gains.dropna(subset=["temporal_variability", "temporal_gain"]))
    if gains.empty:
        raise Unavailable("temporal_variability fehlt")
    gains[["model", "seed", "instance_id", "temporal_variability", "temporal_gain"]].to_csv(
        canvas.out / "06_gain_vs_variability.csv", index=False)
    figure, axes = canvas.figure()
    for model in canvas.order(gains):
        part = gains[gains["model"] == model]
        axes[0].scatter(part["temporal_variability"], part["temporal_gain"], s=22, color=canvas.colors[model],
                        edgecolors=SURFACE, linewidths=0.5, label=model, zorder=2)
    axes[0].axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
    _legend(axes[0])
    _label(axes[0], f"Bezug: {static_model}; je Punkt eine Instanz. Numerische Streuung, nicht Entscheidungsrelevanz",
           "temporal_gain (%)", "temporal_variability")
    return canvas.save(figure, "06_gain_vs_variability", "Makespan-Gewinn gegen Zeitabhaengigkeit der Instanz")


def plot_waiting(canvas: Canvas, tests: pd.DataFrame) -> Path:
    rows = canvas.known(real_duration_rows(tests)).dropna(subset=["wait_actions", "decisions"])
    if rows.empty:
        raise Unavailable("WAIT-Kennzahlen fehlen (Regeln messen sie nicht)")
    grouped = rows.groupby("model")
    figure, axes = canvas.figure(2, height=max(2.6, 0.5 * grouped.ngroups + 1.4))
    _horizontal(axes[0], canvas, 100 * grouped["wait_actions"].sum() / grouped["decisions"].sum())
    _label(axes[0], "WAIT-Anteil an den Entscheidungen (%)")
    _horizontal(axes[1], canvas, grouped["voluntary_idle_time"].mean())
    axes[1].set_yticklabels([])
    _label(axes[1], "Freiwilliger Leerlauf (Ressourcen-Minuten je Instanz)")
    return canvas.save(figure, "07_waiting", "Freiwilliges Warten")


def plot_inference(canvas: Canvas, tests: pd.DataFrame) -> Path:
    rows = canvas.known(tests.dropna(subset=["instance_size", "inference_time"])) if not tests.empty else tests
    if rows.empty:
        raise Unavailable("Rechenzeit oder Instanzgroesse fehlt")
    if rows["inference_batch_size"].nunique() > 1:
        rows = rows[rows["inference_batch_size"] == rows["inference_batch_size"].min()]
    figure, axes = canvas.figure()
    for model in canvas.order(rows):
        means = rows[rows["model"] == model].groupby("instance_size")["inference_time"].mean()
        axes[0].plot(means.index, means.to_numpy(), color=canvas.colors[model], linewidth=1.6, marker="o",
                     markersize=4, label=model)
    _legend(axes[0])
    _label(axes[0], f"Batch-Groesse {int(rows['inference_batch_size'].iloc[0])}; Regeln einschliesslich Bewertung",
           "Sekunden je Instanz", "Operationen je Instanz")
    return canvas.save(figure, "08_inference_time", "Rechenzeit gegen Instanzgroesse")


def plot_distributions(canvas: Canvas, tests: pd.DataFrame) -> Path:
    rows = canvas.known(common_instances(real_duration_rows(tests)))
    if rows.empty:
        raise Unavailable("keine Testergebnisse")
    models = canvas.order(rows)
    figure, axes = canvas.figure(2, height=max(3.0, 0.5 * len(models) + 1.6))
    for ax, column, title in ((axes[0], "total_tardiness", "Summe der Verspaetungen (Minuten)"),
                              (axes[1], "makespan", "Makespan (Minuten)")):
        data = [rows.loc[rows["model"] == model, column].to_numpy() for model in models[::-1]]
        boxes = ax.boxplot(data, orientation="horizontal", patch_artist=True, widths=0.55,
                           medianprops={"color": INK, "linewidth": 1.2},
                           whiskerprops={"color": MUTED_INK}, capprops={"color": MUTED_INK},
                           flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": MUTED_INK,
                                       "markeredgewidth": 0})
        for box, model in zip(boxes["boxes"], models[::-1]):
            box.set(facecolor=canvas.colors[model], edgecolor=SURFACE, alpha=0.85)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        _label(ax, title)
    axes[0].set_yticks(range(1, len(models) + 1), models[::-1])
    axes[1].set_yticks(range(1, len(models) + 1), [""] * len(models))
    return canvas.save(figure, "09_distributions", f"Verteilung ueber {rows['instance_id'].nunique()} Testinstanzen")


def plot_seeds(canvas: Canvas, tests: pd.DataFrame) -> Path:
    rows = canvas.known(common_instances(real_duration_rows(tests)))
    if rows.empty:
        raise Unavailable("keine Testergebnisse")
    per_seed = rows.groupby(["model", "seed"]).agg(tardiness=("total_tardiness", "mean"),
                                                   makespan=("makespan", "mean")).reset_index()
    stats = per_seed.groupby("model").agg(seeds=("seed", "nunique"), tardiness_mean=("tardiness", "mean"),
                                          tardiness_std=("tardiness", "std"), makespan_mean=("makespan", "mean"),
                                          makespan_std=("makespan", "std"))
    stats.to_csv(canvas.out / "10_seeds.csv")
    if not (stats["seeds"] > 1).any():
        raise Unavailable("jedes Modell hat nur einen Seed; ohne weitere Seeds gibt es keine Streuung")
    models = [model for model in canvas.colors if model in stats.index][::-1]
    figure, axes = canvas.figure(2, height=max(2.6, 0.5 * len(models) + 1.4))
    for ax, key, title in ((axes[0], "tardiness", "Summe der Verspaetungen (Minuten)"),
                           (axes[1], "makespan", "Makespan (Minuten)")):
        for position, model in enumerate(models):
            ax.errorbar(stats.loc[model, f"{key}_mean"], position, xerr=stats[f"{key}_std"].fillna(0)[model], fmt="o",
                        color=canvas.colors[model], markersize=6, capsize=3, zorder=2)
        ax.set_yticks(range(len(models)), models if ax is axes[0] else [""] * len(models))
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        ax.margins(y=0.2)
        _label(ax, title)
    return canvas.save(figure, "10_seeds", "Mittel und Standardabweichung ueber die Seeds")


def make_all(scalars: pd.DataFrame, tests: pd.DataFrame, out: Path, static_model: str) -> list[dict]:
    """Alle Diagramme; Rueckgabe: je Diagramm erzeugt oder der Grund, warum nicht."""
    if not tests.empty:
        tests = tests.reindex(columns=[*tests.columns, *(c for c in _TEST_COLUMNS if c not in tests.columns)])
    models = sorted(set(tests["model"]) if not tests.empty else set())
    models += sorted(set(scalars["model"]) - set(models)) if not scalars.empty else []
    canvas = Canvas(models, out)
    plots = {
        "01_val_reward": lambda: plot_curve(canvas, scalars, "01_val_reward", "val/reward_mean",
                                            "Validierung: Reward", "Reward"),
        "02_val_makespan": lambda: plot_curve(canvas, scalars, "02_val_makespan", "val/makespan_mean",
                                              "Validierung: Makespan", "Minuten"),
        "03_val_tardiness_compliance": lambda: plot_tardiness_and_compliance(canvas, scalars),
        "04_variants": lambda: plot_variants(canvas, tests),
        "05_reference_gap": lambda: plot_reference_gap(canvas, tests),
        "06_gain_vs_variability": lambda: plot_gain_vs_variability(canvas, tests, static_model),
        "07_waiting": lambda: plot_waiting(canvas, tests),
        "08_inference_time": lambda: plot_inference(canvas, tests),
        "09_distributions": lambda: plot_distributions(canvas, tests),
        "10_seeds": lambda: plot_seeds(canvas, tests),
    }
    status = []
    for name, plot in plots.items():
        try:
            status.append({"plot": name, "status": "erzeugt", "detail": plot().name})
        except Unavailable as reason:
            status.append({"plot": name, "status": "nicht verfuegbar", "detail": str(reason)})
    if len(models) > _MAX_SERIES:
        status.append({"plot": "alle", "status": "gekuerzt",
                       "detail": _SERIES_NOTE.format(_MAX_SERIES, ", ".join(models[_MAX_SERIES:]))})
    pd.DataFrame(status).to_csv(out / "status.csv", index=False)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Vergleichsdiagramme aus gespeicherten Laeufen")
    parser.add_argument("--runs", nargs="*", type=Path, default=None,
                        help="Ordner in <monitoring.folder>/results (Standard: alle)")
    parser.add_argument("--baselines", type=Path, default=None,
                        help="Tabelle der Regeln (Standard: experiments.results_file, falls vorhanden)")
    parser.add_argument("--no-baselines", action="store_true", help="Regeln nicht mit vergleichen")
    parser.add_argument("--static", default="EDD+ECT statisch", help="Modell, auf das sich temporal_gain bezieht")
    parser.add_argument("--out", type=Path, default=None,
                        help="Zielordner (Standard: <monitoring.folder>/figures/comparison)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Einstellungen (Standard: problem_config.json)")
    args = parser.parse_args()

    config = load_config(args.config)
    root = resolve_path(config["monitoring"]["folder"], args.config)
    folders = args.runs
    if folders is None:
        folders = sorted(path for path in (root / "results").glob("*") if (path / "run.json").exists())
    scalars, tests, _ = load_runs(folders)
    baselines = args.baselines or resolve_path(config["experiments"]["results_file"], args.config)
    if not args.no_baselines and baselines.exists():
        rules = load_baselines(baselines, tests)
        if not tests.empty:     # nur die Instanzen, die auch die Laeufe haben
            rules = rules[rules["instance_id"].isin(tests["instance_id"])]
        tests = pd.concat([tests, rules], ignore_index=True)
    out = args.out or root / "figures" / "comparison"
    for entry in make_all(scalars, tests, out, args.static):
        print(f"{entry['plot']:<30} {entry['status']:<18} {entry['detail']}")
    print(f"\nOrdner: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
