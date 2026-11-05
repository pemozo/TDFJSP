# Instanzgenerator für ein zeitabhängiges flexibles Job-Shop-Problem

Dieses Projekt erzeugt **lösbare Testinstanzen** für ein flexibles Job-Shop-Scheduling-Problem (FJSP) mit **zeitabhängigen Bearbeitungszeiten**, **reihenfolgeabhängigen Rüstzeiten**, **Pausen**, **Arbeitsende** und **Deadlines**. Ein unabhängiger **Validator** prüft Instanzen und beliebige Schedules gegen dieselben Regeln, eine **Bewertung** misst Makespan und Verspätungen. Ein **Simulator** führt Pläne mit den echten, zeitabhängigen Dauern aus, und erste **Baselines** (EDD + ECT, EDD + SPT, jeweils statisch und zeitbewusst) lassen sich auf sechs Instanzklassen mit je 100 Testinstanzen auswerten. Eine **RL-Umgebung** und ein **zeitabhängiges HetGNN mit PPO** (`rl/`) stehen bereit; das vollständige Training steht noch aus.

Alle Regeln, das Zeitmodell, die Instanzgrößen und die Einstellungen der Umgebung stehen an **einer Stelle**: in `problem_config.json`. Der Code enthält selbst keine Regeln.

---

## Inhalt

1. [Projektstruktur](#projektstruktur)
2. [Schnellstart](#schnellstart)
3. [Das Planungsproblem](#das-planungsproblem)
4. [Modellannahmen](#modellannahmen)
5. [Was eine Instanz berücksichtigt](#was-eine-instanz-berücksichtigt)
6. [Instanzklassen für die Experimente](#instanzklassen-für-die-experimente)
7. [problem_config.json – Regeln und Zeitmodell](#problem_configjson--regeln-und-zeitmodell)
8. [generator.py – Instanzen erzeugen](#generatorpy--instanzen-erzeugen)
9. [Instanzformat](#instanzformat)
10. [schedule_validator.py – Instanzen und Schedules prüfen](#schedule_validatorpy--instanzen-und-schedules-prüfen)
11. [evaluation.py – Makespan und Verspätungen](#evaluationpy--makespan-und-verspätungen)
12. [Simulator, Baselines und Experimente](#simulator-baselines-und-experimente)
13. [RL-Umgebung](#rl-umgebung)
14. [RL-Modell und Training](#rl-modell-und-training)
15. [Experiment-Tracking und Visualisierung](#experiment-tracking-und-visualisierung)
16. [Typische Abläufe](#typische-abläufe)
17. [Wichtige Hinweise und Grenzen](#wichtige-hinweise-und-grenzen)

---

## Projektstruktur

```
problem_config.json     Alle Einstellungen: Regeln, Zeitmodell, Generator, Instanzklassen,
                        Datensätze, Experimente, Umgebung, Reward, Modell, Training, Monitoring
schedule_validator.py   Wertet die Regeln aus, prüft Config, Instanzen und Schedules
evaluation.py           Kennzahlen: Makespan, Verspätungen
generator.py            Erzeugt Instanzen anhand der Regeln und der Instanzklassen
instances/              Instanzdateien: instances/test/<Klasse>, instances/val/<Klasse>

simulator.py            Ereignisgesteuerte Planung und Ausführung mit echten Dauern
baselines/              Prioritätsregeln
  rules.py                Reihenfolge (EDD), Zuordnung (ECT, SPT)
  dispatcher.py           Plan aus Regel × Dauer-Sicht bauen, dann ausführen
run_experiments.py      Klassen × Instanzen × Verfahren → results/*.csv
results/                Ergebnisse der Experimente

rl/                     RL-Umgebung, Modell und Training (nur Torch)
  env.py                  TDFJSPEnv: reset, step, Maske, Reward, Merkmale
  generators.py           Instanzen on the fly und aus Dateien
  parser.py               Instanz-JSON ↔ Tensoren
  time_model.py           Kalender als Tensoren
  reward.py               Hierarchische Kosten, Schätzung der Fertigstellung
  policies.py             Einfache Policies ohne Netz, rollout
  evaluate.py             Policies und Modelle auf denselben Instanzen wie die Regeln bewerten
  observation.py          Eingaben des Netzes aus dem Zustand
  encoder.py              TemporalHetGNNLayer, HetGNNEncoder
  decoder.py              TimeAwareDecoder, StaticPairDecoder
  policy.py               ActorCritic, Varianten A0–A3, NeuralPolicy
  ppo.py, train.py        PPO und Training
  instance_results.py     Ergebnisse je Instanz: Kennzahlen, Plan, freiwillige Wartezeiten
monitoring/             Experiment-Tracking, unabhängig vom Trainingscode
  experiment_logger.py    TensorBoard, CSV, Metadaten je Lauf
  metrics_aggregator.py   Mittel über Intervalle und Datensätze
  schedule_metrics.py     Zeiten, Faktoren, Leerlauf, Zeitabhängigkeit, Gewinn
  training_monitor.py     Intervalle, Verbindung zur Trainingsschleife
  gantt_plotter.py        Gantt-Diagramme als PNG und PDF
  comparison_plotter.py   Vergleichsdiagramme aus den Rohdaten
experiments/            Ergebnisse der Läufe (entsteht beim Training)
tests/                  Abnahmetests von Umgebung, Modell und Monitoring, Lerntest

requirements.txt        Abhängigkeiten
README.md               Diese Datei
```

Die Abhängigkeiten laufen nur in eine Richtung:

```
problem_config.json ─► schedule_validator.py ─┬─► evaluation.py ──────────────┬─► generator.py
                                              ├─► simulator.py ─┬─────────────┘
                                              │                 └─► baselines/ ─► run_experiments.py
                                              │                 └─► rl/ (nutzt auch generator.py, run_experiments.py)
                                              └─────────────────────────────────► run_experiments.py
```

`schedule_validator.py` hängt von nichts anderem im Projekt ab. Er bleibt die unabhängige Prüfinstanz für **jedes** Ergebnis, auch das eines RL-Agenten. Pfade in der Config gelten relativ zur Config-Datei; die Skripte startet man aus dem Projektordner.

---

## Schnellstart

**Voraussetzungen:** Python 3 (getestet mit 3.12), `pip install -r requirements.txt`. `torch` wird nur für `rl/` gebraucht, `matplotlib` nur für das Gantt-Diagramm (`--show`).

```powershell
# alle Instanzklassen aus der Config erzeugen (Test und Validierung)
python generator.py --classes all --splits test val --processes 10

# Dispatching-Regeln auf allen Klassen auswerten
python run_experiments.py

# RL-Umgebung prüfen und Policies ohne Netz auswerten
python -m unittest tests.test_rl_env
python -m rl.evaluate --policies edd_ect random --classes klein_11

# einzelne Instanzen ohne Klasse: 6 Jobs, 3 Maschinen, 3 Operationen pro Job, 2 Worker
python generator.py --count 5 --jobs 6 --machines 3 --ops 3 --workers 2 --out instances/beispiel

# dasselbe, mit Anzeige des Referenz-Schedules (Tabelle + Gantt-Diagramm)
python generator.py --count 1 --jobs 6 --machines 3 --ops 3 --workers 2 --out instances/beispiel --show

# alle Instanzen im Ordner prüfen (auch in Unterordnern)
python schedule_validator.py instances/beispiel

# einen eigenen Schedule gegen eine Instanz prüfen
python schedule_validator.py instances/beispiel/instance_000.json --schedule mein_plan.json

# Makespan und Verspätungen messen
python evaluation.py instances/beispiel
```

---

## Das Planungsproblem

### Problembeschreibung

**Hintergrund.** In der Sitzmontage für die Automobilindustrie liefern Tier-1-Zulieferer just-in-sequence (JIS): Fertige Sitze müssen zu fest vereinbarten Zeitpunkten zur Abholung bereitstehen. In der Fertigung arbeiten Menschen und Maschinen zusammen. Die Leistung der Menschen schwankt über den Arbeitstag, zum Beispiel durch Ermüdung und Tageszeiteffekte. Deshalb sind Bearbeitungs- und Rüstzeiten nicht konstant, sondern hängen davon ab, **wann** eine Tätigkeit beginnt.

**Problem.** Gegeben sind Jobs, die jeweils aus einer festen Folge von Operationen bestehen. Jede Operation ist **manuell** (an einem Handarbeitsplatz durch einen Worker) oder **automatisch** (auf einer Maschine) und darf nur auf bestimmten Ressourcen laufen, mit ressourcenabhängiger Dauer. Gesucht ist ein Plan, der jeder Operation eine zulässige Ressource und eine Startzeit zuweist, sodass:

- die Operationen eines Jobs in ihrer Reihenfolge laufen,
- jede Ressource zu jedem Zeitpunkt höchstens eine Tätigkeit ausführt,
- vor einer Operation auf einer Maschine die passende Rüstung stattfindet,
- manuelle Tätigkeiten und Rüstungen nicht in der Pause liegen und nichts über das Arbeitsende hinausläuft,
- möglichst jeder Job bis zu seiner Deadline (Abholzeitpunkt) fertig ist.

Dabei hängen die Dauern manueller Operationen und aller Rüstungen von der **Startzeit** ab. Die Deadlines sind standardmäßig **weich**: Ein Plan darf sie überschreiten, und die Verspätungen werden gemessen. So lässt sich vergleichen, wie zuverlässig verschiedene Verfahren die Abholzeitpunkte einhalten.

**Einordnung.** Flexibles Job-Shop-Scheduling-Problem (FJSP) mit
- **partieller Flexibilität** und **maschinenabhängigen Bearbeitungszeiten** (jede Kombination aus Operation und Ressource hat eine eigene Dauer),
- **zeitabhängigen Bearbeitungszeiten** für manuelle Operationen,
- **reihenfolge-, maschinen- und zeitabhängigen Rüstzeiten**,
- **Verfügbarkeitszeitfenstern** (Pause, Arbeitsende),
- **Deadlines** je Job (standardmäßig weich, per Schalter `due_dates_hard` hart),
- nicht-präemptiven Tätigkeiten.

Die **Zielfunktion** legt der Generator nicht fest. Sie wird vom Lösungsverfahren bestimmt, zum Beispiel Makespan, Summe der Verspätungen oder Anzahl verspäteter Jobs. Der Generator liefert die Daten, die Config die Regeln, der Validator die Prüfung auf Zulässigkeit und `evaluation.py` die Kennzahlen.

### Ressourcen

| Ressource | Name | Aufgabe |
|---|---|---|
| Handarbeitsplatz | `H1`, `H2`, … | manuelle Operationen |
| Worker | `W1`, `W2`, … | arbeitet an genau einem festen Handarbeitsplatz (`W1 ↔ H1`, `W2 ↔ H2`, …) |
| Maschine | `M1`, `M2`, … | automatische Operationen |

Weil jeder Worker fest zu einem Handarbeitsplatz gehört, ist `Hi` genau dann belegt, wenn `Wi` arbeitet.

### Jobs und Operationen

- Jeder **Job** besteht aus einer festen Anzahl **Operationen**, die **nacheinander** bearbeitet werden müssen (`J1_O1` vor `J1_O2` vor …).
- Jede Operation ist entweder **manuell** oder **automatisch**. Sie wechseln sich im Schachbrettmuster ab: `J1` beginnt manuell, `J2` automatisch, `J3` manuell, …
- **Manuelle Operationen** dürfen auf **jedem** Handarbeitsplatz laufen und haben dort überall **dieselbe** Basisdauer (20–40 min): Alle Worker sind gleich schnell. Umschaltbar über `generation.manual_same_duration_on_all_stations`.
- **Automatische Operationen** dürfen auf einer oder mehreren Maschinen laufen. Jede Operation hat eine **Basiszeit** (30–60 min), und jede zulässige Maschine weicht davon um bis zu **±20 %** ab (zum Beispiel Basiszeit 45 min: 36–54 min je nach Maschine). Umschaltbar über `generation.automatic_machine_deviation`.
  - Die erste automatische Operation der Instanz darf auf **allen** Maschinen laufen. So ist jede Maschine mindestens einmal zulässig.
  - Jede weitere ist mit 50 % Wahrscheinlichkeit **flexibel** (2 bis alle Maschinen), sonst **fest** an eine Maschine gebunden.
- Der Scheduler entscheidet, **auf welcher** zulässigen Ressource und **wann** eine Operation läuft. Das ist das „flexibel“ im FJSP.

### Rüstzeiten

- Wechselt eine **Maschine** von Operation A zu Operation B, muss sie umgerüstet werden.
- Die Rüstzeit ist **reihenfolgeabhängig** und **asymmetrisch**: A → B kann anders dauern als B → A (Basis 5–15 min).
- Die erste Operation auf einer Maschine braucht keine Rüstung (`initial_setup_minutes = 0`).
- Rüsten belegt **nur die Maschine, keinen Worker**. Es wird von nicht modelliertem Rüstpersonal erledigt.
- An Handarbeitsplätzen wird nicht gerüstet. Das ist in der Config umstellbar.

### Zeitmodell

Ein Arbeitstag geht von **06:00 bis 18:00 (720 Minuten)**. Er ist in Zeitfenster mit einem **Faktor** eingeteilt:

| Zeitfenster | Minuten | Faktor |
|---|---|---|
| 06:00–09:00 | 0–180 | 1.0 |
| 09:00–10:00 | 180–240 | 1.3 |
| 10:00–12:00 | 240–360 | 1.2 |
| 12:00–12:30 | 360–390 | **Pause** |
| 12:30–14:00 | 390–480 | 1.6 |
| 14:00–16:00 | 480–600 | 1.3 |
| 16:00–18:00 | 600–720 | 1.1 |

**Zeitabhängige Dauer:** Bei manuellen Operationen und Rüstungen gilt

```
tatsächliche Dauer = ceil(Basisdauer × Faktor zum Startzeitpunkt)
```

Die Dauer wird **beim Start festgelegt**. Eine manuelle Operation mit 30 Minuten Basisdauer dauert also um 08:00 genau 30 Minuten, um 10:30 36 Minuten und um 13:00 48 Minuten. Automatische Operationen dauern immer ihre Basisdauer.

**Mehrere Arbeitstage:** Die Zeit zählt nur **Arbeitsminuten**, die Nächte sind ausgeblendet. Tag 1 ist Minute 0–720, Tag 2 ist Minute 720–1440 usw. Nachts steht **alles** still, auch die Maschinen.

### Was ist wann erlaubt?

| Tätigkeit | zeitabhängig | in der Pause | über Nacht |
|---|---|---|---|
| manuelle Operation | ja | **nein** | **nein** |
| automatische Operation | nein | **ja** (Maschine läuft allein) | **nein** |
| Rüstung | ja | **nein** | **nein** |

„Nein“ heißt: Die Tätigkeit darf weder dort **starten** noch dort **hineinlaufen**. Eine Unterbrechung gibt es nicht (nicht-präemptiv). Eine manuelle Operation, die um 11:45 starten würde und 40 Minuten dauert, ist also unzulässig.

### Weitere Regeln

Harte Regeln (ein Verstoß macht einen Plan unzulässig):
- Jede Ressource bearbeitet zu jedem Zeitpunkt **höchstens eine Tätigkeit**. Rüsten und Bearbeiten auf derselben Maschine finden nacheinander statt.
- Die Rüstung liegt **vor** der Operation auf derselben Ressource.
- Die Operationen eines Jobs laufen **in ihrer Reihenfolge**. Eine Operation startet erst, wenn ihr Vorgänger fertig ist.

Deadlines:
- **Standard (`due_dates_hard: false`):** Ein Job darf nach `due` fertig werden. Der Plan bleibt zulässig, die Verspätung `max(0, Fertigstellung − due)` wird von `evaluation.py` gemessen.
- **Umschaltbar (`due_dates_hard: true`):** Jede Verspätung macht den Plan unzulässig. Der Schalter ändert den Fingerprint nicht, vorhandene Instanzen bleiben also gültig.

---

## Modellannahmen

Alle Annahmen auf einen Blick. Die Spalte „Config“ nennt den Schlüssel, falls die Annahme in `problem_config.json` umstellbar ist.

### Zeit

| Annahme | Config |
|---|---|
| Zeit in **ganzen Minuten**; tatsächliche Dauern werden auf ganze Minuten **aufgerundet**. | `duration_rounding` |
| Ein Arbeitstag geht von **06:00 bis 18:00** (720 min). Der Horizont umfasst ein oder mehrere Tage. | `day_minutes`, `origin` |
| **Nächte sind ausgeblendet:** Die Zeit zählt nur Arbeitsminuten. Nachts steht **alles** still, auch die Maschinen. | `allowed_overnight` |
| Die **Zeitfaktoren** (1.0 / 1.3 / 1.2 / 1.6 / 1.3 / 1.1) sind an jedem Tag und für alle Worker **gleich**. Sie bilden Tageszeit und Ermüdung pauschal ab. | `day_windows` |
| Der Faktor **zum Startzeitpunkt** gilt für die gesamte Tätigkeit, auch wenn sie in ein anderes Zeitfenster hineinreicht. | `duration_fixed_at_start` |
| Alle Daten sind **deterministisch** und **von Anfang an bekannt**. Es gibt keine zufälligen Abweichungen, Ausfälle oder neu eintreffenden Aufträge. | – |

### Ressourcen

| Annahme | Config |
|---|---|
| Jeder Worker gehört **fest zu einem** Handarbeitsplatz (`W1 ↔ H1`); er wechselt nicht. | `station_worker_mapping` |
| Worker und Maschinen sind **über den ganzen Arbeitstag verfügbar**, Worker außer in der Pause. | – |
| Die **Pause** (12:00–12:30) gilt **für alle Worker gleichzeitig**. | `day_windows` |
| Jede Ressource führt **höchstens eine Tätigkeit gleichzeitig** aus. | `one_task_per_resource_at_a_time` |
| Worker sind **gleich schnell**: Eine manuelle Operation hat auf jedem Handarbeitsplatz dieselbe Basisdauer. Unterschiede entstehen nur durch die Tageszeit. | `generation.manual_same_duration_on_all_stations` |
| Maschinen sind **ähnlich, aber nicht gleich schnell**: Eine automatische Operation hat eine Basiszeit, auf jeder zulässigen Maschine weicht ihre Dauer um bis zu ±20 % davon ab. | `generation.automatic_machine_deviation` |
| Automatische Operationen brauchen **keinen Worker** (Be- und Entladen der Maschine wird nicht modelliert). | – |
| Es gibt **keine hybriden Operationen** (Maschine und Worker gleichzeitig). | – |

### Operationen und Jobs

| Annahme | Config |
|---|---|
| Jede Operation läuft **genau einmal** auf **genau einer** zulässigen Ressource. | – |
| Die Operationen eines Jobs laufen **strikt nacheinander**. Die nächste kann **sofort** nach dem Ende der vorherigen starten: keine Transportzeiten, unbegrenzte Puffer zwischen den Stationen. | `job_operations_in_order` |
| Alle Jobs sind **ab Schichtbeginn** verfügbar (Freigabe 0). | `generation.release` |
| Tätigkeiten werden **nicht unterbrochen** (nicht-präemptiv). Eine Tätigkeit startet nur, wenn sie vollständig in die erlaubte Zeit passt. Es wird nichts abgebrochen. | `non_preemptive` |
| **Manuelle Operationen** sind zeitabhängig, dürfen **nicht in die Pause** und **nicht über das Arbeitsende** laufen. | `time_dependent`, `allowed_in_break`, `allowed_overnight` |
| **Automatische Operationen** haben eine feste Dauer, laufen **durch die Pause** weiter, aber **nicht über das Arbeitsende**. | ebenso |

### Rüsten

| Annahme | Config |
|---|---|
| Gerüstet wird **nur an Maschinen**, nicht an Handarbeitsplätzen. | `setups_on_manual_stations` |
| Rüstzeiten sind **reihenfolgeabhängig**, **asymmetrisch** und **maschinenabhängig**. | – (Daten) |
| Rüstzeiten sind **zeitabhängig** wie manuelle Arbeit: nicht in der Pause, nicht über das Arbeitsende. | `time_dependent`, `allowed_in_break` |
| Rüsten erledigt **separates Rüstpersonal**, das nicht modelliert wird und **unbegrenzt verfügbar** ist: Rüstungen belegen keinen Worker, und mehrere Maschinen können gleichzeitig gerüstet werden. | `setup_requires_worker` |
| Die **erste** Operation auf einer Maschine braucht **keine Rüstung**. | `initial_setup_minutes` |
| Die Rüstung muss **vor** der Operation auf derselben Maschine liegen, nach dem vorherigen Auftrag dort. Sie darf **früher** stattfinden, auch bevor der Job bereitsteht, und zwischen Rüstung und Bearbeitung darf eine Lücke liegen. | `setup_directly_before_operation_on_same_resource` |
| Eine Maschine **bleibt über Nacht gerüstet**: Rüsten am Abend, Bearbeitung am nächsten Morgen ist erlaubt. | – |

Der Referenz-Plan des Generators und alle Baselines nutzen das vorgezogene Rüsten nicht: Dort beginnt die Rüstung erst, wenn Maschine und Job bereit sind, und die Bearbeitung folgt direkt. Für die Prüfung eines Plans ist vorgezogenes Rüsten trotzdem erlaubt.

### Deadlines

| Annahme | Config |
|---|---|
| Jeder Job hat **eine eigene Deadline** (Abholzeitpunkt). Gemeinsame Abholungen mehrerer Jobs gibt es nicht. | – |
| Deadlines sind standardmäßig **weich**; die Verspätung wird gemessen. | `due_dates_hard` |
| Deadlines werden aus einem **zulässigen Referenz-Plan** abgeleitet. Es gibt also immer einen Plan **ohne Verspätung**. | `generation.due_slack` |
| Ein **frühester** Abholzeitpunkt existiert nicht; zu frühe Fertigstellung kostet nichts. | – |

### Planung und Ausführung (Experimente)

| Annahme | wo |
|---|---|
| Ein **statisches** Verfahren rechnet mit Faktor 1.0, kennt aber **Pause und Arbeitsende**. | `simulator.DURATION_VIEWS` |
| Planung und Ausführung dürfen **Zusatztage** nutzen (Standard: einen); Deadlines bleiben gleich, Verspätung wird gemessen. | `simulation.overflow_days` |
| **Rüsten** beginnt, wenn Maschine und Job bereit sind; die Bearbeitung folgt direkt. Für alle Verfahren gleich. | `simulator.py` |
| Die Planung ist **ereignisgesteuert (non-delay)**: Gewählt wird nur unter Operationen und freien Ressourcen, die **jetzt** regelkonform starten können. | `simulator.SimulationState` |
| Ein Plan wird mit den **echten Dauern ausgeführt**; Zuordnung und Reihenfolge je Ressource bleiben, die Uhrzeiten werden neu berechnet. | `simulator.execute_plan` |

Details im Abschnitt [Simulator, Baselines und Experimente](#simulator-baselines-und-experimente).

---

## Was eine Instanz berücksichtigt

| Aspekt | Umsetzung | steht in |
|---|---|---|
| Jobs mit fester Operationsfolge | jeder Job hat `n_ops` Operationen, die nacheinander laufen | Instanz |
| manuelle und automatische Operationen | abwechselnd im Schachbrettmuster | Instanz (`kind`) |
| flexible Zuordnung | manuell: jeder Handarbeitsplatz; automatisch: 1 bis alle Maschinen | Instanz (`base_processing`) |
| Bearbeitungszeiten | automatisch: Basiszeit je Operation (30–60 min), je Maschine ±20 %; manuell: gleiche Basisdauer auf allen Handarbeitsplätzen (20–40 min) | Instanz (`base_processing`) |
| zeitabhängige Bearbeitungszeiten | manuelle Operationen: Basisdauer × Zeitfaktor beim Start | Config (`day_windows`, `time_dependent`) |
| Rüstzeiten | reihenfolgeabhängig, asymmetrisch, je Maschine (Basis 5–15 min) | Instanz (`setups`) |
| zeitabhängige Rüstzeiten | Basis-Rüstzeit × Zeitfaktor beim Start | Config |
| Rüsten ohne Worker | Rüstung belegt nur die Maschine | Config (`setup_requires_worker`) |
| Worker und Handarbeitsplätze | fest 1:1 (`W1 ↔ H1`, …) | Instanz (`station_worker`) |
| Tageszeitfenster | Faktor 1.0 / 1.3 / 1.2 / 1.6 / 1.3 / 1.1 | Config (`day_windows`) |
| Pause | 12:00–12:30; keine manuelle Arbeit, keine Rüstung | Config |
| Arbeitsende | 18:00; nichts läuft über das Tagesende, nachts steht alles still | Config (`allowed_overnight`) |
| mehrere Arbeitstage | optional, Horizont = Tage × 720 Arbeitsminuten | Instanz (`days`) |
| Deadlines (Abholzeitpunkte) | je Job, aus einem zulässigen Referenz-Plan abgeleitet; standardmäßig weich | Instanz (`due`), Config (`due_dates_hard`) |
| Lösbarkeit | ein Plan ohne Verspätung ist garantiert (gespeicherter Referenz-Plan) | Instanz (`reference_schedule`) |

### Bewusst nicht modelliert

Diese Punkte gehören zum Anwendungsfall, sind in den Instanzen aber (noch) nicht abgebildet:

- **Hybride Operationen**, die gleichzeitig eine Maschine und einen Worker brauchen.
- **JIS-Abholungen als Gruppe:** Mehrere Jobs (ein Sitzsatz) mit gemeinsamem Abholzeitpunkt und vorgegebener Reihenfolge. Derzeit hat jeder Job eine eigene Deadline, und es gibt keinen frühesten Abholzeitpunkt.
- **Ermüdung abhängig von der geleisteten Arbeit:** Die Leistung hängt nur von der Tageszeit ab, nicht davon, wie lange ein Worker schon ohne Pause gearbeitet hat.
- **Unterschiedliche Freigabezeitpunkte:** Alle Jobs sind ab Schichtbeginn verfügbar (`release = 0`).
- **Störungen und Unsicherheit:** keine Maschinenausfälle, keine zufälligen Abweichungen der Dauern während der Ausführung.

---

## Instanzklassen für die Experimente

Alle Instanzgrößen stehen im Abschnitt `instance_classes` von `problem_config.json`. Dieselben Klassen gelten für die Dispatching-Regeln und für die RL-Umgebung. Eine neue Größe braucht nur einen Eintrag in der Config und `python generator.py --classes NAME --splits test val`.

Es gibt drei Ressourcenstufen mit je zwei Auslastungen, alle mit **einem Arbeitstag**, 5 Operationen je Job und `due_slack = 0.05`:

| Klasse | Stufe | Jobs | Operationen | Worker | Maschinen | Operationen je Ressource |
|---|---|---|---|---|---|---|
| `klein_11` | Hauptstufe | 11 | 55 | 2 | 4 | 9,2 |
| `mittel_16` | Hauptstufe | 16 | 80 | 3 | 6 | 8,9 |
| `gross_21` | Hauptstufe | 21 | 105 | 4 | 8 | 8,8 |
| `klein_12` | hohe Auslastung | 12 | 60 | 2 | 4 | 10,0 |
| `mittel_18` | hohe Auslastung | 18 | 90 | 3 | 6 | 10,0 |
| `gross_24` | hohe Auslastung | 24 | 120 | 4 | 8 | 10,0 |

### Warum diese Größen

**Kaum verlorene Seeds.** Ein Seed wird übersprungen, wenn der Generator in `schedule_attempts` Versuchen keinen Referenz-Plan im Tag findet. Bis 12, 17 und 23 Jobs passiert das nicht, bei 18 und 24 Jobs in 4 % der Fälle. Ab 13 Jobs (klein) und 19 Jobs (mittel) sind es über 20 %, bei 25 Jobs (groß) etwa 30 %. Solche Klassen enthielten nur Instanzen, die gerade noch passen.

**Nicht zu einfach.** Die Schwierigkeit kommt vor allem aus dem engen Deadline-Puffer. Er kostet keine Seeds, weil die Deadlines erst nach dem Referenz-Plan gesetzt werden. Mehr Jobs machen eine Instanz dagegen nicht zuverlässig schwerer.

**Zeitabhängigkeit wirkt.** Die Referenz-Pläne enden im Median nach 16:00 und laufen damit durch 12:30–14:00 mit Faktor 1.6. Eine deutlich kleinere Instanz enthält das nicht: 6 Jobs × 5 Ops mit 2 Workern und 4 Maschinen sind im Median um 11:25 fertig, nur 3 von 100 reichen über die Pause hinaus.

**Zwei Stufen.** Innerhalb einer Stufe ist die Last je Ressource ähnlich, Unterschiede lassen sich also eher auf die Größe zurückführen. Die Hauptstufe braucht den Zusatztag kaum. Die hohe Auslastung zeigt das Verhalten bei knappem Tag und hängt stärker an der Annahme `simulation.overflow_days`.

### Messwerte

Gemessen an den 100 Testinstanzen je Klasse (`due_slack = 0.05`, `flexible_probability = 0.7`). Die Zeiten gelten für den Referenz-Plan, die letzte Spalte für EDD+ECT:

| Klasse | übersprungene Seeds | Ende Median | Ende spätestens | manuelle Ops mit Faktor > 1 | Läufe mit Zusatztag (statisch / zeitbewusst, von 100) |
|---|---|---|---|---|---|
| `klein_11` | 0 | 16:17 | 17:10 | 58 % | 3 / 3 |
| `mittel_16` | 0 | 16:17 | 17:11 | 58 % | 2 / 2 |
| `gross_21` | 0 | 16:26 | 17:13 | 57 % | 2 / 2 |
| `klein_12` | 0 | 16:59 | 17:43 | 62 % | 21 / 11 |
| `mittel_18` | 4 | 17:16 | 17:51 | 61 % | 26 / 24 |
| `gross_24` | 4 | 17:27 | 18:00 | 62 % | 40 / 28 |

### Erzeugen

```powershell
python generator.py --classes all --splits test val --processes 10
python schedule_validator.py instances
```

Das dauert mit 10 Prozessen etwa 1,5 Minuten und belegt etwa 370 MB. Die Dateien entstehen deterministisch aus Config und Seeds, lassen sich also jederzeit löschen und neu erzeugen. Für Tests und Validierung gelten die Einstellungen aus `data_splits`.

---

## problem_config.json – Regeln und Zeitmodell

Diese Datei steuert das ganze Projekt. Schlüssel, die mit `_` beginnen (`_comment`), sind Kommentare und werden ignoriert. `load_config` prüft die Datei beim Laden und lehnt ungültige Werte mit einer Meldung ab.

| Abschnitt | Inhalt | Geht in den Fingerprint ein? |
|---|---|---|
| `time_model` | Zeitfenster und Faktoren | ja |
| `rules` | Scheduling-Regeln | ja (außer `due_dates_hard`) |
| `generation` | Wie Instanzen erzeugt werden | nein |
| `simulation` | Zusatztage bei der Ausführung | nein |
| `instance_classes` | Alle Instanzgrößen | nein |
| `data_splits` | Trainings-, Validierungs- und Testdaten | nein |
| `experiments` | Regeln, Klassen und Ergebnisdateien der Auswertung | nein |
| `environment` | RL-Umgebung | nein |
| `reward` | Hierarchisches Ziel | nein |
| `model` | Neuronales Netz und Variante | nein |
| `training` | PPO und Trainingslauf | nein |
| `monitoring` | Experiment-Tracking | nein |

### `time_model` – Zeitmodell

| Schlüssel | Wert | Bedeutung |
|---|---|---|
| `unit` | `"minute"` | Zeiteinheit (ganze Minuten) |
| `origin` | `"06:00"` | Uhrzeit von Minute 0 jedes Tages (nur für die Anzeige) |
| `day_minutes` | `720` | Länge eines Arbeitstages |
| `day_windows` | Liste | Zeitfenster eines Tages mit `start`, `end`, `factor`. `factor: null` = Pause. Sie müssen den Tag lückenlos abdecken. |
| `resource_factors` | optional | Eigene Zeitprofile: `manual_stations` mit einer Liste je Handarbeitsplatz (erste Liste H1, zweite H2 usw.) und `setup` für die Rüstung, auf allen Maschinen gleich. Jede Liste hat einen Faktor je Zeitfenster, `null` genau in den Pausen. Fehlt der Schlüssel, gilt für alle `day_windows.factor`. Ist er gesetzt, ist `day_windows.factor` nur noch das Bezugsprofil für die Uhrzeit-Merkmale und für die Schätzung noch nicht zugeordneter manueller Operationen. |
| `duration_rounding` | `"ceil"` | Rundung der zeitabhängigen Dauer (auf ganze Minuten aufrunden) |
| `duration_fixed_at_start` | `true` | Faktor zum Startzeitpunkt gilt für die ganze Tätigkeit |

### `rules` – Scheduling-Regeln

| Schlüssel | Wert | Bedeutung |
|---|---|---|
| `non_preemptive` | `true` | Tätigkeiten werden nicht unterbrochen |
| `time_dependent` | je Art | Wird der Zeitfaktor angewendet? |
| `allowed_in_break` | je Art | Darf die Tätigkeit die Pause berühren? |
| `allowed_overnight` | je Art | Darf die Tätigkeit über ein Tagesende laufen? |
| `setup_requires_worker` | `false` | Rüsten belegt keinen Worker |
| `setups_on_manual_stations` | `false` | An Handarbeitsplätzen wird nicht gerüstet |
| `initial_setup_minutes` | `0` | Rüstzeit vor der ersten Operation einer Ressource |
| `station_worker_mapping` | `"one_to_one"` | Jeder Worker hat genau einen Handarbeitsplatz |
| `one_task_per_resource_at_a_time` | `true` | Keine Doppelbelegung |
| `setup_directly_before_operation_on_same_resource` | `true` | Rüstung vor der Operation auf derselben Ressource |
| `job_operations_in_order` | `true` | Reihenfolge innerhalb eines Jobs |
| `due_dates_hard` | `false` | `false`: Verspätung erlaubt und gemessen; `true`: Verspätung macht den Plan unzulässig |

„Je Art“ bedeutet ein Objekt mit den Schlüsseln `manual`, `automatic` und `setup`.

**Was sich ohne Code-Änderung umstellen lässt:** die Zeitfenster und Faktoren, Pausen, Tageslänge, `time_dependent`, `allowed_in_break`, `allowed_overnight`, `setups_on_manual_stations`, `initial_setup_minutes` und `due_dates_hard`. Alle außer `due_dates_hard` ändern den Fingerprint.

**Was nicht unterstützt wird:** Einige Schlüssel dokumentieren nur die geltende Annahme, zum Beispiel `setup_requires_worker: true` oder `non_preemptive: false`. Setzt man sie auf einen nicht umgesetzten Wert, lehnt `load_config` die Datei mit einer klaren Fehlermeldung ab, statt die Regel still zu ignorieren.

### `generation` – Parameter des Generators

Keine Scheduling-Regeln, sondern Einstellungen dafür, wie Instanzen erzeugt werden:

| Schlüssel | Wert | Bedeutung |
|---|---|---|
| `min_jobs` | `2` | Mindestanzahl Jobs |
| `release` | `0` | Freigabezeitpunkt aller Jobs |
| `kind_pattern` | `"alternating"` | manuell und automatisch im Schachbrettmuster |
| `manual_processing_range` | `[20, 40]` | Basisdauer manueller Operationen (min) |
| `manual_same_duration_on_all_stations` | `true` | `true`: eine Basisdauer für alle Handarbeitsplätze (Worker gleich schnell); `false`: eigene Dauer je Platz |
| `automatic_processing_range` | `[30, 60]` | Basiszeit automatischer Operationen (min) |
| `automatic_machine_deviation` | `0.2` | Abweichung je Maschine von der Basiszeit (±20 %); `null`: Zeit je Maschine unabhängig aus dem Bereich |
| `setup_range` | `[5, 15]` | Basis-Rüstzeit (min) |
| `first_automatic_on_all_machines` | `true` | erste automatische Operation auf allen Maschinen zulässig |
| `flexible_probability` | `0.7` | Anteil flexibler automatischer Operationen |
| `due_slack` | `0.05` | Standard-Puffer der Deadlines (siehe unten) |
| `schedule_attempts` | `50` | Versuche für den Referenz-Schedule |
| `max_consecutive_seed_failures` | `200` | Klassen-Modus: Abbruch nach so vielen Seeds in Folge ohne Referenz-Plan |

### `simulation` – Ausführung

| Schlüssel | Wert | Bedeutung |
|---|---|---|
| `overflow_days` | `1` | Zusatztage nach dem Horizont der Instanz, für Regeln und Umgebung. Deadlines bleiben unverändert. |

### `instance_classes` – Instanzgrößen

Je Klasse: `jobs`, `operations_per_job`, `workers`, `machines`, `days`. Optional `due_slack` und `schedule_attempts`; sie überschreiben den Wert aus `generation`.

### `data_splits` – Datensätze

| Datensatz | Quelle | Seeds | Anzahl |
|---|---|---|---|
| `train` | on the fly, immer neue Instanzen | ab `seed_start` = 1.000.000 | beliebig |
| `val` | Dateien in `instances/val/{class}` | ab `seed_start` = 100.000 | `count` = 50 je Klasse |
| `test` | Dateien in `instances/test/{class}` | ab `seed_start` = 42 | `count` = 100 je Klasse |

Instanz i bekommt den i-ten erzeugbaren Seed. `load_config` lehnt Seed-Bereiche ab, die sich überschneiden könnten. So sieht ein Agent seine Testinstanzen nie im Training.

### `experiments` – Auswertung

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `classes` | `"all"` | ausgewertete Klassen: `"all"` oder eine Liste |
| `split` | `"test"` | ausgewerteter Datensatz |
| `methods` | 4 Verfahren | Dispatching-Regeln als `sequencing`, `routing`, `view` |
| `results_file`, `summary_file` | `results/baselines*.csv` | Ergebnisse von `run_experiments.py` |
| `policy_results_file`, `policy_summary_file` | `results/policies*.csv` | Ergebnisse von `rl/evaluate.py` |
| `policy_seed` | `12345` | Seed der Zufalls-Policy |
| `evaluation_batch_size` | `1` | Batch-Größe beim Auswerten von Policies |

### `environment` – RL-Umgebung

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `train_classes` | `"all"` | Klassen fürs Training: `"all"` oder eine Liste; der Generator wechselt je Batch die Klasse |
| `view` | `"time_aware"` | `time_aware`: echte Faktoren; `static`: Faktor 1.0, Pause und Arbeitsende bekannt |
| `allow_wait` | `true` | Aktion 0 erlaubt: warten bis zum nächsten Ereignis |
| `device` | `"auto"` | `auto` nimmt die GPU, wenn es eine gibt; sonst `cpu` oder `cuda` |
| `batch_size` | `32` | Standard-Batchgröße von `env.reset()` |
| `generator_workers` | `0` | Prozesse für das Erzeugen on the fly (ändert das Ergebnis nicht) |
| `compute_features` | `true` | Merkmale für den Agenten berechnen |
| `time_scale_minutes` | `60` | teilt alle Zeit-Merkmale |
| `truncation_penalty_minutes` | `720` | Strafe je offener Operation, wenn der Horizont nicht reicht |

### `reward` – Hierarchisches Ziel

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `levels` | `["total_tardiness", "makespan"]` | Ebenen von wichtig nach unwichtig; erlaubt auch `late_jobs`, `max_tardiness` |
| `level_weights` | `null` | `null`: exakt lexikografisch; Liste je Ebene: stattdessen gewichtete Summe |
| `mode` | `"stepwise"` | `stepwise`: Änderung der geschätzten Kosten je Schritt; `terminal`: nur am Ende |
| `scale_minutes` | `60` | Reward = −Kosten / (Gewicht der obersten Ebene × Jobs × `scale_minutes`) |

### `model` – Neuronales Netz

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `variant` | `"A3"` | `A0`, `A1`, `A2`, `A3`, siehe [Varianten](#varianten) |
| `time_information` | `"full"` | `none`: Agent ohne Zeitfaktoren und Uhrzeit |
| `d_model`, `num_layers`, `num_heads` | `128`, `3`, `4` | Größe des Encoders |
| `hidden_dim` | `128` | Breite der MLPs |

### `training` – PPO

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `seed` | `0` | Startwert des Laufs |
| `iterations` | `500` | Anzahl Iterationen; jede nimmt die nächste Klasse aus `environment.train_classes` |
| `instances_per_iteration` | `32` | Neue Instanzen je Iteration |
| `ppo_epochs`, `minibatch_size` | `3`, `256` | Durchläufe und Größe der Minibatches |
| `learning_rate` | `0.0002` | Adam |
| `gamma`, `gae_lambda` | `1.0`, `0.95` | Ohne Abzinsung, weil die Rewards sich zu den echten Kosten aufsummieren |
| `clip_range`, `value_coef`, `entropy_coef`, `max_grad_norm` | `0.2`, `0.5`, `0.01`, `1.0` | PPO-Verlust |
| `reward_scale` | `10.0` | Multipliziert die Rewards nur fürs Lernen; die Rangfolge der Ziele bleibt |
| `validate_every`, `validation_instances` | `20`, `20` | Prüfung alle so viele Iterationen auf den ersten so vielen Validierungsinstanzen je Klasse |
| `validation_decode` | `"greedy"` | `greedy` oder `sample` |
| `best_metric` | `"val/reward_mean"` | Kennzahl für `best.pt`: `val/reward_mean`, `val/tardiness_mean`, `val/makespan_mean` oder `val/window_compliance` |
| `output_folder` | `"experiments/checkpoints"` | Ausgabeordner, relativ zur Config |

### `monitoring` – Experiment-Tracking

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `enabled` | `true` | `false`: kein Tracking, Training unverändert |
| `folder` | `"experiments"` | Ablage für `tensorboard`, `results`, `figures`, relativ zur Config |
| `flush_seconds` | `30` | So oft erscheinen neue Werte in TensorBoard |
| `log_every_updates` | `20` | `train/*` und `ppo/*` je so viele Optimizer-Schritte |
| `temporal_every_validations` | `1` | `temporal/*` je so viele Validierungen; die letzte immer |
| `gantt_every_validations` | `5` | Gantt-Diagramme ab der ersten Validierung je so viele; die letzte immer |
| `gantt_instances` | `3` | So viele feste Validierungsinstanzen, reihum aus den Klassen |
| `gantt_to_tensorboard` | `true` | Bilder auch nach TensorBoard |
| `save_train_instances` | `true` | Eine Zeile je Trainingsinstanz |
| `test_after_training` | `true` | `best.pt` am Ende auf den Testinstanzen auswerten |
| `test_instances` | `null` | `null`: alle; sonst die ersten N je Klasse |
| `static_reference` | EDD+ECT, `static` | Regel für `temporal/*_gain`; `null`: keine |

### Fingerprint

Jede Instanz speichert einen **Fingerprint**, einen Hash über `time_model` und `rules`. Ändert man später die Regeln, passt eine alte Instanz nicht mehr dazu, und der Validator meldet:

```
Instanz wurde mit anderen Regeln erzeugt als in der aktuellen problem_config ...
```

So wird eine Instanz nie versehentlich mit anderen Regeln geprüft, als mit denen sie erzeugt wurde. **Nicht** in den Fingerprint gehen ein: alle Abschnitte außer `time_model` und `rules`, Kommentare und reine Bewertungsschalter wie `due_dates_hard` (Liste `EVALUATION_SWITCHES` in `schedule_validator.py`). Diese ändern die Daten einer Instanz nicht, deshalb kann man sie umschalten, ohne Instanzen neu zu erzeugen.

---

## generator.py – Instanzen erzeugen

### Ablauf

`generate_instance(...)` arbeitet in vier Schritten:

1. **Jobs und Operationen** mit zulässigen Ressourcen und Basisdauern erzeugen.
2. **Rüstzeiten** für jedes geordnete Paar von Operationen erzeugen, die auf derselben Maschine laufen dürfen.
3. **Referenz-Schedule** bauen, also einen zulässigen Plan, der alle Regeln einhält.
4. **Deadlines** aus dem Referenz-Schedule ableiten.

Gleicher Seed und gleiche Parameter ergeben immer **exakt dieselbe Instanz**.

### Lösbarkeit per Konstruktion

Jede gespeicherte Instanz ist **garantiert lösbar, und zwar ohne Verspätung**. Das gilt auch, wenn Deadlines hart geschaltet sind. Das Vorgehen entspricht dem bei TSP mit Zeitfenstern (vgl. Dumas et al. 1995): Erst wird ein gültiger Plan gebaut, dann werden die Deadlines so gelegt, dass dieser Plan sie einhält:

```
due_j = min(Horizontende, C_j + Zufall(0 … due_slack) × C_j)
```

`C_j` ist die Fertigstellungszeit von Job j im Referenz-Schedule. Über `due_slack` steuerst du die Schwierigkeit:

| `due_slack` | Wirkung |
|---|---|
| `0` | Deadlines so eng wie der Referenz-Schedule, schwer |
| `0.05` | bis zu 5 % Puffer (Standard) |
| `0.2` | bis zu 20 % Puffer |
| `0.5` | bis zu 50 % Puffer, eher leicht |

Der Referenz-Schedule wird in der Instanz unter `reference_schedule` gespeichert. Damit ist die Lösbarkeit jeder Instanz nachweisbar.

### Die Heuristik für den Referenz-Schedule

Es ist eine **randomisierte serielle Listenplanung mit mehreren Starts**:

1. **Reihenfolge:** Aus allen Jobs mit offenen Operationen wird **zufällig** einer gewählt und seine nächste Operation eingeplant.
2. **Zuordnung:** Für jede zulässige Ressource wird berechnet, wann die Operation dort fertig wäre, einschließlich Rüstung und zeitabhängiger Dauer. Gewählt wird die Ressource mit dem **frühesten Ende**.
3. **Startzeit:** der früheste Zeitpunkt nach dem letzten Auftrag dieser Ressource und nach dem Vorgänger im Job, an dem keine Regel verletzt wird. Pause und Nacht werden übersprungen.
4. **Nur anhängen:** Neue Operationen kommen hinter den letzten Auftrag einer Ressource, frühere Lücken werden nicht aufgefüllt.
5. **Mehrere Starts:** Das läuft `schedule_attempts`-mal (Standard 50). Der Plan mit dem **kleinsten Makespan** wird genommen.

Ob ein Zeitpunkt zulässig ist, entscheidet ausschließlich `duration(...)` aus dem Validator, also die Regeln der Config. Die Heuristik ist bewusst einfach und **nicht optimal**. Sie dient als Lösbarkeitsnachweis und als **Vergleichswert (Baseline)**: Ein guter Scheduler sollte sie schlagen.

Findet keiner der Versuche einen Plan im Horizont, bricht der Generator mit einer Meldung ab, statt eine unlösbare Instanz zu speichern:

```
ValueError: Kein zulaessiger Referenz-Schedule im Horizont gefunden. Instanz zu gross ...
```

### Parameter

| Größe | Kommandozeile | Python (`generate_instance`) | Standard CLI | Standard Python | Minimum |
|---|---|---|---|---|---|
| Jobs | `--jobs` | `n_jobs` | 2 | 5 | 2 |
| Operationen pro Job | `--ops` | `n_ops` | 2 | 5 | 1 |
| Maschinen | `--machines` | `n_machines` | 2 | 5 | 1 |
| Worker (= Handarbeitsplätze) | `--workers` | `n_workers` | 1 | 2 | 1 |
| Arbeitstage | `--days` | `n_days` | 1 | 1 | 1 |
| Deadline-Puffer | `--slack` | `due_slack` | aus Config (0.05) | aus Config (0.05) | 0 |
| Seed | `--seed` | `seed` | 42 | 42 | – |
| Versuche Referenz-Schedule | – | `schedule_attempts` | aus Config (50) | aus Config (50) | 1 |

Nur auf der Kommandozeile:

| Option | Standard | Bedeutung |
|---|---|---|
| `--classes` | – | Klassen aus der Config oder `all`. Größen, Seeds, Anzahl und Ordner kommen dann aus der Config; Seeds ohne Referenz-Plan werden übersprungen |
| `--splits` | `test` | Datensätze im Klassen-Modus: `test`, `val` |
| `--processes` | 1 | Prozesse im Klassen-Modus (ändert das Ergebnis nicht) |
| `--count` | 3 | Anzahl Instanzen; Instanz i bekommt Seed `seed + i` (Klassen-Modus: aus der Config) |
| `--out` | `instances` | Ausgabeordner (`instance_000.json`, `instance_001.json`, …) |
| `--config` | `problem_config.json` | Regeldatei |
| `--show` | aus | Referenz-Schedule als Tabelle und Gantt-Diagramm anzeigen |

### Welche Größen passen in einen Arbeitstag?

Richtwerte aus einer Messung mit je 50 Seeds, 1 Tag, `due_slack = 0.2`, gleich schnellen Workern und Maschinenzeiten ±20 % um die Basiszeit. Ab 10 Jobs wurde in Zweierschritten gesucht. Angegeben ist die größte Zahl an Jobs, bei der **jede** Instanz erzeugt werden konnte:

| Worker | Maschinen | 2 Ops/Job | 3 Ops/Job | 4 Ops/Job | 5 Ops/Job |
|---|---|---|---|---|---|
| 1 | 2 | 16 | 10 | 8 | 6 |
| 1 | 3 | 16 | 10 | 8 | 6 |
| 2 | 2 | 22 | 14 | 10 | 8 |
| 2 | 3 | 32 | 20 | 14 | 10 |
| 2 | 4 | 34 | 22 | 14 | 12 |
| 3 | 3 | 34 | 22 | 16 | 12 |
| 3 | 4 | 44 | 28 | 18 | 14 |

Faustregeln:
- höchstens etwa **15 manuelle Operationen pro Worker** und **12 automatische Operationen pro Maschine** und Tag
- es zählt der **Engpass**: Mit 1 Worker bringen mehr Maschinen kaum etwas, mit 2 Maschinen bringen mehr Worker kaum etwas
- mit **mehr Tagen** steigen die Grenzen

Für Experimente empfiehlt sich etwas Abstand zu den Grenzen, etwa 75 % des Tabellenwerts.

### Anzeige (`--show`)

Im Gantt-Diagramm ist der Teil eines Balkens dunkler, den der Zeitfaktor zur Basisdauer hinzufügt. Bei Operationen steht die Verlängerung in Minuten unter dem Namen (z. B. `+12`), bei Rüstungen ist sie nur abgedunkelt. Der Titel nennt die Summe. Jeder Job hat eine eigene Farbe. Unter den Ressourcen steht ein Streifen mit den **Deadlines**: Raute, Job und Uhrzeit, dazu eine gestrichelte Linie in der Farbe des Jobs nach oben. Das gilt für jeden Plan, den `generator.plot_schedule` zeichnet, also auch für `env.render` und `rl.evaluate --show`.

Im Terminal erscheint je Ressource der Plan mit Uhrzeiten, dazu eine Übersicht der Jobs:

```
Referenz-Schedule
  H1 (W1)
    06:00-06:40  J1_O1
    06:40-07:04  J2_O2
  M1
    06:00-06:37  J2_O1
  M2
    06:40-07:10  J1_O2

  Job   fertig  Deadline  Puffer
  J1    07:10   07:17      7 min
  J2    07:04   07:16     12 min
  Makespan: 07:10
```

Bei mehreren Tagen steht der Tag vor der Uhrzeit, zum Beispiel `T2 07:15`. Außerdem öffnet sich ein **Gantt-Diagramm**: Zeilen sind Ressourcen, die x-Achse ist die Uhrzeit. Operationen sind farbig (eine Farbe je Job, beschriftet mit der Operations-ID), Rüstungen schraffiert, die Pause ist grau hinterlegt, Tagesgrenzen sind als senkrechte Linie markiert. Ab 9 Jobs sind alle Balken einfarbig, die Jobs erkennt man dann an der Beschriftung.

Im Code lässt sich das Diagramm auch speichern:

```python
from pathlib import Path
from schedule_validator import load_config
from generator import generate_instance, plot_reference_schedule

config = load_config()
instance = generate_instance(seed=42, n_jobs=5, n_machines=2, n_ops=3, n_workers=2, config=config)
plot_reference_schedule(config, instance, Path("gantt.png"))
```

---

## Instanzformat

Eine Instanz enthält **nur Daten**, keine Regeln. Beispiel (2 Jobs, 2 Ops, 2 Maschinen, 1 Worker, Seed 42, gekürzt):

```json
{
  "schema_version": 2,
  "problem_config": {"file": "problem_config.json", "fingerprint": "d3db92fe01d6a4b1"},
  "seed": 42,
  "days": 1,
  "resources": {"manual_stations": ["H1"], "machines": ["M1", "M2"], "workers": ["W1"]},
  "station_worker": {"H1": "W1"},
  "jobs": [
    {"id": "J1", "operations": ["J1_O1", "J1_O2"], "release": 0, "due": 77},
    {"id": "J2", "operations": ["J2_O1", "J2_O2"], "release": 0, "due": 76}
  ],
  "operations": [
    {"id": "J1_O1", "job": "J1", "kind": "manual",    "base_processing": {"H1": 40}},
    {"id": "J1_O2", "job": "J1", "kind": "automatic", "base_processing": {"M1": 33, "M2": 30}},
    {"id": "J2_O1", "job": "J2", "kind": "automatic", "base_processing": {"M1": 37}},
    {"id": "J2_O2", "job": "J2", "kind": "manual",    "base_processing": {"H1": 24}}
  ],
  "setups": [
    {"resource": "M1", "from_op": "J1_O2", "to_op": "J2_O1", "base_minutes": 6},
    {"resource": "M1", "from_op": "J2_O1", "to_op": "J1_O2", "base_minutes": 15}
  ],
  "reference_schedule": [
    {"op": "J1_O1", "resource": "H1", "start": 0, "end": 40},
    {"op": "J2_O1", "resource": "M1", "setup_from": null, "setup_start": 0, "setup_end": 0, "start": 0, "end": 37},
    {"op": "J1_O2", "resource": "M2", "setup_from": null, "setup_start": 40, "setup_end": 40, "start": 40, "end": 70},
    {"op": "J2_O2", "resource": "H1", "start": 40, "end": 64}
  ]
}
```

| Feld | Bedeutung |
|---|---|
| `problem_config` | Name und Fingerprint der Regeldatei, mit der die Instanz erzeugt wurde |
| `days` | Anzahl Arbeitstage; der Horizont ist `days × day_minutes` |
| `jobs[].operations` | Operationen in der Reihenfolge, in der sie bearbeitet werden müssen |
| `jobs[].release` / `due` | frühester Start und Deadline (Arbeitsminuten) |
| `operations[].kind` | `"manual"` oder `"automatic"` |
| `operations[].base_processing` | **zulässige Ressourcen** (Schlüssel) mit Basisdauer (Wert) |
| `setups` | Basis-Rüstzeit von `from_op` auf `to_op` auf `resource` |
| `reference_schedule` | ein zulässiger Plan (Lösbarkeitsnachweis, siehe unten) |

Alle Zeiten sind **Arbeitsminuten ab Beginn von Tag 1**, die Nächte sind nicht mitgezählt.

### Schedule-Format

Jeder Eintrag eines Schedules beschreibt, wo und wann eine Operation läuft:

```json
{"op": "J1_O2", "resource": "M2", "setup_from": "J2_O1", "setup_start": 40, "setup_end": 52, "start": 52, "end": 82}
```

- `op`, `resource`, `start`, `end` sind immer Pflicht.
- Auf Ressourcen mit Rüstung (Maschinen) kommen dazu `setup_from` (vorherige Operation auf dieser Ressource, `null` bei der ersten), `setup_start` und `setup_end`. Auch eine Rüstung von 0 Minuten wird so angegeben.

Dein Scheduler muss Schedules in diesem Format ausgeben, damit der Validator sie prüfen kann. Eine Schedule-Datei ist entweder eine JSON-Liste solcher Einträge oder ein Objekt `{"schedule": [...]}`.

> **Hinweis:** Der Referenz-Schedule ist ein **Lösungsbeispiel**. Ein Scheduler oder Agent darf ihn beim Lösen nicht lesen.

---

## schedule_validator.py – Instanzen und Schedules prüfen

### Kommandozeile

```powershell
python schedule_validator.py instances/instance_000.json            # eine Instanz (prüft reference_schedule)
python schedule_validator.py instances                              # alle *.json im Ordner
python schedule_validator.py instances/instance_000.json --schedule mein_plan.json
python schedule_validator.py instances --config andere_regeln.json
```

Beispielausgabe:

```
OK      instances\instance_000.json
FEHLER  instances\instance_001.json
        - J5_O3: Manuelle Operation darf nicht in der Pause starten

1 von 2 zulaessig
```

Exit-Code `0`, wenn alles zulässig ist, sonst `1`. Damit lässt sich der Validator auch in Skripten verwenden.

Der Validator beantwortet nur die Frage **„zulässig oder nicht?“**. Verspätungen meldet er nur, wenn Deadlines hart geschaltet sind. Wie stark ein Plan Deadlines verfehlt, misst [`evaluation.py`](#evaluationpy--makespan-und-verspätungen).

### Was geprüft wird

**Instanz (`validate_instance`)**
- Fingerprint passt zur aktuellen Config
- Tage ≥ 1, jeder Handarbeitsplatz hat genau einen Worker
- eindeutige Operations-IDs, jede Operation gehört genau einem Job
- Release und Deadline liegen im Horizont
- manuelle Operationen nur auf Handarbeitsplätzen, automatische nur auf Maschinen, Basisdauern positiv
- Rüstzeiten nur auf Ressourcen mit Rüstung und nur für passende Operationen

**Schedule (`check_schedule`)**
- jede Operation genau einmal eingeplant
- nur auf einer zulässigen Ressource
- richtige Dauer zum Startzeitpunkt (Zeitfaktor)
- nicht in der Pause, nicht über Nacht, nicht über das Horizontende (je nach Tätigkeitsart)
- Reihenfolge innerhalb des Jobs, Release
- Deadline, **nur wenn** `due_dates_hard: true`
- je Ressource: richtige Rüstung (Vorgänger, Dauer, Zeitpunkt), keine Überlappung

### Python-Schnittstelle

| Funktion | Zweck |
|---|---|
| `load_config(path)` | Config laden und auf Gültigkeit prüfen |
| `load_json(path)` | Instanz oder Schedule laden |
| `horizon(config, instance)` | Horizont in Arbeitsminuten |
| `factor_at(config, t)` | Zeitfaktor zum Zeitpunkt t (`None` = Pause) |
| `duration(config, instance, kind, base, start)` | tatsächliche Dauer; `ValueError`, wenn der Start unzulässig ist |
| `interval_violation(config, instance, kind, start, end)` | `None` oder Grund, warum ein Intervall unzulässig ist |
| `processing_duration(config, instance, op_id, resource, start)` | Dauer einer Operation auf einer Ressource |
| `setup_duration(config, instance, resource, previous, next_op, start)` | Dauer einer Rüstung |
| `validate_instance(config, instance)` | Liste der Strukturfehler (leer = ok) |
| `check_schedule(config, instance, schedule)` | Liste der Regelverstöße (leer = zulässig) |
| `all_resources(instance)`, `setup_resources(config, instance)` | alle Ressourcen bzw. Ressourcen mit Rüstung |
| `instance_files(paths)`, `load_schedule(path)` | Dateien/Ordner auflösen, Schedule-Datei laden |

```python
from schedule_validator import load_config, load_json, validate_instance, check_schedule

config = load_config()
instance = load_json("instances/test/klein_11/instance_000.json")
schedule = mein_scheduler(instance)          # Liste im Schedule-Format

errors = validate_instance(config, instance) or check_schedule(config, instance, schedule)
print("zulässig" if not errors else errors)
```

---

## evaluation.py – Makespan und Verspätungen

Bewertet einen Schedule: Er wird zuerst mit dem Validator auf die harten Regeln geprüft, danach werden die Kennzahlen berechnet.

### Kommandozeile

```powershell
python evaluation.py instances/test/klein_11                         # Referenz-Pläne aller Instanzen
python evaluation.py instances/test/klein_11/instance_000.json --schedule mein_plan.json
```

Ausgabe, eine Zeile je Instanz:

```
Instanz                                  zulaessig Makespan verspaetet  Summe   Max
instances\test\klein_11\instance_000.json        ja      629    0 / 11       0     0
```

Zeiten in Arbeitsminuten. „verspaetet“ ist die Anzahl verspäteter Jobs, „Summe“ und „Max“ beziehen sich auf die Verspätung. Exit-Code `1`, wenn ein Plan unzulässig ist.

### Python-Schnittstelle

| Funktion | Ergebnis |
|---|---|
| `evaluate(config, instance, schedule)` | Dictionary mit allen Kennzahlen (siehe unten) |
| `job_tardiness(instance, schedule)` | je Job: Fertigstellung, Deadline, Verspätung |
| `job_completion(schedule, job)` | Fertigstellungszeit eines Jobs |
| `makespan(schedule)` | Ende der letzten Operation |

Rückgabe von `evaluate`:

| Schlüssel | Bedeutung |
|---|---|
| `feasible` | `True`, wenn keine harte Regel verletzt ist |
| `violations` | Liste der Regelverstöße |
| `makespan` | Ende der letzten Operation |
| `jobs` | Anzahl Jobs |
| `late_jobs` | Anzahl verspäteter Jobs |
| `total_tardiness` | Summe der Verspätungen |
| `max_tardiness` | größte Verspätung |
| `mean_tardiness` | mittlere Verspätung über alle Jobs |
| `per_job` | Fertigstellung, Deadline und Verspätung je Job |

```python
from schedule_validator import load_config, load_json
from evaluation import evaluate

config = load_config()
instance = load_json("instances/test/klein_11/instance_000.json")
result = evaluate(config, instance, mein_schedule)
print(result["feasible"], result["late_jobs"], result["total_tardiness"])
```

Der Referenz-Plan hat immer 0 Verspätung, weil die Deadlines aus ihm abgeleitet sind.

---

## Simulator, Baselines und Experimente

### Festgelegte Annahmen für Planung und Ausführung

Diese Festlegungen gelten für alle Verfahren gleich, also für alle Baselines und später auch für den RL-Ansatz. Sie stehen auch in den Docstrings von `simulator.py`.

| # | Frage | Festlegung |
|---|---|---|
| 1 | Was weiß ein **statisches** Verfahren? | Es rechnet mit **Faktor 1.0**, kennt aber **Pause und Arbeitsende**. Der Vergleich statisch / zeitbewusst misst damit genau den Effekt der zeitabhängigen Leistung und nicht nebenbei den einer unbekannten Pause. |
| 2 | Was, wenn ein Plan in der Realität nicht mehr in den Tag passt? | Planung und Ausführung dürfen **Zusatztage** nutzen (`simulation.overflow_days`, Standard 1). Die Deadlines bleiben unverändert, die Verspätung wird normal gemessen. Bewertet wird deshalb gegen den verlängerten Horizont. |
| 3 | Wann wird gerüstet? | Die Rüstung beginnt, wenn **Maschine und Job bereit** sind, die Bearbeitung folgt **direkt**. Passt beides zu diesem Zeitpunkt nicht (Pause, Arbeitsende), beginnen beide zum nächsten zulässigen Zeitpunkt. |
| 4 | Wie läuft die Planung ab? | **Ereignisgesteuert (non-delay):** Die Uhr springt von Ereignis zu Ereignis (Ressource wird frei, Job wird bereit, Zeitfenster beginnt). Wählbar sind nur Paare aus *bereiter Operation* und *freier zulässiger Ressource*, die **jetzt** regelkonform starten können. |

### simulator.py – Planung und Ausführung

Ein statisches Verfahren plant mit Basisdauern. Sein Plan enthält deshalb **falsche Uhrzeiten**: Um 13:00 dauert eine manuelle Operation in Wirklichkeit 1,6-mal so lange, und alles danach verschiebt sich. `execute_plan` spielt einen Plan **in der Wirklichkeit nach**: Zuordnung und Reihenfolge je Ressource werden übernommen, die Uhrzeiten mit den echten Dauern neu berechnet. Erst dann sieht man, welche Deadlines tatsächlich verfehlt werden.

| Element | Aufgabe |
|---|---|
| `DURATION_VIEWS` | `"time_aware"` (echte Dauer) oder `"static"` (Faktor 1.0, Pause und Arbeitsende bekannt) |
| `view_config(config, view)` | Config aus Sicht eines Verfahrens |
| `execution_instance(config, instance)` | Instanz mit Ausführungshorizont (Tage + `simulation.overflow_days`) |
| `SimulationState(config, instance, view)` | Zustand der ereignisgesteuerten Planung |
| `.time` | aktueller Entscheidungszeitpunkt |
| `.candidate_operations()` | Operationen, die **jetzt** auf mindestens einer freien Ressource starten können |
| `.options(op_id)` | alle jetzt möglichen Schedule-Einträge dieser Operation (je freie zulässige Ressource, mit Rüstung und Dauer) |
| `.apply(entry)` | einen dieser Einträge einplanen |
| `.advance()` | Uhr zum nächsten Ereignis; `False`, wenn keins mehr kommt |
| `.done()`, `.schedule` | fertig?, bisheriger Plan |
| `plan_from_schedule(schedule)` | Zuordnung und Reihenfolge aus einem Schedule ablesen |
| `execute_plan(config, instance, plan)` | Plan mit echten Dauern ausführen (frühester zulässiger Start, Zusatztag erlaubt) |
| `first_feasible_start(...)` | frühester zulässiger Start für Rüstung + Bearbeitung (nutzt auch der Generator) |
| `SimulationError` | Plan ist auch mit Zusatztag nicht ausführbar |

### baselines/ – Prioritätsregeln

| Regel | Art | Wahl | Gleichstand |
|---|---|---|---|
| **EDD** (Earliest Due Date) | Reihenfolge | Operation des Jobs mit der frühesten Deadline | Reihenfolge der Jobs |
| **ECT** (Earliest Completion Time) | Zuordnung | Ressource mit dem frühesten Ende (Rüstung + Bearbeitung) | Reihenfolge der Ressourcen |
| **SPT** (Shortest Processing Time) | Zuordnung | Ressource mit der kürzesten Bearbeitungszeit (ohne Rüstung) | früheres Ende, dann Reihenfolge |

Zur Auswahl stehen immer nur die **jetzt freien** zulässigen Ressourcen (non-delay). ECT und SPT unterscheiden sich daher vor allem darin, ob die **Rüstzeit** mitzählt. Alle Dauern gelten in der Sicht des Verfahrens.

`dispatcher.py`:
- `build_plan(config, instance, sequencing, routing, view)` baut den Plan ereignisgesteuert in der Dauer-Sicht des Verfahrens.
- `run_dispatcher(...)` baut den Plan und führt ihn mit den echten Dauern aus. Rückgabe: `{"planned": ..., "executed": ...}`.

| Verfahren | plant mit | ausgeführt mit |
|---|---|---|
| EDD + ECT **statisch** | Basiszeiten (Faktor 1.0) | echten Dauern (`execute_plan`) |
| EDD + ECT **zeitbewusst** | echten Dauern | echten Dauern |
| EDD + SPT **statisch** | Basiszeiten (Faktor 1.0) | echten Dauern |
| EDD + SPT **zeitbewusst** | echten Dauern | echten Dauern |
| RL-Agent (später) | echten Dauern (über `SimulationState`) | echten Dauern |

Bei zeitbewussten Verfahren stimmt die Ausführung mit dem Plan überein oder ist an einzelnen Stellen früher, weil die Ausführung minutengenau den frühesten zulässigen Start sucht.

**Weitere Regel ergänzen:** Funktion in `baselines/rules.py` schreiben, in `SEQUENCING_RULES` bzw. `ROUTING_RULES` eintragen und in `problem_config.json` unter `experiments.methods` aufnehmen.

### Ablauf eines Experiments: statisch und zeitbewusst

Je Instanz und Verfahren laufen vier Schritte (`run_experiments.run_instance`):

| Schritt | Funktion | statisch | zeitbewusst |
|---|---|---|---|
| 1. Sicht festlegen | `view_config` | alle Faktoren 1.0, Pause und Arbeitsende bekannt | echte Faktoren (1.0 bis 1.6) |
| 2. Planen | `build_plan` mit `SimulationState` | EDD + ECT/SPT ereignisgesteuert, Dauern mit Faktor 1.0 | dasselbe Verfahren, Dauern mit echtem Faktor zum Startzeitpunkt |
| 3. Ausführen | `plan_from_schedule` + `execute_plan` | Zuordnung und Reihenfolge je Ressource werden übernommen, **alle Uhrzeiten neu** mit echten Dauern | dasselbe; Ergebnis gleich dem Plan oder an einzelnen Stellen früher |
| 4. Bewerten | `evaluate` | nur der **ausgeführte** Plan | nur der **ausgeführte** Plan |

Beide Varianten unterscheiden sich **nur** in Schritt 1. Regeln, Ereignislogik, Ausführung und Bewertung sind identisch.

**Schritt 2 im Detail (Planen).** Die Uhr steht auf einem Ereignis. Für jede nächste Operation eines Jobs prüft `SimulationState.options`, auf welchen **jetzt freien** Ressourcen sie **jetzt** starten kann. Dafür berechnet es die Dauer in der Sicht des Verfahrens und prüft die Regeln (keine Handarbeit oder Rüstung in der Pause, nichts über Nacht). EDD wählt unter diesen Operationen die mit der frühesten Deadline, ECT oder SPT wählt die Ressource. Gibt es keine Wahl, springt die Uhr zum nächsten Ereignis. Diese Ereignisse sind: eine Ressource wird frei, ein Job wird bereit, ein Zeitfenster beginnt.

**Wo die Zeitabhängigkeit beim Planen eingeht** (nur zeitbewusst):
1. **Was jetzt starten darf.** Eine 25-min-Handoperation um 11:35 passt statisch bis 12:00 in den Vormittag. Mit Faktor 1.2 dauert sie 30 min und würde in die Pause laufen, also bietet die zeitbewusste Sicht sie jetzt nicht an.
2. **Wann die nächsten Ereignisse liegen.** Das Ende jeder Operation und damit der Zeitpunkt der nächsten Entscheidung stimmt mit der Realität überein. Spätere Entscheidungen beruhen deshalb auf dem richtigen Zustand.
3. **Wahl der Ressource kaum.** Alle Optionen beginnen zum selben Zeitpunkt und haben damit denselben Faktor. Die Rangfolge von ECT und SPT ändert sich dadurch fast nie.

Vor 09:00 gilt überall Faktor 1.0. Bis dahin treffen beide Sichten **dieselben Entscheidungen**. Gemessen bei EDD+ECT: Die erste Abweichung liegt je nach Klasse im Median zwischen 09:32 und 09:53. Völlig gleiche Entscheidungen über die ganze Instanz gibt es in keiner Testinstanz.

**Schritt 3 im Detail (statischer Plan in der Realität).** Der statische Plan hat falsche Uhrzeiten, weil er Faktor 1.0 annimmt. `execute_plan` übernimmt deshalb nur, *welche Operation auf welcher Ressource* und *in welcher Reihenfolge* läuft. Jede Operation startet zum frühesten zulässigen Zeitpunkt nach ihrem Vorgänger im Job und dem vorherigen Auftrag auf ihrer Ressource. Die Dauer ist die echte Dauer zu diesem Start. Folgen:
- Operationen und Rüstungen ab 09:00 dauern länger, am stärksten 12:30–14:00. Alles danach auf derselben Ressource und im selben Job verschiebt sich.
- Eine Handoperation, die statisch noch vor die Pause passte, passt real nicht mehr und beginnt erst um 12:30. Die Ressource steht dann bis zur Pause still.
- Was real nicht mehr in den Tag passt, läuft in den Zusatztag.
- Die Entscheidungen selbst (Reihenfolge, Zuordnung) passt der statische Plan **nicht** an. Er reagiert nicht darauf, dass sich die Lage geändert hat.

Der statische Plan unterschätzt deshalb Makespan und Verspätungen (vergleiche `planned_*` mit den ausgeführten Werten). Bei EDD+ECT liegt der echte Makespan im Mittel 87–117 Minuten über dem geplanten, je nach Klasse.

**Warum auch zeitbewusste Verfahren verspätete Jobs haben.** Die zeitbewussten Baselines *kennen* die echten Dauern, *nutzen* sie aber nicht zur Optimierung:
1. **EDD schaut nur auf die Deadline**, nicht auf Dauern, Restarbeit oder Zeitfenster. Eine Operation in ein günstigeres Fenster zu legen (z. B. 14:00 statt 13:50, wenn der Faktor von 1.6 auf 1.3 fällt) ist keine Option der Regel.
2. **Non-delay, kein Vorausschauen.** Ist eine Ressource frei, wird sie sofort mit einer *jetzt bereiten* Operation belegt, auch wenn wenige Minuten später ein dringenderer Job dort ankommt. Gezielt warten ist nicht möglich. Bei den Wartezeiten verspäteter Jobs (EDD+ECT zeitbewusst) war die Ressource zu 7–16 % von einem Job mit **späterer** Deadline belegt, der sie vorher bekommen hatte, und zu 82–92 % von dringenderen Jobs. Nur 1–2 % der Wartezeit entfallen auf Pause, Arbeitsende oder Rüsten.
3. **Enge Deadlines.** Jede Deadline liegt nur 0–5 % hinter der Fertigstellung im Referenz-Plan (Median ca. 2,5 %). Der Referenz-Plan ist der beste aus 50 zufälligen Durchläufen. Er plant seriell, nicht non-delay, und darf Ressourcen deshalb für einen Job frei halten. Wer bei einem Job nicht mindestens so früh fertig wird wie der Referenz-Plan, verfehlt die Deadline schnell. Schon kleine Verzögerungen reichen deshalb; der Puffer verspäteter Jobs (Median 2,4–2,9 %) unterscheidet sich kaum vom Durchschnitt.

**Beispiel** `instances/test/klein_11/instance_000.json`, Job J6 (Deadline 09:27), EDD+ECT:

| | J6_O1 | J6_O2 | J6_O3 | J6_O4 | J6_O5 | fertig |
|---|---|---|---|---|---|---|
| Referenz-Plan | M2 06:00–06:38 | H1 06:38–07:01 | M3 07:01–07:44 | H1 07:44–08:18 | M3 08:25–09:17 | **09:17** pünktlich |
| zeitbewusst = statisch | M3 06:00–06:37 | H2 06:37–07:00 | M4 07:16–07:56 | H1 08:05–08:39 | M1 08:46–09:55 | **09:55** (+28 min) |

Die Zeiten enthalten die Rüstung. Um 07:00 ist J6_O3 bereit, darf aber nur auf M3 oder M4 laufen, und beide sind belegt, M4 bis 07:16 mit J2_O1 (Deadline 13:25), das dort schon um 06:30 gestartet wurde. Der Rückstand zieht sich durch die folgenden Operationen. Die Zeitinformation hilft hier nicht: Beide Sichten entscheiden für J6 gleich.

**Folgerung für den Vergleich:** Zeitbewusst heißt bei den Baselines nur, dass das Verfahren die Dauern *richtig vorhersagt*. Das senkt die Anzahl verspäteter Jobs je nach Klasse und Regel um 7–26 %.

Eine Zufallssuche zeigt, wie viel Spielraum darüber hinaus besteht. Sie baut auf den ersten 30 Testinstanzen je 80 Pläne, wählt die Operation zu 70 % nach EDD und behält den in ihrer Sicht besten Plan. Gezählt wird immer in der echten Ausführung. Die Regeln laufen auf allen 100 Testinstanzen, p stammt aus einem gepaarten Vorzeichentest ohne Korrektur für Mehrfachtests:

| Klasse | EDD+ECT: verspätete Jobs (statisch → zeitbewusst) | p | Suche: verspätete Jobs (statisch → zeitbewusst) | Suche: Verspätung je Instanz | Suche zeitbewusst: Instanzen ohne Verspätung |
|---|---|---|---|---|---|
| `klein_11` | 17,3 → 14,7 % | 0,033 | 12,1 → 3,6 % | 45 → 7 min | 18 / 30 |
| `mittel_16` | 14,0 → 12,0 % | 0,004 | 16,0 → 3,5 % | 87 → 20 min | 16 / 30 |
| `gross_21` | 13,0 → 9,5 % | < 0,0001 | 14,0 → 4,0 % | 102 → 25 min | 12 / 30 |
| `klein_12` | 14,6 → 11,7 % | 0,005 | 14,4 → 1,4 % | 61 → 1 min | 25 / 30 |
| `mittel_18` | 11,1 → 10,2 % | 0,14 | 12,0 → 2,2 % | 66 → 14 min | 19 / 30 |
| `gross_24` | 11,7 → 9,1 % | 0,0001 | 15,7 → 4,3 % | 122 → 40 min | 8 / 30 |

Daraus folgt: Bessere Auswahl schließt einen großen Teil der Lücke zum Optimum 0, aber nur mit Zeitinformation. Ohne sie ist die Suche nicht klar besser als die Regeln. Zufälliges Warten hilft nicht: Auf 20 Instanzen je Klasse hat die Suche mit 80 Plänen ohne Warten 66 verspätete Jobs, mit 5 % Warten 82 und mit 10 % Warten 124. Ob gezieltes Warten hilft, ist damit nicht beantwortet. Ein späterer Start kann im aktuellen Zeitmodell früher fertig sein, weil der Faktor um 10:00, 14:00 und 16:00 fällt. Die Suche prüft jeden Plan bis zum Ende; sie ist deshalb ein Maßstab für den Spielraum und zugleich der Vergleich, an dem sich ein RL-Agent messen muss.

### Kennzahlen

Grundlage ist immer der **ausgeführte** Plan (echte Dauern). Je Instanz und Verfahren (`results/baselines.csv`):

| Spalte | Bedeutung |
|---|---|
| `late_jobs` | Anzahl Jobs mit Fertigstellung C_j > Deadline d_j |
| `late_share` | `late_jobs` / Jobs der Instanz |
| `max_tardiness`, `total_tardiness` | größte und summierte Verspätung max(0, C_j − d_j) in Minuten |
| `makespan` | Ende der letzten Operation (Arbeitsminuten ab 06:00 an Tag 1) |
| `reference_makespan` | Makespan des Referenz-Plans des Generators (nur zur Information) |
| `planned_late_jobs`, `planned_makespan` | was das Verfahren in seinem eigenen Plan erwartet; beim statischen Verfahren zeigt der Abstand zur Ausführung, wie stark es sich verschätzt |
| `runtime_s` | Planen + Ausführen, in Sekunden |

Je Klasse und Verfahren, also die Ergebnistabelle (`results/baselines_summary.csv`):

| Kennzahl | Spalte | Berechnung |
|---|---|---|
| **Verspätete Jobs** (Hauptkennzahl) | `late_share` (`late_jobs_total`, `jobs_total`) | Summe der verspäteten Jobs über alle Instanzen der Klasse / Summe **aller Jobs** über alle Instanzen. Das Optimum ist **bekannt und 0**, denn der Referenz-Plan jeder Instanz hält alle Deadlines ein. |
| **Max. Verspätung** | `max_tardiness_mean` | größte Verspätung je Instanz (über alle Jobs der Instanz), dann gemittelt über die Instanzen. Instanzen ohne Verspätung zählen mit 0. |
| **Makespan** | `makespan_mean` | Makespan gemittelt über die Instanzen |
| **Gap** beim Makespan | `makespan_gap_pct` | 100 · (`makespan_mean` − bester `makespan_mean` der Klasse) / bester `makespan_mean` der Klasse. Bezug ist der **beste Wert dieser Klasse in der Tabelle** (`best_makespan_mean`). Das beste Verfahren hat also 0 %. Kommt ein Verfahren hinzu (z. B. RL), kann sich der Bezug ändern; die Tabelle muss dann mit allen Verfahren zusammen berechnet werden. |
| **Rechenzeit** | `runtime_s_mean` | Planen + Ausführen je Instanz, gemittelt |
| Instanzen ohne Verspätung | `instances_without_late` | Anzahl Instanzen mit `late_jobs = 0` |
| Verspätung gesamt | `total_tardiness_mean` | Summe der Verspätungen je Instanz, gemittelt |
| Selbsteinschätzung | `planned_late_share` | wie `late_share`, aber im eigenen Plan des Verfahrens |

Gezählt und gemittelt wird nur über ausführbare Lösungen (`feasible`). In allen bisherigen Läufen waren alle Lösungen ausführbar.

### Anzahl Instanzen

Je Klasse **100 Testinstanzen** (ab Seed 42) und **50 Validierungsinstanzen** (ab Seed 100.000), einstellbar in `data_splits`. Alle Verfahren laufen auf **denselben** Instanzen, sodass sich die Verfahren auch **paarweise** je Instanz vergleichen lassen, zum Beispiel statisch gegen zeitbewusst auf derselben Instanz.

### Experimente ausführen

Aus dem Projektordner, in dieser Reihenfolge:

```powershell
# 1. Instanzen erzeugen (ca. 1,5 Minuten mit 10 Prozessen, ca. 370 MB)
python generator.py --classes all --splits test val --processes 10

# 2. optional prüfen
python schedule_validator.py instances

# 3. Baselines auswerten (ca. 1 Minute)
python run_experiments.py
```

Optionen von `run_experiments.py`:

| Option | Standard | Bedeutung |
|---|---|---|
| `--classes` | `experiments.classes` | nur bestimmte Klassen, z. B. `--classes klein_11 gross_24` |
| `--split` | `experiments.split` | `test` oder `val` |
| `--limit N` | alle | nur die ersten N Instanzen je Klasse (Schnelltest) |
| `--out` | `experiments.results_file` | Ergebnis je Instanz und Verfahren |
| `--summary` | `experiments.summary_file` | Kennzahlen je Klasse und Verfahren |
| `--config` | `problem_config.json` | Einstellungen |

**Ausgabe:** Im Terminal steht die Ergebnistabelle je Klasse und Verfahren: verspätete Jobs (verspätete / alle Jobs), Anzahl Instanzen ganz ohne Verspätung, mittlere maximale Verspätung, mittlerer Makespan, Gap zum besten Makespan der Klasse und mittlere Rechenzeit. Unter jeder Klasse steht der beste Makespan, auf den sich der Gap bezieht. Beispiel (Ausschnitt):

```
Klasse     Verfahren                         n         verspaetete Jobs ohne Versp. max. Versp.  Makespan  Gap %  Zeit ms
gross_21   EDD+ECT statisch                100     272 / 2100 = 12.95 %           4        54.4     640.2   1.05     22.2
gross_21   EDD+ECT zeitbewusst             100     200 / 2100 =  9.52 %           9        45.5     634.5   0.14     21.8
gross_21   EDD+SPT statisch                100     285 / 2100 = 13.57 %           7        53.7     643.2   1.52     22.0
gross_21   EDD+SPT zeitbewusst             100     221 / 2100 = 10.52 %           9        46.0     633.6   0.00     21.5
           bester Makespan gross_21 (Referenz fuer Gap): 633.6
```

**Dateien:**
- `results/baselines.csv`: eine Zeile je Instanz und Verfahren, mit den Spalten aus der ersten Tabelle unter „Kennzahlen“ sowie `class`, `instance`, `seed`, `method`, `sequencing`, `routing`, `view`, `status`, `feasible`, `jobs`.
- `results/baselines_summary.csv`: die Ergebnistabelle, eine Zeile je Klasse und Verfahren, mit den Spalten aus der zweiten Tabelle unter „Kennzahlen“.

Beide Dateien sind CSV mit Komma als Trennzeichen und Punkt als Dezimalzeichen, also direkt mit pandas oder R lesbar. Ein zweiter Lauf überschreibt sie.

---

## RL-Umgebung

`rl/` enthält die Umgebung für den späteren RL-Agenten. Der Aufbau orientiert sich an der FJSP-Umgebung von RL4CO (`reset`, `step`, Maske, Reward, Generator, Datensätze), der Code ist aber eigenständig und braucht nur Torch. Alle Einstellungen stehen in `problem_config.json`. Das Netz beschreibt der Abschnitt [RL-Modell und Training](#rl-modell-und-training).

### Benutzung

```python
from rl.env import TDFJSPEnv

env = TDFJSPEnv()                               # liest problem_config.json
td = env.reset(batch_size=4)                    # neue Instanzen on the fly
while not td["done"].all():
    td["action"] = policy(td)                   # Auswahl aus td["action_mask"]
    td = env.step(td)["next"]
env.get_objectives(td)                          # Verspätung, Makespan ... je Instanz
env.get_reward(td)                              # Endreward je Instanz
```

Der Zustand `td` ist ein dict aus Tensoren. Formen: B = Batch, J = Jobs, O = Operationen, R = Ressourcen (erst Handarbeitsplätze, dann Maschinen). Ein Batch enthält Instanzen gleicher Größe; zwischen Batches darf sich die Größe ändern.

### Aktionen und Ablauf

| Aktion | Bedeutung |
|---|---|
| `0` | warten bis zum nächsten Ereignis (nur mit `allow_wait`, solange noch ein Ereignis kommt); fertige Instanzen geben immer 0 |
| `1 + Job * R + Ressource` | die nächste Operation des Jobs startet **jetzt** auf der Ressource |

Die Uhr ist ereignisgesteuert wie in `SimulationState`: Ressource wird frei, Job wird bereit, Zeitfenster beginnt. Kann nichts starten, springt die Umgebung selbst weiter. Die Episode endet, wenn alles eingeplant ist. Reicht der Horizont nicht, endet sie mit `truncated`.

**Warten** ist standardmäßig erlaubt, damit der Agent warten kann, wenn es besser ist. Das kann sich lohnen: Eine Handoperation mit 40 Minuten Basisdauer ist bei Start um 13:55 um 14:59 fertig (Faktor 1,6), bei Start um 14:00 schon um 14:52 (Faktor 1,3). Mit `allow_wait: false` hat der Agent dieselbe Entscheidungsmenge wie die Regeln (non-delay).

### Funktionen

| Funktion | Aufgabe |
|---|---|
| `reset(td, batch_size)` | Startzustand aus Instanzdaten oder neuen Instanzen |
| `step(td)` | führt `td["action"]` aus; gibt `{"next": neuer Zustand}` zurück; der alte Zustand bleibt unverändert |
| `get_action_mask(td)` | erlaubte Aktionen (B, 1 + J·R), identisch zu `SimulationState.options` |
| `_make_step` | Operation einplanen: Rüstung und Bearbeitung mit dem jeweils gültigen Faktor |
| `_advance_until_decision`, `_next_event` | Uhr weiterstellen, Abbruch am Horizont |
| `_update_options` | Dauern und Zulässigkeit aller Paare aus Job und Ressource |
| `_get_features` | Merkmale für den Agenten |
| `get_reward(td)`, `td["reward"]` | Endreward bzw. Reward des letzten Schritts |
| `get_objectives(td)` | Zielgrößen je Instanz |
| `to_schedule(td, i, instance)` | Plan im Schedule-Format des Projekts |
| `check_solution_validity(td, instances)` | prüft Pläne mit dem unabhängigen Validator |
| `render(td, i, instance)` | Gantt-Diagramm |
| `dataset(n, phase, class_name)` | `torch`-Dataset: `train` neu erzeugt, `val`/`test` aus Dateien |

### Merkmale für den Agenten

Alle als `float32`, Zeiten geteilt durch `time_scale_minutes`. Die Namen stehen in `rl.env.FEATURE_NAMES`.

| Schlüssel | Form | Spalten |
|---|---|---|
| `time_features` | (B, 7) | Uhrzeit im Tag, Tag, Faktor jetzt, nächster Faktor, Minuten bis zum nächsten Fenster, zur Pause, zum Tagesende |
| `op_features` | (B, O, 10) | mittlere Basisdauer, Dauer bei Start jetzt, manuell?, nächste Operation?, bereit?, eingeplant?, Anteil zulässiger Ressourcen, Position im Job, Puffer des Jobs bis zur Deadline, Restarbeit des Jobs |
| `resource_features` | (B, R, 4) | belegt für, Handarbeitsplatz?, Anteil offener Operationen, die hier laufen dürfen, hat Vorgänger? |
| `pair_features` | (B, J, R, 5) | Rüstung, Bearbeitung und Summe bei Start jetzt, zulässig?, jetzt möglich? |
| `ops_ma_adj` | (B, R, O) | offene Kanten zwischen Ressourcen und Operationen |

Der Referenz-Plan bleibt dem Agenten verborgen. In der Sicht `static` sieht er nur Faktor 1.0, Pause und Arbeitsende aber schon.

### Reward: erst Verspätung, dann Makespan

Standard ist eine exakte Hierarchie (`reward.levels`):

```text
Kosten = Summe der Verspätungen × (Horizont + 1) + Makespan
Reward = −Kosten / ((Horizont + 1) × Jobs × scale_minutes)
```

Der Makespan kann höchstens den Horizont erreichen. Eine Minute Verspätung wiegt also immer mehr als jeder Makespan-Unterschied; der Makespan entscheidet nur bei gleicher Verspätung. Die Kosten stehen als `float64` im Zustand, weil `float32` hier zu ungenau wäre.

Im Modus `stepwise` ist der Reward je Schritt die Änderung der geschätzten Kosten. Die Schätzung nutzt für jeden Job die Restoperationen mit Faktoren, Pause und Arbeitsende (`rl.reward.estimate_completion`); am Ende gilt die echte Fertigstellung. In der Summe bleibt genau `(Startschätzung − echte Kosten) / Teiler`, Schätzfehler verfälschen das Ziel also nicht. Im Modus `terminal` gibt es nur am Ende `−Kosten / Teiler`.

Reicht der Horizont nicht, kostet jede offene Operation `truncation_penalty_minutes` zusätzliche Verspätung.

**Hinweis fürs Training:** Durch die exakte Hierarchie ist das Makespan-Signal sehr klein, solange Verspätungen vorkommen. Wenn das später stört, lässt es sich über `level_weights` abschwächen, zum Beispiel `[60, 1]`: Eine Minute Verspätung wiegt dann 60 Minuten Makespan.

### Instanzen

| Phase | Quelle | Klasse |
|---|---|---|
| Training | `rl.generators.TDFJSPGenerator`: neue Instanzen, Seeds ab `data_splits.train.seed_start`; nutzt `generator.generate_class_instances` | `train_classes`, wechselt je Batch |
| Validierung | `TDFJSPFileGenerator`, `instances/val/<Klasse>` | je Klasse |
| Test | `TDFJSPFileGenerator`, `instances/test/<Klasse>`, dieselben Dateien wie für die Regeln | je Klasse |

Dateien werden beim Laden gegen die Config geprüft; eine Datei mit altem Fingerprint führt zu einer Fehlermeldung.

### Auswerten

`rl/evaluate.py` wertet Policies auf denselben Instanzen und mit derselben Bewertung aus wie die Regeln: Der Plan läuft durch `execute_plan` und `evaluate`, und alle Verfahren landen in einer Tabelle.

```powershell
python -m rl.evaluate --policies edd_ect random                     # alle Klassen, mit Regeln
python -m rl.evaluate --policies random --classes klein_11 --split val --no-rules
python -m rl.evaluate --policies edd_ect --classes klein_11 --limit 1 --show
python -m rl.evaluate --policies --checkpoints experiments/checkpoints/A3_zeit_warten_ppo_seed0/best.pt experiments/checkpoints/A1_zeit_warten_ppo_seed0/best.pt
```

Vorhanden sind `edd_ect`, `edd_spt` und `random`, dazu trainierte Modelle über `--checkpoints`. Eine Policy ist eine Funktion `policy(td) -> Aktionen`.

**Freiwilliges Warten:** `execute_plan` startet jede Operation zum frühesten Zeitpunkt und würde das Warten zurücknehmen. Hat eine Policy gewartet und plant sie mit den echten Dauern, gilt ihr Plan deshalb unverändert. Pläne ohne Warten laufen wie bei den Regeln durch `execute_plan`.

### Tests

```powershell
python -m unittest tests.test_rl_env                        # ca. 1 Minute
$env:RL_FULL_TEST = "1"; python -m unittest tests.test_rl_env   # zusätzlich alle Testinstanzen
```

| Test | Gegen |
|---|---|
| Dauer und Zulässigkeit für alle Arten, Basisdauern und Minuten, auch mit geänderten Regeln | `schedule_validator.duration` |
| Maske, Uhrzeit und Dauern in jedem Zustand zufälliger Läufe, mit und ohne Warten, in beiden Sichten | `SimulationState` |
| EDD+ECT und EDD+SPT ergeben exakt dieselben Pläne, in beiden Sichten | `baselines.dispatcher.build_plan` |
| Jeder Plan ist zulässig, Zielgrößen stimmen | `check_schedule`, `evaluate` |
| Summe der Rewards = Startschätzung − echte Kosten; Hierarchie exakt lexikografisch | eigene Rechnung |
| Batch = Einzelläufe; alte Zustände bleiben unverändert | eigene Läufe |
| Generator: gleiche Config gibt gleiche Instanzen, unabhängig von der Prozesszahl; Seeds getrennt | `generate_class_instances` |
| Ungültige Einstellungen werden abgelehnt; geänderte Einstellungen wirken | `check_config` |

Die GPU-Variante läuft nur, wenn eine GPU vorhanden ist, und vergleicht deren Pläne mit denen der CPU.

### Geschwindigkeit

Grob gemessen auf der CPU (10 Torch-Threads, Merkmale und Stepwise-Reward an). Die Werte schwanken zwischen Läufen deutlich, sie sind also nur Größenordnungen:

| Batch | Zeit je Schritt | Zeit je Episode und Instanz |
|---|---|---|
| 1 | 4–10 ms | 0,3–1,1 s |
| 32 | 23–35 ms | 40–130 ms |
| 128 | 25–79 ms | 11–70 ms |

Die Umgebung ist auf Batches ausgelegt. Für einzelne Instanzen ist sie deutlich langsamer als `SimulationState` (EDD+ECT dort 3–28 ms je Instanz); die Rechenzeit eines Agenten in der Ergebnistabelle hängt deshalb auch von `evaluation_batch_size` ab.

Neue Instanzen on the fly (Batch 32): 4–19 s in einem Prozess, 1–4 s mit 10 Prozessen (`generator_workers`). Mit `generator_workers > 0` muss das aufrufende Skript seinen Code unter `if __name__ == "__main__":` haben, sonst starten die Worker ihn erneut.

### Auf dem Uni-Cluster

1. `problem_config.json`, Code und `requirements.txt` kopieren. Torch passend zur CUDA-Version des Clusters installieren.
2. In der Config `generator_workers` auf die Zahl der CPU-Kerne des Jobs setzen; `device` bleibt `auto`.
3. Instanzen dort erzeugen (`python generator.py --classes all --splits test val --processes N`) oder den Ordner `instances/` kopieren. Pfade gelten relativ zur Config.
4. `python -m unittest tests.test_rl_env` ausführen. Dabei läuft auch der GPU-Vergleich. Die Umgebung ist bisher nur auf der CPU getestet.

---

## RL-Modell und Training

Das Netz ist ein heterogenes Graph-Netz nach dem Vorbild von HGNN (Song et al. 2022) und dem L2D-Aufbau in RL4CO, erweitert um zeitabhängige Kanten nach NCO4TDTSP (Yang & Fan 2025). Der Encoder läuft je Instanz einmal; alles Dynamische sieht der Decoder.

### Tensoren an den Schnittstellen

B = Batch, O = Operationen (mit Padding), J = Jobs, R = Ressourcen (Handarbeitsplätze und Maschinen; im Modell die Maschinenknoten), K = Zeitfenster eines Tages, d = `d_model`, H = `num_heads`.

| Schnittstelle | Tensor | Form |
|---|---|---|
| Encoder ein | `proc_profiles` | [B, O, R, K] |
| | `proc_ref` | [B, O, R] |
| | `profile_lengths`, `profile_starts` | [K] |
| | `eligible` | [B, O, R] |
| | `precedence` | [B, O, O] |
| | `pad_mask` | [B, O] |
| | `op_features` | [B, O, 10] |
| | `machine_features` | [B, R, 5] |
| Kanten im Encoder | statisch / zeitabhängig | [B, O, R, 1] / [B, O, R, 1 + 2K] |
| Attention je Richtung | `e_om`, `alpha_om` | [B, O, R, H] |
| Encoder aus | `h_op`, `h_ma` | [B, O, d], [B, R, d] |
| Decoder ein | `next_op` | [B, J] |
| | `candidates`, `candidate_valid` | [B, J, R, 9], [B, J, R] |
| | `time_features`, `state_features` | [B, 7], [B, 7] |
| | `action_mask` | [B, 1 + J·R] |
| Decoder aus | `logits` | [B, 1 + J·R] (Position 0 = WAIT) |
| Critic aus | `value` | [B] |

**Die K Zeitfenster** sind die Fenster aus `time_model.day_windows`, also derzeit K = 7:

| k | Beginn (`profile_starts`, Minuten) | Uhrzeit | Länge (`profile_lengths` × 720) | Faktor |
|---|---|---|---|---|
| 0 | 0 | 06:00 | 180 | 1,0 |
| 1 | 180 | 09:00 | 60 | 1,3 |
| 2 | 240 | 10:00 | 120 | 1,2 |
| 3 | 360 | 12:00 | 30 | Pause |
| 4 | 390 | 12:30 | 90 | 1,6 |
| 5 | 480 | 14:00 | 120 | 1,3 |
| 6 | 600 | 16:00 | 120 | 1,1 |

Der Faktor gilt im ganzen Fenster und jeden Tag gleich; ein Eintrag beschreibt also jeden Start in diesem Fenster. Die Fenster sind ungleich lang, deshalb gehen ihre Längen als eigene Merkmale in die Kanten ein.

**Normalisierung:** `proc_profiles` = Dauer bei Start im Fenster / Basisdauer. 0 heißt: nicht zulässig oder in diesem Fenster nicht startbar. Die absolute Dauer ergibt sich aus `proc_profiles × proc_ref × time_scale_minutes`. Alle Zeiten sind durch `time_scale_minutes` geteilt.

### Encoder (`rl/encoder.py`)

Je Schicht und Richtung (Ressource → Operation und Operation → Ressource, mit getrennten Parametern):

```text
e        = dot(W_Q h_ziel, W_K h_quelle) / sqrt(d_head)
e_temp   = AttentionMLP(concat(e, Kante))
alpha    = masked_softmax(e_temp) über die Nachbarn
message  = W_V h_quelle + EdgeValueMLP(Kante)
aggregat = sum(alpha * message)
```

Die Operationen bekommen zusätzlich die Einbettungen von Vorgänger und Nachfolger im Job. Die Aktualisierung nutzt Residual-Verbindung, MLP und LayerNorm. Knoten ohne Nachbarn erhalten 0 statt NaN, Padding-Operationen bleiben 0 und ändern die übrigen Einbettungen nicht.

### Decoder (`rl/decoder.py`)

Für jedes Paar aus Job und Ressource bewertet der Decoder die nächste Operation des Jobs:

| Merkmal in `candidates` | Bedeutung |
|---|---|
| `setup_start_offset` | frühester zulässiger Rüstbeginn minus jetzt |
| `setup_time` | Rüstdauer an diesem Beginn |
| `processing_start_offset` | Bearbeitungsbeginn direkt nach der Rüstung minus jetzt |
| `processing_time` | Dauer mit dem Faktor am Bearbeitungsbeginn |
| `finish_offset` | Ende minus jetzt |
| `job_slack` | Deadline minus Ende minus geschätzte Restarbeit des Jobs |
| `due_in` | Deadline minus jetzt |
| `feasible_now` | Paar steht in der Aktionsmaske |
| `wait_gain` | Ende bei Start ab der nächsten Fenstergrenze minus Ende bei Start jetzt; negativ heißt: Warten lohnt sich |

Rüstung und Bearbeitung werden an ihren eigenen Startzeitpunkten bewertet (`rl.observation.first_feasible`, im Test identisch zu `simulator.first_feasible_start`). Dazu kommt ein Produkt aus Knoten und Zeit nach NCO4TDTSP (Gl. 8): Erst damit lernt das Netz, welcher Teil eines Zeitprofils zur aktuellen Uhrzeit zählt. Die Uhrzeit ist zusätzlich sinusförmig eingebettet. WAIT bewertet ein eigenes MLP aus den gemittelten Paaren, der Zeit und dem Zustand.

### Varianten

| Variante | Encoder | Decoder |
|---|---|---|
| `A0` | statisch: Kante = Basisdauer | `StaticPairDecoder`: nur Operation und Ressource, WAIT als fester Wert |
| `A1` | statisch: Kante = Basisdauer | `TimeAwareDecoder` |
| `A2` | Zeitprofile in der Attention, keine Edge-Values | `TimeAwareDecoder` |
| `A3` | Zeitprofile in Attention und Edge-Values | `TimeAwareDecoder` |

Unabhängig davon wählbar:
- **Warten:** `environment.allow_wait` oder `--no-wait`.
- **Agent ohne Zeitinformation:** `model.time_information = "none"` oder `--no-time-information`. Der Agent rechnet alle Dauern mit Faktor 1.0, kennt aber Pause und Arbeitsende wie die statischen Regeln. Uhrzeit und Zeitfaktoren bekommt er nicht, die absolute Deadline auch nicht. Die Umgebung bleibt echt: Maske, Reward und Ablauf nutzen die echten Dauern. Völlig blind ist er dadurch nicht: Die Maske und die tatsächliche Belegung der Ressourcen spiegeln die echten Dauern, und der Fortschritt lässt die Uhrzeit grob erahnen.
- **Statische Welt:** `environment.view = "static"` lässt zusätzlich die ganze Umgebung mit Faktor 1.0 rechnen, wie bei den statischen Regeln. Die Auswertung in `rl.evaluate` führt den Plan dann mit echten Dauern aus.

### Abgleich mit Song et al. (2022)

| Baustein | Paper | Hier | Grund |
|---|---|---|---|
| Reihenfolge in einer Schicht | Ressourcen, dann Operationen mit den neuen Ressourcen (Abb. 3, Gl. 7) | beide gleichzeitig aus der vorigen Schicht | Entscheidung, siehe unten |
| Operationen aktualisieren | fünf MLPs mit ELU (Gl. 7) | lineare Projektion für Vorgänger und Nachfolger, ein gemeinsames MLP, dazu Residual und LayerNorm | Entscheidung, siehe unten |
| Paar-Bewertung | MLP über Operation, Ressource, h_t, zwei verdeckte Schichten, tanh (Gl. 9) | ohne h_t, eine verdeckte Schicht, ReLU; beim zeitbewussten Decoder dazu Zeit und Kandidatenmerkmale | Entscheidung, siehe unten; Auftrag, Schritt 8 |
| Kantenmerkmale | in allen Schichten | gleich | |
| PPO | Clip 0,2, Value 0,5, Entropie 0,01, 3 Epochen, γ = 1, lr 2·10⁻⁴ | gleich | |
| Reward | Änderung der geschätzten Kosten, summiert sich zum Ziel | gleich, Ziel hierarchisch | Projektziel |
| Attention | additiv mit LeakyReLU, mit Selbst-Koeffizient, nur Operation → Maschine (Gl. 4–6) | Skalarprodukt mit Zeitprofil-MLP, in beiden Richtungen, Mehrkopf | Auftrag, Schritte 3–5 |
| Maschinen in Stufe 2 | Summe über die Nachbarn | Attention-Aggregat | Auftrag, Schritt 4 |
| **Encoder** | **läuft in jedem Schritt** auf dem aktuellen Graphen, geplante O-M-Kanten fallen weg, dynamische Merkmale | **läuft je Instanz einmal**, nur statische Merkmale; alles Dynamische im Decoder | Auftrag, Schritte 3 und 6 |
| Critic | MLP über h_t, zwei verdeckte Schichten, tanh | dazu Uhrzeit und Zustand, eine verdeckte Schicht, ReLU | Auftrag, Schritt 10; Entscheidung, siehe unten |
| Größen | L = 2, d = 8, dh = 128, dπ = dφ = 64 | L = 3, d = 128, H = 4, Breite 128 | Auftrag, Schritt 3; alles in der Config |
| Neue Instanzen | alle 20 Iterationen | jede Iteration | ersetzen wäre schneller, siehe Grenzen |
| Validierung | 100 Instanzen alle 10 Iterationen | `validation_instances` je Klasse alle `validate_every` Iterationen | Config |
| Sampling-Auswertung (DRL-S) | beste aus 100 Stichproben | nur greedy oder einzelne Stichprobe | noch nicht gebaut |
| Vergleichsregeln | FIFO, MOR, MWKR u. a. | EDD+ECT, EDD+SPT | Ziel ist zuerst Verspätung |

**Entscheidung:** Die Angleichung an Gl. 7 und Gl. 9 samt tanh-MLPs wurde gebaut und wieder zurückgenommen. Damit schwankte die Validierung im Lauf mit Zeitinformation deutlich stärker (`klein_11`, 80 Iterationen). Das beruht auf einem einzigen Seed; welcher Teil der Angleichung die Ursache war, ist nicht untersucht.

**Folge für `A0`:** `A0` ist nicht das HGNN des Papers. Der Encoder läuft nur einmal und `A0` sieht keinen dynamischen Zustand; als Vergleich würde es das HGNN also unterschätzen. Als faire Referenz bräuchte es eine zusätzliche Variante mit Encoder je Schritt.

### Training

```powershell
python -m rl.train                                      # Einstellungen aus der Config
python -m rl.train --variant A1 --seed 1 --iterations 500
python -m rl.train --variant A3 --no-time-information
python -m rl.train --variant A3 --no-wait --classes mittel_16 gross_21
python -m rl.train --resume experiments/checkpoints/A3_zeit_warten_ppo_seed0/latest.pt --iterations 1000
python -m rl.evaluate --policies --checkpoints experiments/checkpoints/A3_zeit_warten_ppo_seed0/best.pt --classes klein_11
```

Jeder Lauf schreibt nach `experiments/checkpoints/<Variante>_<zeit|ohnezeit>_<warten|nondelay>_ppo_seed<n>/`: `config.json`, `training_log.csv` (eine Zeile je Iteration), `latest.pt` und `best.pt` (bester Wert von `training.best_metric`). Checkpoints aus der Zeit vor dem Monitoring lassen sich nicht laden, weil ihrer Config der Abschnitt `monitoring` fehlt. Gleiche Einstellungen und gleicher Seed ergeben im Test dasselbe Log. Kurven und Auswertung beschreibt [Experiment-Tracking und Visualisierung](#experiment-tracking-und-visualisierung). Für `generator_workers > 0` hat `rl/train.py` den nötigen Schutz durch `__main__`.

PPO nutzt Returns mit GAE und γ = 1. Der Encoder läuft beim Sammeln einmal je Instanz und beim Lernen mit Gradienten für die Instanzen jedes Minibatches.

### Tests

```powershell
python -m unittest tests.test_model         # ca. 5 Minuten
python -m unittest tests.test_monitoring    # ca. 1 Minute
python -m tests.synthetic_learning          # ganze Tabelle des Lerntests, ca. 3 Minuten
```

Der Lerntest prüft die Architektur ohne PPO: Zwei Maschinen haben dieselbe mittlere Dauer, aber je nach Zeitfenster ist eine andere günstiger. Ohne die Dauer im Decoder muss die Information durch den Encoder kommen.

| Muster | A0 | A1 | A2 | A3 | A1–A3 mit Dauer im Decoder |
|---|---|---|---|---|---|
| eine Maschine vormittags, die andere nachmittags günstiger | 0,49 | 0,49 | 0,49 | **1,00** | 1,00 |
| je Fenster zufällig | 0,51 | 0,51 | 0,51 | **1,00** | 1,00 |

A2 kann das Profil in diesem Test nicht weitergeben, weil Attention bei gleichen Nachrichten nichts unterscheidet. Das Beispiel ist bewusst so gebaut.

### Was das Zeitprofil in deinen Instanzen leisten kann

Der Faktor hängt derzeit nur von der Uhrzeit und der Art der Operation ab. Alle Handoperationen haben bis auf das Runden dasselbe relative Profil, automatische Operationen sind immer 1. Anders als im Lerntest und bei NCO4TDTSP unterscheiden sich also keine zwei Maschinen darin, *wann* sie günstig sind. Außerdem sieht der Decoder die Dauer am tatsächlichen Start direkt. Der Abstand zwischen A1 und A3 kann auf diesen Instanzen deshalb klein bleiben; das ist kein Fehler der Architektur. Mehr Spielraum gäbe es mit Faktoren je Worker oder Maschine.

### Stand der Prüfung

| Baustein | Geprüft | Noch offen |
|---|---|---|
| Umgebung, Maske, WAIT, Dauern | auf allen Test- und Validierungsinstanzen gegen Validator und Simulator | GPU |
| Beobachtung, Encoder, Decoder | Formen, Maskierung, Padding, Gradienten, Permutation, Lerntest | Padding nur synthetisch; die Umgebung erzeugt noch keine gemischten Batches |
| Agent ohne Zeitinformation | Eingaben und Umgebung getestet | Vergleich auf vollen Läufen |
| PPO | Reproduzierbarkeit, Checkpoints, kurzer Lauf | Konvergenz, Parameterwahl, mehrere Seeds |
| Varianten A0–A3 | laufen und erzeugen zulässige Pläne | Vergleich der Qualität auf vollständigen Instanzen |

### Herkunft

RL4CO ist nur Vorbild, sein Code ist nicht im Projekt. Übernommen sind die Ideen: Aktionsraum aus Job und Maschine plus WAIT, ein Encoder für Operationen und Maschinen, Decoder aus MLP über beide Einbettungen. Neu sind Mehrkopf-Attention, Zeitprofile in Attention und Nachrichten, Präzedenz in der Aktualisierung (RL4CO nutzt nur eine Positionskodierung), der zeitbewusste Decoder, die Rüstmerkmale und PPO mit echten Returns (`StepwisePPO` von RL4CO lernt nur den Reward des einzelnen Schritts). `A0` ist deshalb eine eigene statische Variante desselben Aufbaus und nicht der RL4CO-Code selbst.

---

## Experiment-Tracking und Visualisierung

Läuft lokal mit TensorBoard, Pandas und Matplotlib, ohne Cloud-Dienst und ohne Konto. Das Paket `monitoring/` kennt weder den Trainingscode noch RL4CO oder Lightning; es funktioniert also auch mit `A0` oder einer anderen Baseline. Mit `monitoring.enabled = false` oder `--no-monitoring` läuft das Training unverändert und liefert im Test dasselbe Ergebnis.

### Starten und im Browser ansehen

```powershell
python -m pip install -r requirements.txt
python -m rl.train --variant A3 --seed 42

# in einem zweiten Terminal, aus dem Projektordner:
tensorboard --logdir experiments/tensorboard --host 127.0.0.1 --port 6006
```

Dann im Browser `http://localhost:6006` öffnen. Das Training schreibt neue Werte spätestens alle `monitoring.flush_seconds` (30 s). TensorBoard lädt sie mit dem Pfeil oben rechts neu, oder automatisch, wenn im Zahnrad daneben **Reload data** aktiv ist. Gantt-Diagramme stehen unter **Images**, die Verteilung der Startzeiten unter **Histograms** und **Distributions**.

**Auf dem Cluster oder einer VM:** TensorBoard dort wie oben starten und vom eigenen Rechner aus den Port weiterleiten:

```powershell
ssh -L 6006:127.0.0.1:6006 benutzer@rechner
```

Danach öffnet man lokal ebenfalls `http://localhost:6006`. Der Port bleibt so nur über SSH erreichbar. Vor einem SLURM-Job den Rechenknoten als Ziel einsetzen, zum Beispiel `ssh -L 6006:knoten12:6006 benutzer@login`, falls TensorBoard dort läuft.

### Varianten und Seeds vergleichen

- **In TensorBoard:** Jeder Lauf hat einen eigenen Ordner unter `experiments/tensorboard` und erscheint links als eigene Zeile. Im Feld **Filter runs (regex)** auswählen: `A3_` alle Seeds von A3, `seed42$` alle Varianten mit Seed 42, `A[13]_zeit_warten` A1 und A3 mit Zeitinformation. Alle Kurven eines Tags liegen dann im selben Diagramm; `val/reward_mean` zeigt so den direkten Vergleich. Die x-Achse **Step** zählt Optimizer-Schritte; **Relative** zeigt die Zeit seit dem Start. Mehrere Läufe dürfen gleichzeitig schreiben.
- **Auf denselben Testinstanzen, mit Mittel über Seeds:** `python -m monitoring.comparison_plotter` zeichnet die zehn Vergleichsdiagramme aus den gespeicherten Rohdaten nach `experiments/figures/comparison/` und nimmt die Regeln aus `results/baselines.csv` dazu. Einzelne Läufe wählt `--runs`. `status.csv` nennt je Diagramm, ob es entstanden ist oder was fehlt; nichts wird geschätzt.

### Was wo liegt

```text
experiments/
  tensorboard/<run_id>/                     Ereignisdateien
  results/<run_id>/
    run.json                                Variante, Seed, Hyperparameter, Encoder, Decoder, Algorithmus, Ziel,
                                            Generator, Zeitprofile, Commit, Start, Laufzeit, bestes Modell,
                                            nicht verfügbare Kennzahlen
    config.json                             vollständige Einstellungen
    scalars.csv                             step, tag, value, elapsed_seconds: alle Kurven als Tabelle
    train_instances.csv, val_instances.csv,
    test_instances.csv                      eine Zeile je Instanz, getrennt nach Datensatz
    val_static_reference_instances.csv      die statische Referenzregel auf den Validierungsinstanzen
  figures/<run_id>/gantt/                   Gantt-Diagramme als PNG und PDF
  figures/comparison/                       Vergleichsdiagramme
  checkpoints/<run_id>/                     latest.pt, best.pt, training_log.csv, config.json
```

Die **Lauf-ID** besteht aus Variante, Zeitinformation, Warten, Algorithmus und Seed, zum Beispiel `A3_zeit_warten_ppo_seed42`. Wiederholungen bekommen `_r2`, `_r3`, ... Der Name wird atomar vergeben, gleichzeitig gestartete Läufe kommen sich also nicht in die Quere. Dieses Projekt ist kein Git-Repository; `git_commit` in `run.json` steht deshalb auf `null`, bis es eines wird.

### Kennzahlen

Die x-Achse ist der **globale Schritt** (Anzahl Optimizer-Schritte). Eine Iteration umfasst viele Schritte; `train/*` erscheint deshalb frühestens am Ende einer Iteration.

| Tag | Bedeutung und Grundlage |
|---|---|
| `train/reward_mean`, `reward_std` | `env.get_reward` je Episode, Mittel und Streuung über alle Episoden seit dem letzten Schreiben. Dieselbe Definition wie bei `val/`: −Kosten / Teiler, nicht die Summe der Schritt-Rewards |
| `train/makespan_mean`, `tardiness_mean` | `env.get_objectives` je Episode |
| `train/window_compliance` | Anteil pünktlicher Jobs über alle Jobs (siehe unten) |
| `train/feasibility_rate` | Anteil Episoden, die nicht am Horizont abgebrochen sind. Pläne der Umgebung halten die harten Regeln ein; der Validator läuft erst in `val/` und `test/` |
| `train/episode_length` | Entscheidungen je Episode, WAIT eingeschlossen |
| `train/wait_share` | WAIT-Anteil an den Entscheidungen |
| `train/learning_rate` | aktueller Wert des Optimizers |
| `train/processed_instances`, `elapsed_seconds` | Instanzen bisher, Trainingszeit |
| `ppo/policy_loss`, `value_loss` | die Verlustwerte aus `ppo_update`, gewichtet mit der Minibatch-Größe |
| `ppo/entropy` | Entropie der maskierten Verteilung. Gesperrte Aktionen haben Wahrscheinlichkeit 0 und zählen nicht; im Test gleich der Entropie nur über die erlaubten Aktionen |
| `ppo/approx_kl` | Mittel von `(r − 1) − log r` mit r = neue / alte Wahrscheinlichkeit |
| `ppo/clip_fraction` | Anteil mit `abs(r − 1) > clip_range` |
| `ppo/grad_norm` | Rückgabe von `clip_grad_norm_`, also die Norm **vor** dem Kappen, Mittel je Optimizer-Schritt |
| `ppo/explained_variance` | `1 − Var(Return − Wert) / Var(Return)` mit den Werten aus der Datensammlung, je Iteration; fehlt, wenn die Returns nicht streuen |
| `val/*`, `test/*` | siehe nächster Abschnitt |
| `temporal/wait_actions_mean`, `wait_share` | gewählte WAIT-Aktionen je Instanz und Anteil |
| `temporal/voluntary_idle_time_mean` | freiwilliger Leerlauf in Ressourcen-Minuten: Die Ressource hätte bei WAIT anfangen können. Solange etwas starten könnte, geht die Uhr nur durch WAIT weiter; diese Abgrenzung ist also exakt |
| `temporal/unavoidable_idle_time_mean` | der übrige Leerlauf bis zum Makespan: Auf der Ressource konnte in dem Moment nichts starten, auch in der Pause und nach ihrer letzten Operation. Das sagt nichts darüber, ob frühere Entscheidungen ihn vermeidbar gemacht hätten |
| `temporal/processing_time_total_mean`, `setup_time_total_mean` | tatsächliche Bearbeitungs- und Rüstzeit je Instanz |
| `temporal/mean_processing_factor` | mittlerer Zeitfaktor am Start, nur zeitabhängige Operationen (derzeit die manuellen) |
| `temporal/mean_setup_factor` | mittlerer Zeitfaktor am Rüstbeginn |
| `temporal/start_times`, `start_share/<Fenster>` | Startzeiten als Histogramm über den Horizont und als Anteil je Tagesfenster |
| `temporal/temporal_gain` | `(Makespan statisch − Makespan zeitbewusst) / Makespan statisch × 100` gegen `monitoring.static_reference`, gemittelt über dieselben Instanzen; beide Seiten mit den echten Dauern. Statisch heißt hier: statisch geplant, nicht in einer statischen Umgebung |
| `temporal/tardiness_gain`, `compliance_gain` | dasselbe für die Verspätung und die Pünktlichkeit; bei der Pünktlichkeit ist höher besser |
| `temporal/temporal_variability_mean` | numerische Zeitabhängigkeit, siehe unten |

**`window_compliance`:** Das Modell kennt Deadlines, aber keine Abholfenster. Die Kennzahl heißt so wie im Auftrag, misst aber den **Anteil pünktlicher Jobs** über alle Jobs aller Instanzen, also `1 − late_share` aus der Ergebnistabelle der Regeln.

**Zeitabhängigkeit einer Instanz:** `temporal_variability` = Anteil zeitabhängiger Basisarbeit mal Variationskoeffizient der Fensterfaktoren. Die Zahl beschreibt nur die rechnerische Streuung der Dauern. Weil die Faktoren derzeit nur von der Uhrzeit abhängen, unterscheiden sich Instanzen darin wenig. **Entscheidungsrelevanz** (ändert die Zeitabhängigkeit, welche Entscheidung die beste ist?) bräuchte eine verlässliche Referenz, etwa bewiesen optimale Pläne. Die gibt es nicht; `temporal/decision_relevance` steht deshalb mit Begründung unter `unavailable_metrics` in `run.json`.

### Validierung und bestes Modell

Die Validierung läuft auf festen Instanzen: den ersten `training.validation_instances` jeder Klasse in Dateireihenfolge, immer denselben, greedy (`training.validation_decode`), ohne Gradienten und ohne den Zufallszustand des Trainings zu berühren. `feasible` prüft der unabhängige Validator. `inference_time` ist die Laufzeit der Episoden geteilt durch die Instanzen im Batch; `inference_batch_size` steht daneben, denn nur gleiche Batch-Größen sind vergleichbar. Die Testauswertung läuft deshalb in `experiments.evaluation_batch_size`.

**Primäre Kennzahl** für `best.pt` ist `training.best_metric`, Standard `val/reward_mean`. Dieser Reward bildet das hierarchische Ziel ab: erst Verspätung, dann Makespan.

Nach dem Training wird `best.pt` auf dem ganzen Testdatensatz ausgewertet (`monitoring.test_after_training`).

### Rohdaten je Instanz

Spalten in `*_instances.csv`: `run_id, model, seed, step, instance_id, instance_size, temporal_variability, reward, makespan, total_tardiness, window_compliance, feasible, processing_time_total, setup_time_total, voluntary_idle_time, wait_actions, inference_time`, dahinter unter anderem `instance_class, instance_seed, jobs, resources, env_view, reference_makespan, max_tardiness, late_jobs, truncated, decisions, inference_batch_size, unavoidable_idle_time, mean_processing_factor, mean_setup_factor, start_share_<Fenster>`.

`instance_id` ist `<Klasse>/<Dateiname>` und über alle Modelle gleich. **Leere Zelle = nicht verfügbar**, zum Beispiel alle Plan-Kennzahlen bei abgebrochenen Episoden oder `voluntary_idle_time` bei Regeln. `seed` ist der Seed des Laufs, `instance_seed` der der Instanz. Trainingsinstanzen heißen `<Klasse>/seed_<Seed>`; dort fehlen die Kennzahlen, die den fertigen Plan brauchen.

### Vergleichsdiagramme

| Datei | Inhalt | Grundlage |
|---|---|---|
| `01_val_reward` | Validierungs-Reward über die Schritte | `scalars.csv` |
| `02_val_makespan` | Makespan über die Schritte | `scalars.csv` |
| `03_val_tardiness_compliance` | Verspätung und Pünktlichkeit über die Schritte | `scalars.csv` |
| `04_variants` | Varianten auf denselben Testinstanzen | nur Instanzen, die alle Modelle haben |
| `05_reference_gap` | **Reference Gap** Makespan | Bezug: bester bekannter Wert je Instanz aus allen Verfahren und dem Referenz-Plan; nicht als optimal bewiesen |
| `06_gain_vs_variability` | `temporal_gain` gegen `temporal_variability` | je Instanz, gegen `--static` |
| `07_waiting` | WAIT-Anteil und freiwilliger Leerlauf | Testzeilen |
| `08_inference_time` | Rechenzeit gegen Instanzgröße | nur eine Batch-Größe |
| `09_distributions` | Boxplots über die Testinstanzen | nur gemeinsame Instanzen |
| `10_seeds` | Mittel und Standardabweichung über die Seeds | ab zwei Seeds |

Kurven werden je Validierung ausgerichtet: Die Schrittzahl einer Iteration hängt von den Episoden ab, sie unterscheidet sich also zwischen den Seeds leicht. Zeilen aus einer Umgebung mit `environment.view = "static"` zählen in keinem Vergleich (`env_view`). Als Bezug für `temporal_gain` taugen damit zum Beispiel `EDD+ECT statisch` oder ein Modell mit `--no-time-information`, beide mit echten Dauern ausgeführt.

### Fortsetzen nach einem Abbruch

```powershell
python -m rl.train --resume experiments/checkpoints/A3_zeit_warten_ppo_seed42/latest.pt --iterations 1000
```

`latest.pt` entsteht bei jeder Validierung und enthält Modell, Optimizer, Schritt, Iteration, Zufallszustand, den nächsten Instanz-Seed und die Einstellungen. `--iterations` ist die neue Gesamtzahl. Was nach dem Checkpoint noch geschrieben wurde, verschwindet aus TensorBoard und den Tabellen; die Schritte laufen nahtlos weiter. Im Test liefert ein unterbrochener und fortgesetzter Lauf dasselbe Log wie ein durchgehender. Ein Lernraten-Scheduler ist nicht im Einsatz; der Eintrag `scheduler` im Checkpoint ist daher `null`.

---

## Typische Abläufe

**Versuchsreihe mit allen Instanzklassen:**

```powershell
python generator.py --classes all --splits test val --processes 10
python schedule_validator.py instances
python run_experiments.py
```

**Neue Instanzgröße:** in `problem_config.json` unter `instance_classes` eintragen, dann `python generator.py --classes NAME --splits test val`. Generator, Experimente und Umgebung kennen sie danach automatisch.

**Deadlines hart statt weich prüfen:** In `problem_config.json` `"due_dates_hard": true` setzen. Instanzen müssen dafür **nicht** neu erzeugt werden.

**Andere Seeds:** in `data_splits` ändern. Für einzelne Instanzen ohne Klasse:

```powershell
python generator.py --count 10 --seed 1000      # Seeds 1000–1009
```

**Regel ändern**, zum Beispiel: automatische Operationen dürfen nicht durch die Pause laufen.
1. In `problem_config.json` `"allowed_in_break": {"automatic": false, ...}` setzen.
2. Instanzen **neu erzeugen**, denn die alten passen wegen des Fingerprints nicht mehr.
3. Generator und Validator berücksichtigen die neue Regel automatisch.

**Variante neben der Standard-Config:** Kopie anlegen, zum Beispiel `config_ohne_pause.json`, und überall `--config config_ohne_pause.json` angeben.

---

## Wichtige Hinweise und Grenzen

- **Arbeitsminuten:** Puffer, Makespan und alle Zeiten in den Dateien zählen nur Arbeitsminuten. Ein Job, der um T1 17:45 fertig ist und dessen Deadline bei T2 07:40 liegt, hat 115 Minuten Puffer, nicht 14 Stunden.
- **Rüsten über Nacht:** Eine Maschine darf am Abend gerüstet werden, die Operation startet dann am nächsten Morgen. Die Rüstung selbst läuft nicht über Nacht.
- **Zu große Instanzen** brechen ohne Klasse mit einer Meldung ab; im Klassen-Modus wird der Seed übersprungen. Das heißt: Die Heuristik hat in `schedule_attempts` Versuchen keinen Plan gefunden. Ein Beweis, dass es keinen gibt, ist das nicht. Mehr Tage, Maschinen oder Worker helfen.
- **Überschreiben:** Dateinamen hängen nur von der Nummer ab. Ein zweiter Aufruf mit demselben `--out` überschreibt vorhandene Instanzen.
- **Standardwerte:** Die Kommandozeile (2 Jobs / 2 Maschinen / 2 Ops / 1 Worker) und `generate_instance` (5 / 5 / 5 / 2) haben unterschiedliche Standardwerte.
- **Regeln ändern** heißt: Instanzen neu erzeugen (Fingerprint). Ausnahme: `due_dates_hard` lässt sich jederzeit umschalten.
- **Generator-Parameter** (Abschnitt `generation`) gehen nicht in den Fingerprint ein. Instanzen, die mit unterschiedlichen Generator-Einstellungen erzeugt wurden, bleiben gültig, sollten aber nicht in derselben Versuchsreihe gemischt werden. Nach einer Änderung also Test- und Validierungsinstanzen neu erzeugen, sonst passen sie nicht mehr zu den Instanzen, die das Training on the fly erzeugt.
- **Ausgeführte Pläne** können in den Zusatztag reichen. Wer sie selbst prüft, muss die Instanz mit `simulator.execution_instance(config, instance)` verlängern, sonst meldet der Validator „außerhalb des Planungshorizonts“. `run_experiments.py` macht das automatisch.
- **Pickup-Fenster** gibt es im Modell nicht, nur Deadlines.
- **Instanzen vor dem 29.10.2026** passen nicht mehr zum Fingerprint, weil `due_dates_hard` seitdem nicht mehr eingeht. Sie müssen einmal neu erzeugt werden; mit gleichem Seed und gleicher Größe sind die Daten identisch.

### Literatur

- Dumas, Y., Desrosiers, J., Gelinas, E., Solomon, M. M. (1995): *An optimal algorithm for the traveling salesman problem with time windows.* Operations Research 43(2). Vorbild für die Konstruktion lösbarer Instanzen über einen vorab gebauten zulässigen Plan.
