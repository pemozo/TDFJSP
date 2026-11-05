# Instanzeigene Zeitprofile und kontrollierter RL-Vergleich

Die neue Versuchsreihe verwendet `problem_config_generalization.json`. Bestehende
Konfigurationen, Instanzordner und Checkpoints bleiben nutzbar. Die neuen Modelle
muessen neu trainiert werden; die alten Gewichte besitzen die neuen Eingaben nicht.

## Starten

```powershell
# Neue feste Validierungs-/Testdaten; vorhandene vollstaendige Daten werden geprueft und wiederverwendet.
python run_profile_comparison.py --stage prepare --processes 4

# Drei gepaarte Trainingsseeds, jeweils A3 full und A3 mean, je 1500 Iterationen.
python run_profile_comparison.py --stage train

# Ergebnisse inklusive Verschlechterungen und Streuung zwischen Trainingsseeds.
python run_profile_comparison.py --stage report

# Alternativ: kompletter technischer Kurztest in separaten smoke-Ordnern.
python run_profile_comparison.py --smoke --stage all
```

`--seeds 1` beschraenkt einen Versuch auf ein Trainingspaar. `--iterations N`
ueberschreibt das Trainingsbudget, nicht die fest konfigurierte Lernratenkurve.
Der Kurztest verwendet kleine Netze, vier Iterationen und jeweils zwei Instanzen
in zwei Klassen. Seine Seeds sind von den regulaeren Validierungs- und Testdaten
getrennt. Er prueft die Technik und ist kein Leistungsnachweis. `--output ORDNER`
erlaubt einen separaten Ergebnisordner fuer weitere Konfigurationen.

Ein einzelner regulaerer Lauf ist ebenfalls moeglich:

```powershell
python -m rl.train --config problem_config_generalization.json --variant A3 --seed 1
python -m rl.train --config problem_config_generalization.json --variant A3 --mean-time-profiles --seed 1
```

Fuer solche Einzelaufrufe erzeugt das Training dieselben CSVs und Plaene. Der
automatische Paarbericht nutzt die vom Vergleichsskript gespeicherte Laufzuordnung.

## Was sich geaendert hat

**Profile je Instanz.** Der Generator schreibt `time_profiles.manual_stations`,
`time_profiles.setup` und `time_profile_id` in jede Instanz. Er verschiebt die
Phasen der Stationsvorlagen, variiert die Staerke und fuegt kleine Abweichungen
hinzu. Die Faktoren liegen in der neuen Konfiguration zwischen 1,0 und 1,8;
Pausen und Fenstergrenzen bleiben gleich. Ein eigener Zufallsstrom aus Seed und
Instanzgroesse sorgt fuer Reproduzierbarkeit. Die Profile veraendern nicht den
Zufallsstrom fuer Basisbearbeitungszeiten und Maschinenzulaessigkeit.

Die Werte sind synthetische Versuchsannahmen. Unterschiedliche Instanzseeds
erzeugen unterschiedliche Profile; dieselbe Instanz bekommt in beiden
Vergleichsmodellen exakt dieselben Profile. Der Datensatzpruefer kontrolliert
Profilduplikate, Dateihashes und die Trennung von Validierung und Test.
Trainingsinstanzen entstehen aus einem separaten Seedbereich.

**Durchgaengige Ausfuehrung.** Validator, Referenzgenerator, Simulator,
Dispatching-Regeln, RL-Umgebung und Monitoring verwenden die Profile der Instanz.
Ein Batch darf unterschiedliche Profile enthalten. Die Beobachtung und die
Restzeit-Schaetzung erhalten ebenfalls die richtigen Profile je Batchposition.
Eine statische Simulationssicht bleibt statisch und wird nicht versehentlich
durch Instanzprofile ueberschrieben.

**Maschinenversorgung als Eingabe.** Mit `model.flow_features=true` kommen hinzu:

- sofort verfuegbare und verbleibende Arbeit je Ressource;
- der erwartete Freigabeabstand fuer einen automatischen Nachfolger;
- die bereitstehende Arbeit auf dessen moeglichen Maschinen;
- die manuelle Zusatzdauer gegenueber dem guenstigsten Stationsfenster;
- die reine Bearbeitungszeitersparnis bei einem Start ab dem naechsten Fenster.

Freigabeabstand und Arbeitsvorrat sind Entscheidungshilfen, keine exakte
Vorhersage kuenftigen Leerlaufs. Die schon vorhandenen Deadlines, Puffer,
Start-/Endzeiten, Profile und Wartevorteile bleiben erhalten.

## Zielfunktion

Die neue Konfiguration verwendet eine gewichtete Summe, mit folgenden
Startgewichten fuer die Validierung:

| Bestandteil | Normalisierung | Gewicht |
|---|---|---:|
| Summe der Verspaetungen | je Job | 1,0 |
| Zahl verspaeteter Jobs | je Job, also Anteil | 15,0 |
| Makespan | unveraendert | 0,02 |
| Manuelle Profil-Zusatzdauer | je manueller Operation | 0,25 |
| Summe der Fertigstellungszeiten automatischer Operationen | je automatischer Operation | 0,02 |

Die manuelle Profil-Zusatzdauer ist die tatsaechliche Bearbeitungszeit abzueglich
der kleinsten kalenderzulaessigen gerundeten Dauer auf der gewaehlten Station.
Diese Untergrenze ignoriert Praezedenz und Ressourcenkonkurrenz. Sie ist deshalb
kein Beweis, dass ein entsprechender Gesamtplan erreichbar ist.

Makespan und fruehe automatische Fertigstellungen foerdern einen kompakten
Produktionsfluss. Die reine Maschinenbelegung wird nicht direkt belohnt:
laengere Bearbeitung oder mehr Ruestung sollen den Reward nicht verbessern.
Produktive Auslastung und Maschinenleerlauf werden separat gemessen und muessen
sich im Vergleich tatsaechlich verbessern.

Das Schritt-Reward bleibt eine Kostendifferenz. Bei `gamma=1` entspricht seine
Summe weiterhin den Startkosten minus den wirklichen Endkosten. Nicht beendete
Plaene erhalten weiterhin eine starke Strafe fuer offene Operationen.

Die Gewichte sind keine Garantie, dass jedes Ziel gleichzeitig besser wird.
Sie werden anhand der Validierung beurteilt; die Testdaten dienen nicht zur
nachtraeglichen Auswahl einer besonders guenstigen Gewichtung.

## Fairer Informationsvergleich

| Einstellung | Tagesprofile | Uhrzeit und Kalender | Deadlines |
|---|---|---|---|
| `full` | tatsaechliche instanzeigene Verlaeufe | bekannt | bekannt |
| `mean` | zeitgewichtetes konstantes Mittel je Station und Ruestung | bekannt | bekannt |
| `none` | Faktor 1; bisherige Ablation | keine explizite Uhrzeit | relative Puffer, keine absolute Deadline im Encoder |

Der Hauptvergleich ist **A3 full gegen A3 mean**. Beide sehen dieselbe Architektur,
dieselben zusaetzlichen Flussmerkmale, dieselben Deadlines und dieselbe reale
Aktionsmaske. Beide duerfen warten. Auch der Kontrollagent kennt also die
durchschnittliche Geschwindigkeit; nur der zeitliche Verlauf fehlt. Damit wird
der Effekt zeitabhaengiger Profile weniger mit grundsaetzlicher Unkenntnis ueber
Dauern oder Deadlines vermischt. `none` bleibt fuer alte Experimente erhalten.

Die Profile der Testinstanzen sind neu, stammen aber aus derselben
Generatorfamilie. Das prueft Generalisierung innerhalb dieser Verteilung. Fuer
Aussagen ueber reale andere Profilfamilien oder andere Pausenkalender braucht
es einen zusaetzlichen, separat definierten Testdatensatz.

## Stabilitaet, Ausgaben und Beurteilung

Eine anfaengliche WAIT-Ausgabeverzerrung von -2,0 senkt die anfaengliche
Wahrscheinlichkeit freiwilligen Wartens. Sie gilt fuer beide Vergleichsmodelle;
WAIT bleibt erlaubt, und der Parameter wird normal mittrainiert.

Die neue Lernrate startet bei 0,0001 und sinkt ueber 1500 Iterationen linear auf
20 Prozent. `target_kl=0.015` beendet PPO-Updates fuer einen Rollout frueher,
wenn der gemessene KL-Abstand 1,5-mal diesen Wert ueberschreitet. Das beste
Modell wird weiterhin anhand der Validierung ausgewaehlt und erst anschliessend
auf dem Testdatensatz ausgewertet. Alte Konfigurationen ohne diese Felder
behalten ihre bisherige Lernrate und PPO-Updatezahl.

Neue Ergebnisse liegen in `experiments_generalization/`, Instanzen in
`instances_generalization/`. Der Kurztest liegt jeweils im Unterordner `smoke_verified`.
Wichtige Dateien:

- `dataset_manifest.json`: Profil-IDs und Hashes aller festen Eingabedateien;
- `comparison_runs.json`: Zuordnung von Trainingsseed, Informationssicht und Checkpoint;
- `profile_comparison.csv`: Ergebnisse je Seed und Sicht, inklusive Stichprobenumfang;
- `paired_profile_differences.csv`: Differenzen full minus mean auf identischen Instanzen;
- `comparison_report.md`: Vergleich und Standardabweichung zwischen Trainingsseeds;
- `results/<run>/test_schedules.jsonl`: vollstaendige Plaene, WAIT-Intervalle und Profile;
- `test_instances.csv` und Scalars: manuelle Profilnutzung, produktive
  Maschinenauslastung, Ruestbelegung, Maschinenleerlauf, Verspaetung und Puenktlichkeit.

Produktive Auslastung = automatische Bearbeitungszeit / (Maschinenanzahl ×
Makespan), je Instanz berechnet. Ruestung ist separate Belegung. Bei Abbruechen
fehlen Planmetriken; der Bericht zeigt die Anzahl gueltiger Werte und darf diese
Teilmengen nicht als vollstaendigen Qualitaetsnachweis darstellen.

Eine belastbare Verbesserung verlangt insbesondere geringere Verspaetung und
mehr puenktliche Jobs auf neuen Profilen, dazu geringere manuelle Profil-Zusatzdauer
und weniger Maschinenleerlauf beziehungsweise bessere produktive Auslastung.
Der Bericht weist auch Zielkonflikte und Verschlechterungen aus.

Pruefung der Implementierung:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:MPLBACKEND='Agg'
python -m unittest tests.test_instance_profiles tests.test_batched_profiles tests.test_profile_learning tests.test_new_metrics
```
