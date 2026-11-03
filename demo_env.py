from rl.env import TDFJSPEnv
from rl.policies import edd_ect

# Einstellungen dieses Demos
PLOTTEN = True          # True: Loesung jeder Instanz als Gantt-Diagramm anzeigen
ANZAHL_INSTANZEN = 1    # so viele neue Instanzen laufen gleichzeitig
INSTANZKLASSE = "mittel_18"  # Name aus instance_classes in problem_config.json

env = TDFJSPEnv()
print("Geraet:", env.device, "| Warten erlaubt:", env.allow_wait)

instanzen = env.generator.sample_instances(ANZAHL_INSTANZEN, INSTANZKLASSE)    # neue Instanzen on the fly
td = env.load_instances(instanzen)
print("Jobs:", td["due"].shape[1], "| Aktionen:", td["action_mask"].shape[1])

schritte = 0
while not td["done"].all():
    td["action"] = edd_ect(td)              # hier kommt später der Agent hin
    td = env.step(td)["next"]
    schritte += 1

print("Schritte:", schritte)
for name, wert in env.get_objectives(td).items():
    print(f"  {name}: {wert.tolist()}")
print("  reward:", [round(x, 4) for x in env.get_reward(td).tolist()])

if PLOTTEN:
    import matplotlib.pyplot as plt

    for i, instanz in enumerate(instanzen):
        env.render(td, i, instanz)
    plt.show()      # Skript laeuft weiter, sobald alle Fenster geschlossen sind
