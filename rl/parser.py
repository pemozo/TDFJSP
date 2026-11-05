"""Uebersetzung zwischen Instanz-JSON (generator.py) und Tensoren der Umgebung.

Ein Zustand ist ein dict aus Tensoren mit gemeinsamer erster Dimension B (Batch).
Abkuerzungen in den Formen: J = Jobs, O = Operationen, R = Ressourcen (erst
Handarbeitsplaetze, dann Maschinen, wie schedule_validator.all_resources),
Rs = Ressourcen mit Ruestung.

Operationen sind je Job fortlaufend nummeriert; der Referenz-Plan bleibt dem Agenten
verborgen und wird nicht uebersetzt.
"""
from __future__ import annotations

import torch

from rl.time_model import KIND_IDS
from schedule_validator import all_resources, horizon, setup_resources, instance_config
from simulator import execution_instance, make_entry

_NO_SETUP = torch.iinfo(torch.int16).max


def instances_to_td(config: dict, instances: list[dict],
                    device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Instanzen gleicher Groesse zu einem Batch der festen Instanzdaten."""
    tensors = [_instance_to_tensors(config, instance) for instance in instances]
    shapes = {tuple(t["proc_times"].shape) + tuple(t["due"].shape) for t in tensors}
    if len(shapes) != 1:
        raise ValueError("Ein Batch braucht Instanzen gleicher Groesse (Jobs, Operationen, "
                         f"Ressourcen); gefunden: {sorted(shapes)}")
    td = {key: torch.stack([t[key] for t in tensors]).to(device) for key in tensors[0]}
    return _add_derived_keys(td)


def _instance_to_tensors(config: dict, instance: dict) -> dict[str, torch.Tensor]:
    """Eine Instanz ohne Batch-Dimension."""
    jobs = instance["jobs"]
    operations = {o["id"]: o for o in instance["operations"]}
    op_ids = [op_id for job in jobs for op_id in job["operations"]]
    op_index = {op_id: i for i, op_id in enumerate(op_ids)}
    resources = all_resources(instance)
    resource_index = {r: i for i, r in enumerate(resources)}
    with_setup = setup_resources(config, instance)
    setup_index = {r: i for i, r in enumerate(with_setup)}

    proc_times = torch.zeros(len(resources), len(op_ids), dtype=torch.long)
    for o, op_id in enumerate(op_ids):
        for resource, minutes in operations[op_id]["base_processing"].items():
            proc_times[resource_index[resource], o] = minutes

    setup_times = torch.zeros(len(with_setup), len(op_ids), len(op_ids), dtype=torch.int16)
    if instance["setups"]:
        pairs = [(setup_index[s["resource"]], op_index[s["from_op"]], op_index[s["to_op"]])
                 for s in instance["setups"]]
        rows, previous, following = zip(*pairs)
        setup_times[list(rows), list(previous), list(following)] = torch.tensor(
            [s["base_minutes"] for s in instance["setups"]], dtype=torch.int16)

    start_op, end_op, job_of_op, position = [], [], [], []
    for j, job in enumerate(jobs):
        start_op.append(op_index[job["operations"][0]])
        end_op.append(op_index[job["operations"][-1]])
        job_of_op += [j] * len(job["operations"])
        position += list(range(len(job["operations"])))

    exec_horizon = horizon(config, execution_instance(config, instance))
    effective = instance_config(config, instance)
    windows = effective["time_model"]["day_windows"]
    reference = [w["factor"] for w in windows]
    profiles = effective["time_model"].get("resource_factors")
    stations = len(instance["resources"]["manual_stations"])
    station_profiles = profiles["manual_stations"][:stations] if profiles else [reference] * stations
    setup_profile = profiles["setup"] if profiles else reference

    def factors(values):
        return [-1.0 if value is None else float(value) for value in values]

    return {
        "proc_times": proc_times,
        "setup_times": setup_times,
        "setup_resources": torch.tensor([resource_index[r] for r in with_setup], dtype=torch.long),
        "is_station": torch.tensor([r in instance["resources"]["manual_stations"]
                                    for r in resources], dtype=torch.bool),
        "op_kind": torch.tensor([KIND_IDS[operations[op_id]["kind"]] for op_id in op_ids],
                                dtype=torch.long),
        "ops_job_map": torch.tensor(job_of_op, dtype=torch.long),
        "ops_sequence_order": torch.tensor(position, dtype=torch.long),
        "start_op_per_job": torch.tensor(start_op, dtype=torch.long),
        "end_op_per_job": torch.tensor(end_op, dtype=torch.long),
        "release": torch.tensor([job["release"] for job in jobs], dtype=torch.long),
        "due": torch.tensor([job["due"] for job in jobs], dtype=torch.long),
        "horizon": torch.tensor(exec_horizon, dtype=torch.long),
        "seed": torch.tensor(instance["seed"], dtype=torch.long),
        "time_profile_factors": torch.tensor([factors(row) for row in (*station_profiles, setup_profile)],
                                             dtype=torch.float64),
        "time_profile_reference": torch.tensor(factors(reference), dtype=torch.float64),
    }


def _add_derived_keys(td: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Kennzahlen je Operation, die aus den Instanzdaten folgen."""
    eligible = td["proc_times"] > 0
    count = eligible.sum(1)
    td["num_eligible"] = count
    td["mean_proc_times"] = (td["proc_times"].sum(1).to(torch.float32)
                             / count.clamp(min=1).to(torch.float32))
    td["min_proc_times"] = td["proc_times"].masked_fill(~eligible, torch.iinfo(torch.long).max
                                                        ).amin(1).masked_fill(count == 0, 0)
    # kleinste Ruestzeit, um auf eine Operation zu ruesten (0 = nie geruestet)
    if td["setup_times"].shape[1] == 0:
        td["min_setup_times"] = torch.zeros_like(td["min_proc_times"])
    else:
        positive = td["setup_times"].masked_fill(td["setup_times"] <= 0, _NO_SETUP)
        smallest = positive.amin(dim=(1, 2)).to(torch.long)
        td["min_setup_times"] = smallest.masked_fill(smallest == _NO_SETUP, 0)
    return td


def td_to_schedule(config: dict, td: dict[str, torch.Tensor], index: int,
                   instance: dict) -> list[dict]:
    """Plan einer Batch-Position im Format von schedule_validator.check_schedule.
    Reihenfolge wie eingeplant; nicht eingeplante Operationen fehlen."""
    op_ids = [op_id for job in instance["jobs"] for op_id in job["operations"]]
    resources = all_resources(instance)
    with_setup = set(setup_resources(config, instance))

    order = td["op_order"][index]
    schedule = []
    for o in torch.argsort(order).tolist():
        if order[o] < 0:
            continue
        resource = resources[int(td["op_resource"][index, o])]
        previous = int(td["op_predecessor"][index, o])
        setup = (None if previous < 0 else op_ids[previous], 0) if resource in with_setup else None
        decided, start, end = (int(td[key][index, o]) for key in
                               ("op_decision", "op_start", "op_end"))
        schedule.append(make_entry(op_ids[o], resource, setup, decided, start - decided, end - start))
    return schedule
