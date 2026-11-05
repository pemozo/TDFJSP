"""Zeitmodell als Tensoren: dieselben Regeln wie schedule_validator.duration, aber batchweise.

Alle Fenster, Faktoren und Regeln kommen aus problem_config.json. Die Dauerformel
rechnet bewusst mit float64 und demselben Epsilon wie der Validator, damit beide
fuer jeden Faktor exakt dasselbe Ergebnis liefern (tests/test_rl_env.py prueft das).

Aufbau:
    duration          Dauer und Zulaessigkeit einer Taetigkeit je Element
    next_boundary     naechste Fenstergrenze (Ereignis fuer die Uhr)
    finish_after      fruehestes Ende ab einem Zeitpunkt, fuer die Kostenschaetzung
    time_features     Merkmale des aktuellen Zeitpunkts fuer den Agenten
"""
from __future__ import annotations

import copy

import torch

from schedule_validator import KINDS
from simulator import view_config

KIND_IDS = {kind: i for i, kind in enumerate(KINDS)}
SETUP = KIND_IDS["setup"]
MANUAL = KIND_IDS["manual"]

TIME_FEATURES = ("clock_in_day", "day_index", "factor_now", "factor_next",
                 "minutes_to_next_window", "minutes_to_break", "minutes_to_day_end")

_EPSILON = 1e-9     # wie schedule_validator.duration
_IN_BREAK = -1.0    # Markierung in der Faktor-Tabelle
_UNREACHABLE = 10 ** 12


class TimeModel:
    """Kalender eines Tages als Tabellen je Minute. `view` wie simulator.DURATION_VIEWS:
    bei "static" sind alle Faktoren 1.0, Pause und Arbeitsende bleiben bekannt.

    Mit time_model.resource_factors hat jeder Handarbeitsplatz ein eigenes Profil und die Ruestung ebenfalls.
    Ressourcen sind dann Indizes wie in rl/parser.py (erst Handarbeitsplaetze). Ohne Ressource gilt fuer manuelle
    Operationen das Bezugsprofil day_windows, auch fuer die Uhrzeit-Merkmale und noch nicht zugeordnete Operationen."""

    def __init__(self, config: dict, view: str = "time_aware", device: str | torch.device = "cpu"):
        self.view = view
        cfg = view_config(config, "time_aware" if view == "mean" else view)
        if view == "mean":
            cfg = copy.deepcopy(cfg)
            windows = cfg["time_model"]["day_windows"]

            def mean_profile(values):
                weights = [w["end"] - w["start"] for w in windows]
                total = sum(weight for value, weight in zip(values, weights) if value is not None)
                mean = sum(value * weight for value, weight in zip(values, weights)
                           if value is not None) / total
                return [None if value is None else mean for value in values]

            reference = mean_profile([w["factor"] for w in windows])
            for window, value in zip(windows, reference):
                window["factor"] = value
            profiles = cfg["time_model"].get("resource_factors")
            if profiles:
                profiles["manual_stations"] = [mean_profile(row) for row in profiles["manual_stations"]]
                profiles["setup"] = mean_profile(profiles["setup"])
        time_model, rules = cfg["time_model"], cfg["rules"]
        self.day = time_model["day_minutes"]
        self.device = torch.device(device)
        windows = time_model["day_windows"]
        self.breaks = [(w["start"], w["end"]) for w in windows if w["factor"] is None]
        self._overnight_possible = any(rules["allowed_overnight"].values())

        factor, window_end, next_factor = [], [], []
        for i, window in enumerate(windows):
            following = windows[(i + 1) % len(windows)]["factor"]
            for _ in range(window["start"], window["end"]):
                factor.append(_IN_BREAK if window["factor"] is None else window["factor"])
                window_end.append(window["end"])
                next_factor.append(_IN_BREAK if following is None else following)

        self.window_starts = self._tensor([w["start"] for w in windows], torch.long)
        self.window_lengths = self._tensor([w["end"] - w["start"] for w in windows], torch.long)
        self._factor = self._tensor(factor, torch.float64)
        self._window_end = self._tensor(window_end, torch.long)
        self._next_factor = self._tensor(next_factor, torch.float64)
        # Zeilen: je Handarbeitsplatz eine, danach die Ruestung; Spalten: Minuten des Tages
        profiles = time_model.get("resource_factors")
        self.station_count = len(profiles["manual_stations"]) if profiles else 0
        self._profile_factor = None
        if profiles:
            self._profile_factor = self._tensor(
                [[_IN_BREAK if value is None else value
                  for window, value in zip(windows, row) for _ in range(window["start"], window["end"])]
                 for row in (*profiles["manual_stations"], profiles["setup"])], torch.float64)
        self._to_break = self._tensor([self._minutes_to_break(m) for m in range(self.day)],
                                      torch.long)
        self._time_dependent = self._kind_rule(rules["time_dependent"])
        self._in_break_allowed = self._kind_rule(rules["allowed_in_break"])
        self._overnight_allowed = self._kind_rule(rules["allowed_overnight"])
        self._finish: torch.Tensor | None = None    # siehe ensure_finish_table
        self._bound_source = None
        self._bound_reference_source = None
        self._batch_profiles: torch.Tensor | None = None
        self._batch_reference: torch.Tensor | None = None
        self._batch_next_factor: torch.Tensor | None = None
        self._batch_finish: torch.Tensor | None = None
        self._batch_station_count = 0

    def bind(self, td: dict[str, torch.Tensor]) -> None:
        """Bind instance-specific profiles, preserving the fast fixed-profile path.

        Leading dimension is the environment batch. Subset operations must provide
        ``batch_index`` to duration/finish_after; an omitted index means the full batch.
        Rebinding the same state tensors is cheap and does not rebuild finish tables.
        """
        profiles = td.get("time_profile_factors")
        reference = td.get("time_profile_reference")
        if profiles is self._bound_source and reference is self._bound_reference_source:
            return
        self._bound_source, self._bound_reference_source = profiles, reference
        self._batch_profiles = self._batch_reference = self._batch_next_factor = None
        self._batch_finish = None
        if profiles is None or self.view == "static":
            return
        if reference is None:
            raise ValueError("time_profile_reference is required with time_profile_factors")
        if profiles.ndim != 3 or reference.shape != (profiles.shape[0], profiles.shape[2]):
            raise ValueError("Profile tensors must have shapes [B,H+1,K] and [B,K]")
        if profiles.shape[2] != self.window_starts.numel():
            raise ValueError("Instance profiles must use the configured calendar windows")
        values = profiles.to(device=self.device, dtype=torch.float64)
        ref = reference.to(device=self.device, dtype=torch.float64)
        if self.view == "mean":
            def averaged(rows):
                active = rows >= 0
                weights = self.window_lengths.to(torch.float64) * active
                mean = (rows.clamp(min=0) * weights).sum(-1, keepdim=True) / weights.sum(-1, keepdim=True)
                return torch.where(active, mean.expand_as(rows), torch.full_like(rows, _IN_BREAK))
            values, ref = averaged(values), averaged(ref)
        stations = values.shape[1] - 1
        configured_ref = self._factor[self.window_starts]
        if self._profile_factor is None:
            configured = configured_ref.expand(stations + 1, -1)
        elif stations <= self.station_count:
            configured = torch.cat((self._profile_factor[:stations, self.window_starts],
                                    self._profile_factor[self.station_count:self.station_count + 1,
                                                         self.window_starts]), dim=0)
        else:
            configured = None
        if (configured is not None and torch.equal(values, configured.unsqueeze(0).expand_as(values))
                and torch.equal(ref, configured_ref.unsqueeze(0).expand_as(ref))):
            return
        self._batch_station_count = stations
        self._batch_profiles = values.repeat_interleave(self.window_lengths, dim=-1)
        self._batch_reference = ref.repeat_interleave(self.window_lengths, dim=-1)
        self._batch_next_factor = ref.roll(-1, dims=-1).repeat_interleave(self.window_lengths, dim=-1)

    def _batch_index(self, values: torch.Tensor, batch_index: torch.Tensor | None) -> torch.Tensor:
        if batch_index is not None:
            return batch_index.to(device=self.device, dtype=torch.long)
        batch = self._batch_profiles.shape[0]
        if values.ndim == 0 or values.shape[0] != batch:
            raise ValueError("Subset time-model calls require explicit batch_index")
        return torch.arange(batch, device=self.device).view(batch, *((1,) * (values.ndim - 1)))

    def _tensor(self, values: list, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, device=self.device)

    def _kind_rule(self, rule: dict) -> torch.Tensor:
        return self._tensor([rule[kind] for kind in KINDS], torch.bool)

    def _minutes_to_break(self, minute: int) -> int:
        """Minuten bis zur naechsten Pause; 0 in der Pause; ohne Pausen ein ganzer Tag."""
        if not self.breaks:
            return self.day
        if any(start <= minute < end for start, end in self.breaks):
            return 0
        return min((start - minute) % self.day for start, _ in self.breaks)

    # --- Dauer und Zulaessigkeit --------------------------------------------

    def _factor_for(self, kind: torch.Tensor, start: torch.Tensor,
                    resource: torch.Tensor | None, batch_index: torch.Tensor | None = None) -> torch.Tensor:
        """Faktor je Element, wie schedule_validator.factor_at; Pause = _IN_BREAK."""
        minute = start % self.day
        if self._batch_profiles is not None:
            index = self._batch_index(start, batch_index)
            factor = torch.where(kind == SETUP,
                                 self._batch_profiles[index, self._batch_station_count, minute],
                                 self._batch_reference[index, minute])
            if resource is not None and self._batch_station_count:
                own = self._batch_profiles[index, resource.clamp(max=self._batch_station_count - 1), minute]
                factor = torch.where((kind == MANUAL) & (resource < self._batch_station_count), own, factor)
            return factor
        if self._profile_factor is None:
            return self._factor[minute]
        factor = torch.where(kind == SETUP, self._profile_factor[self.station_count, minute], self._factor[minute])
        if resource is None:
            return factor
        own = self._profile_factor[resource.clamp(max=self.station_count - 1), minute]
        return torch.where((kind == MANUAL) & (resource < self.station_count), own, factor)

    def duration(self, kind: int | torch.Tensor, base: torch.Tensor, start: torch.Tensor,
                 horizon: torch.Tensor | int, resource: torch.Tensor | None = None,
                 batch_index: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Dauer und Zulaessigkeit je Element, wie schedule_validator.duration.
        kind = Index aus KIND_IDS. Basisdauer 0 heisst: keine Taetigkeit (immer zulaessig)."""
        kind = torch.as_tensor(kind, device=self.device)
        time_dependent = self._time_dependent[kind]
        factor = self._factor_for(kind, start, resource, batch_index)
        in_break = factor < 0

        scaled = torch.ceil(base.to(torch.float64) * factor - _EPSILON).to(torch.long)
        length = torch.where(time_dependent & ~in_break, scaled, base)
        end = start + length
        first_day, last_day = start // self.day, (end - 1) // self.day

        allowed = (start >= 0) & (end <= horizon) & ~(time_dependent & in_break)
        allowed = allowed & (self._overnight_allowed[kind] | (first_day == last_day))
        allowed = allowed & (self._in_break_allowed[kind]
                             | ~self._overlaps_break(start, end, first_day, last_day))
        nothing = base == 0
        return length.masked_fill(nothing, 0), allowed | nothing

    def _overlaps_break(self, start: torch.Tensor, end: torch.Tensor, first_day: torch.Tensor,
                        last_day: torch.Tensor) -> torch.Tensor:
        """Beruehrt das Intervall [start, end) eine Pause? Ohne Nachtarbeit reicht ein Tag."""
        days = int((last_day - first_day).max().item()) + 1 if self._overnight_possible else 1
        hit = torch.zeros_like(start, dtype=torch.bool)
        for offset in range(days):
            day_start = (first_day + offset) * self.day
            for break_start, break_end in self.breaks:
                hit = hit | ((first_day + offset <= last_day)
                             & (start < day_start + break_end) & (end > day_start + break_start))
        return hit

    def window_profile(self, kinds: torch.Tensor, base: torch.Tensor,
                       resource: torch.Tensor | None = None,
                       batch_index: torch.Tensor | None = None) -> torch.Tensor:
        """Dauer bei Start in jedem Zeitfenster [..., K]. Der Faktor gilt im ganzen Fenster, die Dauer
        haengt also nur vom Fenster ab. 0 = in diesem Fenster nicht startbar (zeitabhaengig in der Pause)."""
        count = len(self.window_starts)
        repeated = base.unsqueeze(-1).expand(*base.shape, count)
        kinds = kinds.unsqueeze(-1).expand(*base.shape, count)
        starts = self.window_starts.expand(*base.shape, count)
        if resource is not None:
            resource = resource.unsqueeze(-1).expand(*base.shape, count)
        if batch_index is not None:
            batch_index = batch_index.unsqueeze(-1)
        length, _ = self.duration(kinds, repeated, starts, _UNREACHABLE, resource, batch_index)
        blocked = self._time_dependent[kinds] & (self._factor[starts] < 0)
        return length.masked_fill(blocked, 0)

    # --- Ereignisse -----------------------------------------------------------

    def next_boundary(self, time: torch.Tensor,
                      horizon: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Naechste Fenstergrenze nach `time` und ob sie noch vor dem Horizont liegt."""
        minute = time % self.day
        boundary = time - minute + self._window_end[minute]
        return boundary, boundary < horizon

    # --- Kostenschaetzung -----------------------------------------------------

    def ensure_finish_table(self, max_base: int) -> None:
        """Tabelle fuer finish_after je Art, Basisdauer und Minute (bei Bedarf neu bauen).
        Mit Profilen je Handarbeitsplatz kommt hinter den Arten je Platz eine Zeile dazu."""
        if self._batch_profiles is not None:
            self._ensure_batch_finish_table(max_base)
            return
        if self._finish is not None and max_base < self._finish.shape[1]:
            return
        size = max_base + 1
        span = 3 * self.day     # heutiger Tag und Folgetage
        starts = torch.arange(span, device=self.device)
        bases = torch.arange(1, size, device=self.device).unsqueeze(1)
        rows = [(kind, None) for kind in range(len(KINDS))]
        rows += [(MANUAL, station) for station in range(self.station_count)]
        table = torch.zeros(len(rows), size, self.day, dtype=torch.long, device=self.device)
        for row, (kind, station) in enumerate(rows):
            all_starts = starts.expand(size - 1, -1)
            resource = None if station is None else torch.full_like(all_starts, station)
            length, allowed = self.duration(kind, bases.expand(-1, span), all_starts, _UNREACHABLE, resource)
            # fruehester zulaessiger Start ab jeder Minute: kumuliertes Minimum von hinten
            candidate = all_starts.masked_fill(~allowed, _UNREACHABLE)
            first = candidate.flip(1).cummin(1).values.flip(1)[:, :self.day]
            found = first < _UNREACHABLE
            first = first.clamp(max=span - 1)
            delay = first - starts[:self.day] + length.gather(1, first)
            table[row, 1:] = torch.where(found, delay, bases.expand(-1, self.day))
        self._finish = table

    def _ensure_batch_finish_table(self, max_base: int) -> None:
        """Vectorized over instances; int32 caches keep per-batch memory bounded."""
        if self._batch_finish is not None and max_base < self._batch_finish.shape[2]:
            return
        batch, size, span = self._batch_profiles.shape[0], max_base + 1, 3 * self.day
        starts = torch.arange(span, device=self.device)
        bases = torch.arange(1, size, device=self.device).view(1, -1, 1)
        all_starts = starts.view(1, 1, -1).expand(batch, size - 1, -1)
        all_bases = bases.expand_as(all_starts)
        rows = [(kind, None) for kind in range(len(KINDS))]
        rows += [(MANUAL, station) for station in range(self._batch_station_count)]
        table = torch.zeros(batch, len(rows), size, self.day, dtype=torch.int32, device=self.device)
        for row, (kind, station) in enumerate(rows):
            resource = None if station is None else torch.full_like(all_starts, station)
            length, allowed = self.duration(kind, all_bases, all_starts, _UNREACHABLE, resource)
            candidate = all_starts.masked_fill(~allowed, _UNREACHABLE)
            first = candidate.flip(2).cummin(2).values.flip(2)[:, :, :self.day]
            found = first < _UNREACHABLE
            first = first.clamp(max=span - 1)
            delay = first - starts[:self.day] + length.gather(2, first)
            table[:, row, 1:] = torch.where(found, delay, bases.expand(batch, -1, self.day)).to(torch.int32)
        self._batch_finish = table

    def finish_after(self, kind: int | torch.Tensor, base: torch.Tensor, time: torch.Tensor,
                     resource: torch.Tensor | None = None,
                     batch_index: torch.Tensor | None = None) -> torch.Tensor:
        """Fruehestes Ende, wenn die Taetigkeit ab `time` am naechsten zulaessigen Zeitpunkt
        beginnt. Der Horizont zaehlt hier nicht, es ist nur eine Schaetzung."""
        row = torch.as_tensor(kind, device=self.device)
        if self._batch_profiles is not None:
            if self._batch_finish is None:
                raise RuntimeError("Call ensure_finish_table after binding profiles")
            if resource is not None and self._batch_station_count:
                row = torch.where((row == MANUAL) & (resource < self._batch_station_count),
                                  len(KINDS) + resource, row)
            return time + self._batch_finish[self._batch_index(time, batch_index), row, base, time % self.day]
        if resource is not None and self.station_count:
            row = torch.where((row == MANUAL) & (resource < self.station_count), len(KINDS) + resource, row)
        return time + self._finish[row, base, time % self.day]

    # --- Merkmale -------------------------------------------------------------

    def time_features(self, time: torch.Tensor, horizon: torch.Tensor,
                      scale: float) -> torch.Tensor:
        """Merkmale je Instanz in der Reihenfolge TIME_FEATURES. In der Pause ist der Faktor 0."""
        minute = time % self.day
        if self._batch_reference is not None:
            index = self._batch_index(time, None)
            current_factor = self._batch_reference[index, minute]
            next_factor = self._batch_next_factor[index, minute]
        else:
            current_factor, next_factor = self._factor[minute], self._next_factor[minute]
        features = [minute.to(torch.float64) / self.day,
                    (time // self.day).to(torch.float64) * self.day / horizon,
                    current_factor.clamp(min=0),
                    next_factor.clamp(min=0),
                    (self._window_end[minute] - minute).to(torch.float64) / scale,
                    self._to_break[minute].to(torch.float64) / scale,
                    (self.day - minute).to(torch.float64) / scale]
        return torch.stack(features, dim=1).to(torch.float32)

    def factor_for_kind(self, kinds: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Faktor je Operation, wenn sie jetzt starten wuerde (in der Pause: Faktor danach).
        time hat die Form (B); kinds (B, O). Nicht zeitabhaengige Arten haben Faktor 1."""
        minute = time % self.day
        if self._batch_reference is not None:
            index = self._batch_index(time, None)
            current, following = self._batch_reference[index, minute], self._batch_next_factor[index, minute]
        else:
            current, following = self._factor[minute], self._next_factor[minute]
        factor = torch.where(current < 0, following, current).clamp(min=0).unsqueeze(1)
        return torch.where(self._time_dependent[kinds], factor, torch.ones_like(factor))
