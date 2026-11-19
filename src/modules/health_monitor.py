"""Module power model and array health monitor."""
from dataclasses import dataclass, field

import numpy as np

from src.geometry import module_ids


@dataclass
class PowerState:
    struck: set = field(default_factory=set)
    open_switch: set = field(default_factory=set)
    replaced: set = field(default_factory=set)


@dataclass
class Event:
    t: float
    kind: str  # degraded|failed|service_request|checkout_pass|checkout_fail
    module: tuple


def module_powers(t, cfg, state, rng, strike=True, sun_factor=1.0):
    """Per-module output in W: aging, strike, noise, switch, replacement."""
    p = cfg["power"]
    base = p["solar_flux"] * p["area_m2"] * p["eff"] * sun_factor
    target = tuple(cfg["client"]["target"])
    if strike and t >= p["strike_t_s"]:
        state.struck.add(target)
    out = {}
    for m in module_ids(cfg):
        if m in state.replaced:
            w = base  # fresh module: no aging, not struck
        elif m in state.open_switch:
            w = 0.0
        else:
            w = base * p["aging"]
            if m in state.struck:
                w *= p["strike_frac_remaining"]
        if w > 0.0:
            w *= 1.0 + p["noise"] * rng.standard_normal()
        out[m] = float(w)
    return out


def array_total(powers):
    return float(sum(powers.values()))


class HealthMonitor:
    def __init__(self, cfg):
        self.p = cfg["power"]
        self._deg = {}     # module -> time first below degraded
        self._fail = {}    # module -> time first below failed
        self._svc = {}     # module -> time first below service
        self._flagged = set()      # (kind, module) already emitted
        self._excluded = set()     # open-switch modules
        self._checkouts = {}       # module -> (t0, duration, ratios)

    def reset(self, module):
        """Call after a module is replaced: clear its history."""
        for d in (self._deg, self._fail, self._svc):
            d.pop(module, None)
        self._excluded.discard(module)
        self._flagged = {f for f in self._flagged if f[1] != module}

    def start_checkout(self, module, t, duration_s):
        self._checkouts[module] = (t, duration_s, [])

    def _track(self, d, m, cond, t):
        if not cond:
            d.pop(m, None)
            return False
        d.setdefault(m, t)
        return t - d[m] >= self.p["persist_s"]

    def step(self, t, powers):
        events = []
        med = float(np.median(list(powers.values())))
        if med <= 0.0:
            return events
        persist_emit = lambda kind, m: self._emit(events, t, kind, m)
        for m, w in powers.items():
            ratio = w / med
            if m in self._checkouts:
                self._checkouts[m][2].append(ratio)
            if m in self._excluded:
                continue
            if self._track(self._deg, m, ratio < self.p["degraded"], t):
                persist_emit("degraded", m)
            failed = self._track(self._fail, m, ratio < self.p["failed"], t)
            svc = self._track(self._svc, m, ratio < self.p["service"], t)
            if failed:
                persist_emit("failed", m)
                self._excluded.add(m)
            if failed or svc:
                persist_emit("service_request", m)
        for m in list(self._checkouts):
            t0, dur, ratios = self._checkouts[m]
            if t - t0 >= dur:
                lo, hi = self.p["checkout_band"]
                ok = lo <= float(np.mean(ratios)) <= hi
                events.append(Event(t, "checkout_pass" if ok else "checkout_fail", m))
                del self._checkouts[m]
        return events

    def _emit(self, events, t, kind, m):
        if (kind, m) not in self._flagged:
            self._flagged.add((kind, m))
            events.append(Event(t, kind, m))
