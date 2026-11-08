"""Thermal pack simulator for testing the hot-cell controller.

The plant is a 32-cell sodium-ion series string (4 x 8 grid, ids C01..C32 as in the Firebreak
layout) built from the published Firebreak cell assumptions (``firebreak/docs/PHYSICS.md``):
100 Ah, sloping Na-ion OCV, R0 1.2 mOhm + one RC branch (0.8 mOhm, 60 s), Arrhenius
resistance (20 kJ/mol), 2.5 kJ/K heat capacity, 1.0 W/K neighbour contact, 0.25/0.55/0.80 W/K
enclosure cooling for interior/edge/corner cells, 0.25 mOhm REON switch path. Every cell sits
in its own local air (room temperature + a pocket offset), which is what the controller's
*floor* has to learn. The plant, its parameters and its per-cell spread are hidden from the
controller, which sees only noisy REON-style telemetry.

Strategies compared (all share the same plant, schedule, noise and seeds, so results are paired):

* ``fixed``: traditional BMS, fixed series string, no switches. The whole string stops at
  57 C (until 50 C), charging stops if any cell is outside 15-45 C or full, discharging stops
  when any cell is empty.
* ``safety_bypass``: REON switches used only for per-cell safety (57 C latch, empty/full
  cells, 15-45 C charge window) with the power/voltage check. No hot-cell logic.
* ``naive``: safety bypass plus a reactive threshold: bypass at T >= 30 C, back at <= 28 C,
  no forecast, no limit checks, no charge caps.
* ``celldx``: the full :class:`~celldx.thermal.controller.ThermalController`.
* ablations of ``celldx``: ``no_forecast``, ``no_rotation``, ``no_charge_caps``,
  ``no_band_protection``, ``no_energy_check``, ``ambient_floor`` (floor = room sensor, no learning),
  and the proposed ``net_benefit`` gate (bypass a hot cell only if it lowers the pack's aging rate).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import Config, load_config
from .controller import CellReading, ThermalController

KEL = 273.15
RG = 8.314462618
DAY = 86400.0


# ---------------------------------------------------------------------------- cell + layout
@dataclass(frozen=True)
class NaIonCell:
    """Firebreak's generic 100 Ah prismatic Na-ion cell (stated assumptions, not a datasheet)."""

    q_nom_ah: float = 100.0
    v_min: float = 2.0
    v_max: float = 3.95
    v_nom: float = 3.1
    ocv_a: float = 2.55
    ocv_b: float = 1.15
    ocv_c: float = 0.07
    ocv_e: float = 0.01
    tail_lo: float = 0.6
    tail_hi: float = 0.3
    tail_w: float = 0.03
    r0_mohm: float = 1.2
    r1_mohm: float = 0.8
    tau1_s: float = 60.0
    ea_r: float = 20_000.0
    heat_capacity_j_k: float = 2500.0
    r_switch_mohm: float = 0.25

    def ocv(self, soc: np.ndarray) -> np.ndarray:
        s = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        e, w = self.ocv_e, self.tail_w
        return (self.ocv_a + self.ocv_b * s + self.ocv_c * np.log((s + e) / (1.0 + e - s))
                - self.tail_lo * np.exp(-s / w) + self.tail_hi * np.exp(-(1.0 - s) / w))

    def r_factor(self, temp_c: np.ndarray) -> np.ndarray:
        """Resistance multiplier vs 25 C (higher when cold)."""
        return np.exp(self.ea_r / RG * (1.0 / (np.asarray(temp_c) + KEL) - 1.0 / (25.0 + KEL)))


def grid_layout(rows: int = 4, cols: int = 8) -> dict[str, Any]:
    ids = [f"C{k + 1:02d}" for k in range(rows * cols)]
    pos = {cid: (k // cols, k % cols) for k, cid in enumerate(ids)}
    nb: dict[str, list[str]] = {}
    for cid, (r, c) in pos.items():
        nb[cid] = [ids[rr * cols + cc] for rr, cc in ((r - 1, c), (r, c - 1), (r, c + 1), (r + 1, c))
                   if 0 <= rr < rows and 0 <= cc < cols]
    return {"rows": rows, "cols": cols, "cell_ids": ids, "positions": pos, "neighbors": nb}


# ---------------------------------------------------------------------------- scenarios
@dataclass
class AirEvent:
    """A local air-temperature excursion (e.g. inverter exhaust after a fan failure)."""

    offsets: dict[str, float]
    start_s: float
    ramp_s: float
    end_s: float


@dataclass
class Scenario:
    name: str
    description: str
    days: int = 3
    room_min_c: float = 24.0
    room_max_c: float = 30.0
    room_peak_hour: float = 15.0
    pockets: dict[str, float] = field(default_factory=dict)          # cell -> local air offset (K)
    weak: dict[str, tuple[float, float]] = field(default_factory=dict)  # cell -> (SOH, resistance multiplier)
    schedule: list[tuple[float, float, float]] = field(default_factory=list)  # (start h, end h, W; + discharge)
    regulation_w: float = 0.0                                        # std of an energy-neutral regulation signal
    regulation_hours: tuple[float, float] = (7.0, 22.0)
    regulation_tau_s: float = 300.0
    soc0: float = 0.30
    events: list[AirEvent] = field(default_factory=list)
    g_neighbour_w_k: float = 1.0      # cell-to-cell contact conductance (Firebreak: 1.0 W/K, cells touching)
    reserve_wh: float = 0.0           # extra energy the site keeps for services (regulation headroom)


HOT_POCKET = {"C07": 5.0, "C08": 5.0, "C15": 5.0, "C16": 5.0,          # next to the inverter exhaust
              "C06": 2.5, "C14": 2.5, "C23": 2.5, "C24": 2.5}           # its fringe
WEAK = {"C20": (0.86, 1.6), "C15": (0.88, 1.4)}                         # one weak cell inside the pocket
DAILY = [(10.0, 15.0, -1700.0), (17.5, 20.0, 2800.0)]                    # 8.5 kWh solar charge, 7.0 kWh at 0.28C
PEAK = [(10.0, 15.0, -1100.0), (18.0, 18.9, 5000.0)]                     # 5.5 kWh solar charge, 4.5 kWh at 0.5C
EXHAUST = AirEvent({"C07": 45.0, "C08": 45.0, "C16": 45.0, "C06": 18.0, "C15": 18.0, "C24": 18.0},
                   start_s=DAY + 12 * 3600, ramp_s=1800.0, end_s=DAY + 18 * 3600)


def scenarios() -> dict[str, Scenario]:
    """The scenario set used in the validation report (3 days each; day 1 is floor learning)."""
    common: dict[str, Any] = {"pockets": dict(HOT_POCKET), "weak": dict(WEAK), "schedule": list(DAILY)}
    return {
        "hot_pocket": Scenario(
            "hot_pocket", "Economy-HVAC room 22-28 C; an inverter-side hot-air pocket (+5 K, fringe +2.5 K); two weak "
            "cells (one in the pocket); solar charge 10-15 h at 1.7 kW, evening discharge 17:30-20:00 at 2.8 kW (0.28C).",
            room_min_c=22.0, room_max_c=28.0, **common),
        "hot_pocket_regulation": Scenario(
            "hot_pocket_regulation", "hot_pocket plus an energy-neutral frequency-regulation signal (std 1.5 kW, "
            "07-22 h; the site keeps 1.5 kWh headroom for it): more self-heating, and a request that changes sign every "
            "few minutes.",
            room_min_c=22.0, room_max_c=28.0, regulation_w=1500.0, reserve_wh=1500.0, **common),
        "peak_shave": Scenario(
            "peak_shave", "hot_pocket room and cells with a 0.5C evening peak (5 kW, 18:00-18:54, 4.5 kWh) on a 5.5 kWh "
            "solar charge: large self-heating and energy slack, so the limits leave room to bypass.",
            room_min_c=22.0, room_max_c=28.0, pockets=dict(HOT_POCKET), weak=dict(WEAK), schedule=list(PEAK)),
        "bad_weld": Scenario(
            "bad_weld", "peak_shave, but the pocket cell C15 has a degraded weld (3.5x resistance, SOH 0.88): its own "
            "I^2R heat dominates, which is the case where taking one cell out pays off.",
            room_min_c=22.0, room_max_c=28.0, pockets=dict(HOT_POCKET), weak={"C20": (0.86, 1.6), "C15": (0.88, 3.5)},
            schedule=list(PEAK)),
        "peak_shave_spaced": Scenario(
            "peak_shave_spaced", "peak_shave in a pack whose cells are spaced or in separate modules (cell-to-cell "
            "contact 0.2 W/K instead of 1.0): a bypassed cell can cool toward its own floor.",
            room_min_c=22.0, room_max_c=28.0, pockets=dict(HOT_POCKET), weak=dict(WEAK),
            schedule=list(PEAK), g_neighbour_w_k=0.2),
        "heatwave": Scenario(
            "heatwave", "hot_pocket during a heatwave the HVAC cannot hold: room 26-33 C, so most cells reach the band "
            "and the limits bind.", room_min_c=26.0, room_max_c=33.0, **common),
        "exhaust_fault": Scenario(
            "exhaust_fault", "hot_pocket plus an inverter fan failure on day 2, 12:00-18:00, that blows exhaust air "
            "(+45 K, 67-73 C air) onto C07/C08/C16 (+18 K on C06/C15/C24): those cells reach the 57 C cutoff.",
            room_min_c=22.0, room_max_c=28.0, events=[EXHAUST], **common),
        "cold_morning": Scenario(
            "cold_morning", "Winter room 10-20 C with the same pocket and schedule: tests the 15 C charge minimum and "
            "shows what it costs.", room_min_c=10.0, room_max_c=20.0, soc0=0.35, **common),
    }


# ---------------------------------------------------------------------------- plant
class ThermalPack:
    """Truth simulator. ``advance`` integrates one control period with explicit substeps."""

    def __init__(self, sc: Scenario, seed: int, cell: NaIonCell | None = None, layout: Mapping[str, Any] | None = None,
                 v_window: tuple[float, float] = (70.0, 130.0), i_max_a: float = 100.0) -> None:
        self.sc, self.cell = sc, cell or NaIonCell()
        lay = dict(layout or grid_layout())
        self.layout = lay
        self.ids: list[str] = list(lay["cell_ids"])
        n = self.n = len(self.ids)
        idx = {c: k for k, c in enumerate(self.ids)}
        rng = np.random.default_rng(seed)
        self.soh = 1.0 - np.abs(rng.normal(0.0, 0.015, n))
        rmult = np.ones(n)
        for cid, (soh, rm) in sc.weak.items():
            self.soh[idx[cid]] = soh
            rmult[idx[cid]] = rm
        self.q_ah = self.cell.q_nom_ah * self.soh
        self.r0 = self.cell.r0_mohm * 1e-3 * (1 + rng.normal(0, 0.05, n)) * rmult
        self.r1 = self.cell.r1_mohm * 1e-3 * (1 + rng.normal(0, 0.05, n)) * rmult
        self.cth = self.cell.heat_capacity_j_k * (1 + rng.normal(0, 0.05, n))
        rows, cols = int(lay["rows"]), int(lay["cols"])
        self.g_amb = np.empty(n)
        G = np.zeros((n, n))
        for cid in self.ids:
            k = idx[cid]
            r, c = lay["positions"][cid]
            edges = int(r in (0, rows - 1)) + int(c in (0, cols - 1))
            base = 0.80 if edges >= 2 else 0.55 if edges == 1 else 0.25
            self.g_amb[k] = base * (1 + rng.normal(0, 0.10))
            for other in lay["neighbors"][cid]:
                j = idx[other]
                if j > k:
                    g = sc.g_neighbour_w_k * (1 + rng.normal(0, 0.10))
                    G[k, j] = G[j, k] = g
        self.L = np.diag(G.sum(1)) - G
        self.offset = np.array([sc.pockets.get(c, 0.0) for c in self.ids])
        self.event_offsets = [np.array([ev.offsets.get(c, 0.0) for c in self.ids]) for ev in sc.events]
        self.v_window, self.i_max = v_window, i_max_a
        self.soc = np.clip(sc.soc0 + rng.normal(0, 0.02, n), 0.06, 0.94)
        self.v1 = np.zeros(n)
        self.T = self.equilibrium(0.0)
        self.t = 0.0
        self.is_weak = np.array([c in sc.weak for c in self.ids])
        self.soc_bias = rng.normal(0.0, 0.01, n)     # BMS SOC estimate error: slow per-cell bias, not white noise

    # environment
    def room(self, t_s: float) -> float:
        sc = self.sc
        h = (t_s % DAY) / 3600.0
        mid, amp = 0.5 * (sc.room_max_c + sc.room_min_c), 0.5 * (sc.room_max_c - sc.room_min_c)
        return mid + amp * math.cos(2 * math.pi * (h - sc.room_peak_hour) / 24.0)

    def air(self, t_s: float) -> np.ndarray:
        a = self.room(t_s) + self.offset
        for ev, off in zip(self.sc.events, self.event_offsets, strict=True):
            if ev.start_s <= t_s < ev.end_s + ev.ramp_s:
                f = min(1.0, (t_s - ev.start_s) / ev.ramp_s) if t_s < ev.end_s else max(0.0, 1 - (t_s - ev.end_s) / ev.ramp_s)
                a = a + f * off
        return a

    def equilibrium(self, t_s: float) -> np.ndarray:
        """Idle steady state of the thermal network: what each cell's floor really is."""
        A = np.diag(self.g_amb) + self.L
        return np.asarray(np.linalg.solve(A, self.g_amb * self.air(t_s)))

    # electrics
    def _resist(self) -> tuple[np.ndarray, np.ndarray]:
        f = self.cell.r_factor(self.T)
        return self.r0 * f, self.r1 * f

    def operating_point(self, p_w: float, on: np.ndarray, switches: bool) -> tuple[float, float, str]:
        """(current, string voltage, status) for constant-power operation with physical limits."""
        if not on.any() or abs(p_w) < 1e-9:
            e = float((self.cell.ocv(self.soc) - self.v1)[on].sum())
            return 0.0, e, "idle" if abs(p_w) < 1e-9 else "no_cells"
        r0, _ = self._resist()
        ocv = self.cell.ocv(self.soc)
        E = float((ocv - self.v1)[on].sum())
        R = float(r0[on].sum() + (self.cell.r_switch_mohm * 1e-3 * on.sum() if switches else 0.0))
        disc = E * E - 4 * R * p_w
        status = "ok"
        if disc < 0:
            i = E / (2 * R)
            status = "power_limited"
        else:
            i = (E - math.sqrt(disc)) / (2 * R)
        if abs(i) > self.i_max:
            i = math.copysign(self.i_max, i)
            status = "current_limited"
        # cell terminal cut-offs (plant protection)
        vt = ocv - self.v1 - i * r0
        if i > 0 and (vt[on] < self.cell.v_min).any():
            imax = float(np.min(((ocv - self.v1 - self.cell.v_min) / r0)[on]))
            i = max(0.0, imax)
            status = "cell_v_min_taper"
        elif i < 0 and (vt[on] > self.cell.v_max).any():
            imin = float(np.max(((ocv - self.v1 - self.cell.v_max) / r0)[on]))
            i = min(0.0, imin)
            status = "cell_v_max_taper"
        v = E - i * R
        if abs(i) > 0 and not (self.v_window[0] <= v <= self.v_window[1]):
            return 0.0, E, "inverter_out_of_window"
        return float(i), float(v), status

    def advance(self, dt_total: float, p_w: float, on: np.ndarray, switches: bool, n_sub: int = 3) -> dict[str, Any]:
        dt = dt_total / n_sub
        e_out = 0.0
        statuses: list[str] = []
        i_rms = 0.0
        v_last = 0.0
        heat_total = np.zeros(self.n)
        for _ in range(n_sub):
            i, v, st = self.operating_point(p_w, on, switches)
            statuses.append(st)
            r0, r1 = self._resist()
            cur = np.where(on, i, 0.0)
            a = math.exp(-dt / self.cell.tau1_s)
            self.v1 = self.v1 * a + cur * r1 * (1 - a)
            heat = cur ** 2 * r0 + self.v1 ** 2 / r1
            if switches:
                heat = heat + i * i * self.cell.r_switch_mohm * 1e-3   # in-string or bypassed slot carries I
            heat_total += heat * dt
            air = self.air(self.t)
            dT = (heat - self.g_amb * (self.T - air) - self.L @ self.T) / self.cth
            self.T = self.T + dT * dt
            self.soc = self.soc - cur * dt / (3600.0 * self.q_ah)
            e_out += i * v * dt / 3600.0
            i_rms += i * i * dt
            v_last = v
            self.t += dt
        bad = [s for s in statuses if s not in ("ok", "idle")]
        return {"energy_wh": e_out, "power_w": e_out * 3600.0 / dt_total, "i_rms": math.sqrt(i_rms / dt_total),
                "v": v_last, "status": bad[-1] if bad else statuses[-1], "heat_j": heat_total}

    def measure(self, rng: np.random.Generator, on: np.ndarray, i_last: float) -> tuple[list[CellReading], float]:
        r0, r1 = self._resist()
        ocv = self.cell.ocv(self.soc)
        cur = np.where(on, i_last, 0.0)
        v = ocv - self.v1 - cur * r0
        n = self.n
        T_m = self.T + rng.normal(0, 0.1, n)
        V_m = v + rng.normal(0, 0.002, n)
        I_m = cur + np.where(on, rng.normal(0, 0.05, n), 0.0)
        soc_m = np.clip(self.soc + self.soc_bias + rng.normal(0, 0.002, n), 0, 1)
        R_m = (r0 + r1) * (1 + rng.normal(0, 0.03, n))
        cells = [CellReading(id=c, temp_c=float(T_m[k]), voltage_v=float(V_m[k]), current_a=float(I_m[k]),
                             soc=float(soc_m[k]), soh=float(self.soh[k]), capacity_ah=float(self.q_ah[k]),
                             r_ohm=float(R_m[k]), v_nominal=self.cell.v_nom) for k, c in enumerate(self.ids)]
        return cells, float(self.room(self.t) + rng.normal(0, 0.1))


# ---------------------------------------------------------------------------- schedule
def request_profile(sc: Scenario, seed: int, period_s: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """(request W per control step, scheduled-only W per step) for the whole scenario."""
    steps = int(sc.days * DAY / period_s)
    t = np.arange(steps) * period_s
    h = (t % DAY) / 3600.0
    sched = np.zeros(steps)
    for a, b, p in sc.schedule:
        sched[(h >= a) & (h < b)] += p
    req = sched.copy()
    if sc.regulation_w > 0:
        rng = np.random.default_rng(seed + 7_777)
        phi = math.exp(-period_s / sc.regulation_tau_s)
        x = np.zeros(steps)
        for k in range(1, steps):
            x[k] = phi * x[k - 1] + math.sqrt(1 - phi * phi) * rng.normal()
        active = (h >= sc.regulation_hours[0]) & (h < sc.regulation_hours[1])
        reg = sc.regulation_w * x * active
        # energy-neutral per day
        for d in range(sc.days):
            m = active & (t >= d * DAY) & (t < (d + 1) * DAY)
            if m.any():
                reg[m] -= reg[m].mean()
        req = req + reg
    return req, sched


def remaining_discharge_profile(sched_w: np.ndarray, period_s: float, days: int) -> np.ndarray:
    """Scheduled discharge energy (Wh) still to come today, from each step to midnight: the
    "rest of the day's schedule" the in-string cells must be able to cover. Regulation is
    energy-neutral and not included. Pass ``-sched_w`` for the scheduled charge still to come."""
    per_day = len(sched_w) // days
    out = np.zeros(len(sched_w))
    for d in range(days):
        e = np.clip(sched_w[d * per_day:(d + 1) * per_day], 0, None) * period_s / 3600.0
        out[d * per_day:(d + 1) * per_day] = np.cumsum(e[::-1])[::-1]
    return out


# ---------------------------------------------------------------------------- strategies
class Strategy:
    name = "base"
    switches = True

    def decide(self, t_s: float, cells: list[CellReading], request_w: float, ambient_c: float,
               remaining_wh: float, remaining_charge_wh: float | None = None) -> tuple[np.ndarray, float, dict[str, Any]]:
        raise NotImplementedError


class FixedStringBMS(Strategy):
    name = "fixed"
    switches = False

    def __init__(self, cfg: Config, ids: list[str]) -> None:
        self.th = cfg.thermal
        self.n = len(ids)
        self.tripped = False

    def decide(self, t_s: float, cells: list[CellReading], request_w: float, ambient_c: float,
               remaining_wh: float, remaining_charge_wh: float | None = None) -> tuple[np.ndarray, float, dict[str, Any]]:
        th = self.th
        T = np.array([c.temp_c for c in cells])
        soc = np.array([c.soc for c in cells])
        if T.max() >= float(th.hard_cutoff_c):
            self.tripped = True
        elif self.tripped and T.max() <= float(th.hard_cutoff_release_c):
            self.tripped = False
        g = request_w
        why = "ok"
        if self.tripped:
            g, why = 0.0, "string tripped at the hard cutoff"
        elif request_w < 0 and (T.min() < float(th.charge_min_temp_c) or T.max() > float(th.charge_max_temp_c)):
            g, why = 0.0, "charge blocked: a cell outside the charge temperature window"
        elif request_w < 0 and soc.max() >= float(th.charge_soc_max):
            g, why = 0.0, "charge stopped: first cell full"
        elif request_w > 0 and soc.min() <= float(th.discharge_soc_min):
            g, why = 0.0, "discharge stopped: first cell empty"
        return np.ones(self.n, bool), float(g), {"why": why, "actions": ["IN"] * self.n}


class ControllerStrategy(Strategy):
    def __init__(self, name: str, cfg: Config, ids: list[str], *, ambient_floor: bool = False) -> None:
        self.name = name
        self.ctrl = ThermalController(cfg, ids)
        self.ambient_floor = ambient_floor
        self.last: Any = None

    def decide(self, t_s: float, cells: list[CellReading], request_w: float, ambient_c: float,
               remaining_wh: float, remaining_charge_wh: float | None = None) -> tuple[np.ndarray, float, dict[str, Any]]:
        d = self.ctrl.step(t_s, cells, request_w, ambient_c=ambient_c, remaining_energy_wh=remaining_wh,
                           remaining_charge_wh=remaining_charge_wh)
        self.last = d
        on = np.array([c.in_string for c in d.cells])
        return on, d.granted_w, {"decision": d, "actions": [c.action for c in d.cells]}


class NaiveThreshold(Strategy):
    """Reactive bypass at the band edge with the same per-cell safety rules, no limit checks."""

    name = "naive"

    def __init__(self, cfg: Config, ids: list[str], on_c: float = 30.0, off_c: float = 28.0) -> None:
        self.th = cfg.thermal
        self.n = len(ids)
        self.off = np.zeros(self.n, bool)
        self.reason = ["in"] * self.n
        self.on_c, self.off_c = on_c, off_c

    def decide(self, t_s: float, cells: list[CellReading], request_w: float, ambient_c: float,
               remaining_wh: float, remaining_charge_wh: float | None = None) -> tuple[np.ndarray, float, dict[str, Any]]:
        th = self.th
        acts = []
        for k, c in enumerate(cells):
            r = self.reason[k]
            if c.temp_c >= float(th.hard_cutoff_c) or (r == "BYPASS_CUTOFF" and c.temp_c > float(th.hard_cutoff_release_c)):
                a = "BYPASS_CUTOFF"
            elif request_w > 0 and c.soc <= float(th.discharge_soc_min):
                a = "BYPASS_SOC_LIMIT"
            elif request_w < 0 and (c.temp_c < float(th.charge_min_temp_c) or c.temp_c > float(th.charge_max_temp_c)
                                    or c.soc >= float(th.charge_soc_max)):
                a = "BYPASS_CHARGE_RULE"
            elif c.temp_c >= self.on_c or (r == "BYPASS_HOT" and c.temp_c > self.off_c):
                a = "BYPASS_HOT"
            else:
                a = "IN"
            self.reason[k] = a if a != "IN" else "in"
            acts.append(a)
        on = np.array([a == "IN" for a in acts])
        return on, float(request_w), {"actions": acts}


ABLATIONS: dict[str, dict[str, Any]] = {
    "no_forecast": {"thermal.trend_horizon_s": 0.0},
    "no_rotation": {"thermal.rotation_quantum_s": 1e15},
    "no_charge_caps": {"thermal.charge_soc_caps": []},
    "no_band_protection": {"thermal.check_band_protection": False},
    "no_energy_check": {"thermal.check_energy": False},
    "ambient_floor": {"thermal.floor_min_idle_points": 10**9},
    "net_benefit": {"thermal.require_net_aging_benefit": True},
}
STRATEGIES = ["fixed", "safety_bypass", "naive", "celldx", *ABLATIONS]


def make_strategy(name: str, ids: list[str], cfg: Config | None = None) -> Strategy:
    cfg = cfg or load_config()
    if name == "fixed":
        return FixedStringBMS(cfg, ids)
    if name == "naive":
        return NaiveThreshold(cfg, ids)
    if name == "safety_bypass":
        c2 = cfg.with_overrides({"thermal.band_edge_c": 1e9, "thermal.charge_soc_caps": []})
        return ControllerStrategy(name, c2, ids)
    if name == "celldx":
        return ControllerStrategy(name, cfg, ids)
    if name in ABLATIONS:
        return ControllerStrategy(name, cfg.with_overrides(ABLATIONS[name]), ids, ambient_floor=name == "ambient_floor")
    raise ValueError(f"unknown strategy {name!r}; choose from {STRATEGIES}")


# ---------------------------------------------------------------------------- run + metrics
def run(scenario: str | Scenario, strategy: str, seed: int = 0, *, cfg: Config | None = None,
        keep_trace: bool = False, on_step: Callable[[int, Any], None] | None = None) -> dict[str, Any]:
    """Simulate one scenario under one strategy. Returns metrics (and the trace if asked)."""
    cfg = cfg or load_config()
    sc = scenarios()[scenario] if isinstance(scenario, str) else scenario
    th = cfg.thermal
    period = float(th.control_period_s)
    plant = ThermalPack(sc, seed, v_window=(float(th.inverter_v_min), float(th.inverter_v_max)),
                        i_max_a=float(th.cell_current_max_a))
    ids = plant.ids
    strat = make_strategy(strategy, ids, cfg)
    req, sched = request_profile(sc, seed, period)
    reserve = remaining_discharge_profile(sched, period, sc.days) + sc.reserve_wh
    charge_left = remaining_discharge_profile(-sched, period, sc.days)
    meas_rng = np.random.default_rng(seed + 31_337)
    steps = len(req)
    n = plant.n
    rec_T = np.zeros((steps, n))
    rec_soc = np.zeros((steps, n))
    rec_on = np.zeros((steps, n), bool)
    rec_cur = np.zeros((steps, n))
    rec_act: list[list[str]] = []
    rec_p = np.zeros(steps)
    rec_g = np.zeros(steps)
    rec_v = np.zeros(steps)
    rec_i = np.zeros(steps)
    rec_status: list[str] = []
    rec_cap = np.full((steps, n), np.nan)
    floors_err: list[float] = []
    floors_err_air: list[float] = []
    amb_err: list[float] = []
    rot = np.zeros(steps, bool)
    ctrl_ok = np.ones(steps, bool)
    derated = np.zeros(steps, bool)
    onset_T: list[float] = []
    onset_pred: list[float] = []
    on = np.ones(n, bool)
    i_last = 0.0
    prev_act = ["IN"] * n
    for k in range(steps):
        t = k * period
        cells, amb = plant.measure(meas_rng, on, i_last)
        on, granted, info = strat.decide(t, cells, float(req[k]), amb, float(reserve[k]), float(charge_left[k]))
        acts = info["actions"]
        d = info.get("decision")
        if d is not None:
            rot[k] = d.rotation_active
            rec_cap[k] = [c.soc_cap for c in d.cells]
            ctrl_ok[k] = bool(d.checks["ok"])
            derated[k] = bool(d.checks["derated"])
            if k % 60 == 0 and t >= DAY:      # floor accuracy once an hour after the first day
                eq = plant.equilibrium(t)
                air = plant.air(t)
                fl = np.array([c.floor_c for c in d.cells])
                floors_err.append(float(np.mean(np.abs(fl - eq))))
                floors_err_air.append(float(np.mean(np.abs(fl - air))))
                amb_err.append(float(np.mean(np.abs(plant.room(t) - eq))))
            for j, c in enumerate(d.cells):
                if c.action == "BYPASS_HOT" and prev_act[j] != "BYPASS_HOT":
                    onset_T.append(float(plant.T[j]))
                    onset_pred.append(c.temp_pred_c)
        prev_act = list(acts)
        rec_T[k] = plant.T
        rec_soc[k] = plant.soc
        rec_on[k] = on
        rec_act.append(list(acts))
        out = plant.advance(period, granted, on, strat.switches)
        i_last = out["power_w"] / out["v"] if out["v"] > 0 and abs(out["power_w"]) > 0 else 0.0
        rec_cur[k] = np.where(on, i_last, 0.0)
        rec_p[k], rec_g[k], rec_v[k], rec_i[k] = req[k], out["power_w"], out["v"], i_last
        rec_status.append(out["status"])
        if on_step is not None:
            on_step(k, info)
    res = metrics(plant, cfg, sc, period, rec_T, rec_soc, rec_on, rec_cur, rec_act, rec_p, rec_g, rec_v, rec_status,
                  rot, ctrl_ok, derated, onset_T, onset_pred, floors_err, floors_err_air, amb_err, strat.switches,
                  eff_caps=rec_cap)
    res.update(scenario=sc.name, strategy=strategy, seed=seed)
    if keep_trace:
        res["trace"] = {"ids": ids, "T": rec_T, "soc": rec_soc, "on": rec_on, "actions": rec_act, "request_w": rec_p,
                        "delivered_w": rec_g, "string_v": rec_v, "status": rec_status, "period_s": period,
                        "room": np.array([plant.room(k * period) for k in range(steps)]),
                        "air": np.array([plant.air(k * period) for k in range(0, steps, 60)])}
    return res


def _jain(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(x.sum() ** 2 / (len(x) * (x * x).sum())) if len(x) and (x * x).sum() > 0 else float("nan")


def metrics(plant: ThermalPack, cfg: Config, sc: Scenario, period: float, T: np.ndarray, soc: np.ndarray, on: np.ndarray,
            cur: np.ndarray, acts: list[list[str]], req: np.ndarray, deliv: np.ndarray, v: np.ndarray, status: list[str],
            rot: np.ndarray, ctrl_ok: np.ndarray, derated: np.ndarray, onset_T: list[float], onset_pred: list[float],
            floors_err: list[float], floors_err_air: list[float], amb_err: list[float], switches: bool,
            eff_caps: np.ndarray | None = None) -> dict[str, Any]:
    th = cfg.thermal
    band = float(th.band_edge_c)
    hrs = period / 3600.0
    n = plant.n
    steps = len(req)
    days = steps * period / DAY
    dis = req > 0
    chg = req < 0
    req_dis_kwh = float(req[dis].sum() * hrs / 1000)
    del_dis_kwh = float(np.clip(deliv[dis], 0, None).sum() * hrs / 1000)
    req_chg_kwh = float(-req[chg].sum() * hrs / 1000)
    del_chg_kwh = float(-np.clip(deliv[chg], None, 0).sum() * hrs / 1000)
    shortfall = dis & (deliv < 0.99 * req)
    hot = T > band
    full = soc > 0.8
    pocket = plant.offset > 0
    healthy = (~pocket) & (~plant.is_weak)
    # aging proxies (model-based; see docs): Firebreak SEI law and the controller's own ranking law
    c_rate = np.abs(cur) / plant.q_ah
    ch = (cur < 0) * c_rate
    dch = (cur > 0) * c_rate
    arr50 = np.exp(50_000 / RG * (1 / (25 + KEL) - 1 / (T + KEL)))
    sei = arr50 * np.exp(1.5 * (soc - 0.5)) * (1 + 2.0 * ch + 0.5 * dch)
    arr30 = np.exp(float(th.aging_ea_j_per_mol) / 8.314 * (1 / (25 + KEL) - 1 / (T + KEL)))
    own = arr30 * np.exp(float(th.aging_k_soc) * (soc - float(th.aging_soc_ref))) / plant.soh ** float(th.aging_soh_gamma)
    sei_days = sei.sum(0) * period / DAY      # "days at 25 C, 50 % SOC, rest" per cell
    own_days = own.sum(0) * period / DAY
    acts_arr = np.array(acts)
    sw = (on[1:] != on[:-1])
    switch_events = int(sw.sum())
    # chatter: a cell switched back within 5 minutes
    chatter = 0
    win = max(1, int(300 / period))
    for j in range(n):
        idx = np.where(sw[:, j])[0]
        chatter += int((np.diff(idx) < win).sum()) if len(idx) > 1 else 0
    charging_cells = (cur < -1e-6)
    cmax = float(th.charge_max_temp_c)
    # true floors (idle equilibrium of the network) -> the charge cap each cell should have had
    A = np.diag(plant.g_amb) + plant.L
    M = np.linalg.solve(A, np.diag(plant.g_amb))
    air_all = np.array([plant.air(k * period) for k in range(steps)])
    eq_all = air_all @ M.T
    caps = np.full((steps, n), float(th.charge_soc_max))
    for rule in sorted(th.charge_soc_caps, key=lambda r: r["floor_c"]):
        caps = np.where(eq_all >= float(rule["floor_c"]), np.minimum(caps, float(rule["soc_max"])), caps)
    hot_bypass_steps = (acts_arr == "BYPASS_HOT")
    waiting = (acts_arr == "HOT_WAITING")
    ever_wait = waiting.any(0) | hot_bypass_steps.any(0)
    off_time_cand = hot_bypass_steps.sum(0)[ever_wait] * period / 3600
    # truth-checked constraint outcomes
    viol = {
        "inverter_out_of_window_min": int(sum(s == "inverter_out_of_window" for s in status)),
        "current_limited_min": int(sum(s == "current_limited" for s in status)),
        "power_limited_min": int(sum(s == "power_limited" for s in status)),
        "cell_voltage_taper_min": int(sum(s.startswith("cell_v_") for s in status)),
        "carrying_current_at_or_above_cutoff_cell_min": int((on & (np.abs(cur) > 0.5) & (T >= float(th.hard_cutoff_c))).sum()),
        "charging_below_min_temp_cell_min": int((charging_cells & (T < float(th.charge_min_temp_c))).sum()),
        "charging_above_charge_limit_cell_min": int((charging_cells & (T > cmax)).sum()),
        "charging_above_soc_cap_cell_min": int((charging_cells & (soc > caps + 0.02)).sum()),
    }
    if eff_caps is not None and np.isfinite(eff_caps).any():
        ec = np.where(np.isfinite(eff_caps), eff_caps, 1.0)
        viol["charging_above_effective_cap_cell_min"] = int((charging_cells & (soc > ec + 0.02)).sum())
        viol["cap_lifted_for_energy_cell_min"] = int(((ec > caps + 1e-9) & charging_cells).sum())
    out = {
        "days": days,
        "energy": {
            "requested_discharge_kwh": req_dis_kwh, "delivered_discharge_kwh": del_dis_kwh,
            "served_discharge_frac": del_dis_kwh / req_dis_kwh if req_dis_kwh else float("nan"),
            "requested_charge_kwh": req_chg_kwh, "accepted_charge_kwh": del_chg_kwh,
            "discharge_shortfall_min": int(shortfall.sum()),
            "discharge_minutes": int(dis.sum()),
        },
        "thermal": {
            "cell_hours_above_band": float(hot.sum() * hrs),
            "degree_hours_above_band": float(np.clip(T - band, 0, None).sum() * hrs),
            "hot_and_full_cell_hours": float((hot & full).sum() * hrs),
            "pocket_cell_hours_above_band": float(hot[:, pocket].sum() * hrs),
            "healthy_cell_hours_above_band": float(hot[:, healthy].sum() * hrs),
            "max_temp_c": float(T.max()),
            "cell_min_at_or_above_55c": int((T >= 55).sum()),
            "max_hours_above_band_one_cell": float(hot.sum(0).max() * hrs),
            "mean_temp_spread_k": float((T.max(1) - T.min(1)).mean()),
        },
        "aging_proxy": {
            "sei_days_mean": float(sei_days.mean()), "sei_days_max": float(sei_days.max()),
            "sei_days_max_over_min": float(sei_days.max() / sei_days.min()),
            "sei_days_cv": float(sei_days.std() / sei_days.mean()),
            "own_model_days_mean": float(own_days.mean()), "own_model_days_max": float(own_days.max()),
            "own_model_cv": float(own_days.std() / own_days.mean()),
        },
        "switching": {
            "switch_events": switch_events, "switch_events_per_cell_day_max": float(sw.sum(0).max() / days),
            "chatter_events_lt5min": chatter,
            "bypass_cell_hours_hot": float(hot_bypass_steps.sum() * hrs),
        },
        "controller": {
            "steps_with_limit_binding_rotation": int(rot.sum()),
            "steps_controller_check_failed": int((~ctrl_ok).sum()),
            "steps_derated": int(derated.sum()),
            "hot_bypass_onsets": len(onset_T),
            "onsets_before_band_frac": float(np.mean(np.array(onset_T) < band)) if onset_T else float("nan"),
            "onset_true_temp_median_c": float(np.median(onset_T)) if onset_T else float("nan"),
            "rotation_jain_fairness": _jain(off_time_cand) if len(off_time_cand) > 1 else float("nan"),
            "floor_mae_vs_idle_equilibrium_k": float(np.mean(floors_err)) if floors_err else float("nan"),
            "floor_mae_vs_local_air_k": float(np.mean(floors_err_air)) if floors_err_air else float("nan"),
            "room_sensor_mae_vs_idle_equilibrium_k": float(np.mean(amb_err)) if amb_err else float("nan"),
        },
        "violations_truth": viol,
    }
    return out


def frame_from_plant(plant: ThermalPack, cells: list[CellReading], t_s: float, ambient_c: float, request_w: float) -> dict[str, Any]:
    """A Firebreak/REON live JSON frame for the current plant state (for replay files)."""
    return {"t": t_s, "ambient_C": round(ambient_c, 3), "request_W": round(request_w, 1),
            "cells": [{"id": c.id, "V": round(c.voltage_v, 4), "I": round(c.current_a, 3), "T": round(c.temp_c, 3),
                       "soc": round(c.soc, 4), "soh": round(c.soh, 4), "R_mOhm": round(c.r_ohm * 1000, 4),
                       "capacity_Ah": round(c.capacity_ah, 2)} for c in cells]}


__all__ = ["ABLATIONS", "STRATEGIES", "AirEvent", "NaIonCell", "Scenario", "ThermalPack", "remaining_discharge_profile",
           "frame_from_plant", "grid_layout", "make_strategy", "metrics", "request_profile", "run", "scenarios"]
