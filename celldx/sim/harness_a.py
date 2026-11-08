"""Simulator A: a reconstruction of the prototype's labelled harness (``realistic_test.py``).

The original file was not supplied. This module rebuilds it from the description in
section 10 of the brief: 18650-class NMC cells, R0 35 to 70 mOhm, Ea 25 to 35 kJ/mol, a 6 to
12 C seasonal swing, 1 to 2% measurement noise, mild calendar ageing on healthy cells,
730 daily samples and injected faults, 25 trials per scenario. Every functional form
below is our choice, written down so it can be compared with the original when available.

Each trial is a pack of ``n_cells`` cells in one string: cell ``C00`` carries the scenario,
the others are healthy neighbours (so false alarms on healthy cells are measured too).

Ground-truth label per scenario: see :data:`EXPECTED`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..physics import arrhenius_factor

SCENARIOS = ["healthy", "cold", "cooling", "plating", "internal_damage", "weld", "corrosion", "sei"]
EXPECTED = {
    "healthy": "healthy",
    "cold": "temperature_effect",
    "cooling": "cooling_fault",
    "plating": "lithium_plating",
    "internal_damage": "internal_damage",
    "weld": "mechanical_weld_or_tab",
    "corrosion": "mechanical_contact_corrosion",
    "sei": "sei_growth",
}
FAULT_SCENARIOS = {k for k, v in EXPECTED.items() if v not in ("healthy", "temperature_effect")}


@dataclass
class CellTruth:
    cell_id: str
    scenario: str
    label: str
    r0_ohm: float
    ea_j_per_mol: float
    params: dict[str, Any] = field(default_factory=dict)


def _cell(rng: np.random.Generator, n_days: int, scenario: str, t_amb: np.ndarray, ea: float, cell_id: str,
          severity: float | None = None) -> tuple[pd.DataFrame, CellTruth]:
    d = np.arange(n_days, dtype=float)
    life = d / (n_days - 1)
    r0 = rng.uniform(0.035, 0.070)
    noise = rng.uniform(0.01, 0.02)
    heat = rng.uniform(0.5, 1.5)
    cal_r = rng.uniform(0.01, 0.04)
    cal_soh = rng.uniform(0.005, 0.015)
    cal_ocv = rng.uniform(0.0, 3.0)
    t_cell = t_amb + heat + rng.normal(0, 0.3, n_days)
    t_pack = t_amb + rng.normal(0, 0.3, n_days)
    r25 = r0 * (1 + cal_r * np.sqrt(life))
    b_ind = np.zeros(n_days)  # temperature-independent resistance (ohm)
    soh = 1 - cal_soh * np.sqrt(life)
    ocv = cal_ocv * np.sqrt(life)
    cold = np.zeros(n_days)
    params: dict[str, Any] = {"r0": r0, "noise": noise, "cal_r": cal_r}
    sev = severity if severity is not None else rng.uniform(0.0, 1.0)
    if scenario == "cold":
        snap = rng.uniform(12, 18)
        t_cell[-60:] -= snap
        t_pack[-60:] -= snap
        params["cold_snap_k"] = snap
    elif scenario == "cooling":
        onset = int(rng.uniform(0.55, 0.85) * n_days)
        extra = 8 + 4 * sev
        ramp = np.clip((d - onset) / 30.0, 0, 1)
        t_cell = t_cell + extra * ramp
        params.update(onset=onset, extra_k=extra)
    elif scenario == "plating":
        onset = int(rng.uniform(0.55, 0.8) * n_days)
        n_ev = int(rng.integers(5, 16))
        ev = np.sort(rng.choice(np.arange(onset, min(n_days - 30, onset + 60)), n_ev, replace=False))
        cold[ev] = 1.0
        t_cell[ev] = np.minimum(t_cell[ev], rng.uniform(-2, 6, n_ev))
        rise_tot = 0.12 + 0.13 * sev
        per = rise_tot / n_ev
        cum = np.zeros(n_days)
        for e in ev:
            cum[e + 1 :] += per
        r25 = r25 * (1 + cum)
        soh = soh - (0.03 + 0.03 * sev) * (cum / rise_tot)
        ocv = ocv - (10 + 10 * sev) * (cum / rise_tot)
        params.update(onset=onset, n_events=n_ev, rise=rise_tot)
    elif scenario == "internal_damage":
        onset = int(rng.uniform(0.55, 0.9) * n_days)
        step = 0.20 + 0.20 * sev
        r25 = r25 * np.where(d >= onset, 1 + step, 1)
        soh = soh - np.where(d >= onset, 0.05 + 0.05 * sev, 0)
        ocv = ocv + np.where(d >= onset, 20 + 20 * sev, 0)
        params.update(onset=onset, step=step)
    elif scenario == "weld":
        onset = int(rng.uniform(0.55, 0.9) * n_days)
        step = 0.20 + 0.20 * sev
        b_ind = np.where(d >= onset, step * r0, 0.0)
        params.update(onset=onset, step=step)
    elif scenario == "corrosion":
        onset = int(rng.uniform(0.15, 0.4) * n_days)
        final = 0.20 + 0.20 * sev
        b_ind = r0 * final * np.clip((d - onset) / (n_days - 1 - onset), 0, 1)
        params.update(onset=onset, final=final)
    elif scenario == "sei":
        grow = 0.20 + 0.20 * sev
        r25 = r0 * (1 + (cal_r + grow) * np.sqrt(life))
        soh = 1 - (0.06 + 0.06 * sev) * np.sqrt(life)
        ocv = (10 + 10 * sev) * np.sqrt(life)
        params.update(grow=grow)
    r = (r25 * arrhenius_factor(t_cell, ea) + b_ind) * (1 + rng.normal(0, noise, n_days))
    df = pd.DataFrame({
        "t": d, "cell_id": cell_id, "resistance_ohm": r, "temp_cell_c": t_cell, "temp_pack_c": t_pack,
        "soc": rng.uniform(0.45, 0.55, n_days), "soh": soh + rng.normal(0, 0.002, n_days),
        "ocv_mv": ocv + rng.normal(0, 0.5, n_days), "current_a": np.full(n_days, 1.0), "cold_charge": cold,
        "ah_throughput": d * 4.0, "cycle_count": d,
    })
    return df, CellTruth(cell_id, scenario, EXPECTED[scenario], r0, ea, params)


def simulate_trial(scenario: str, seed: int, n_days: int = 730, n_cells: int = 8, severity: float | None = None,
                   pack_id: str = "P0") -> tuple[pd.DataFrame, list[CellTruth]]:
    """One pack: cell C00 carries ``scenario``; C01.. are healthy neighbours.

    Ambient: ``20 + A sin(2 pi (d - phase)/365) + N(0, 1.5)`` C with A in [6, 12]. For the
    cold scenario the last 60 days get a 12 to 18 K cold snap (the cell is healthy).
    """
    rng = np.random.default_rng(seed)
    amp = rng.uniform(6, 12)
    phase = rng.uniform(0, 365)
    d = np.arange(n_days, dtype=float)
    t_amb = 20 + amp * np.sin(2 * np.pi * (d - phase) / 365.0) + rng.normal(0, 1.5, n_days)
    ea = rng.uniform(25e3, 35e3)
    frames, truths = [], []
    for k in range(n_cells):
        scen = scenario if k == 0 else "healthy"
        f, tr = _cell(rng, n_days, scen, t_amb.copy(), ea, f"C{k:02d}", severity if k == 0 else None)
        f["pack_id"] = pack_id
        f["string_id"] = "S0"
        frames.append(f)
        truths.append(tr)
    # cold scenario: only C00's cell and pack sensors see the cold snap; its neighbours stay at ambient,
    # the harder case (C00 then also diverges from its neighbours)
    return pd.concat(frames, ignore_index=True), truths
