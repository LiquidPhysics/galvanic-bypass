"""Simulator B: written independently of simulator A, with different functional forms, used
to test the diagnosis WITHOUT retuning (validation step 2 of section 10).

Differences from simulator A (deliberate model mismatch with what celldx assumes):

* resistance is two components with different activation energies (ohmic 12 to 22 kJ/mol,
  charge-transfer 40 to 60 kJ/mol), so the total does not follow one Arrhenius curve;
* a real SOC dependence (U-shaped) while the baseline map is flat;
* healthy ageing follows t^0.75, SEI follows (Arrhenius-weighted time)^0.6 (not sqrt t);
* measurement noise is AR(1)-correlated with Student-t (df 4) innovations;
* correlated sensor drift: a pack-wide temperature-sensor random walk plus per-cell offsets;
* dropouts: 5 to 15% of days missing at random plus one 10 to 40 day outage;
* weather is AR(1) with heatwaves; seasonal amplitude 4 to 14 C;
* weld loosens in two steps, corrosion grows exponentially, internal damage has a precursor,
  plating partially recovers, cooling faults worsen on high-current days.

Optional ``chemistry='na_ion'`` gives REON-like sodium-ion prismatic cells (0.5 to 1.5 mOhm,
100 Ah); the default is 18650-class NMC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .harness_a import EXPECTED, SCENARIOS, CellTruth

__all__ = ["simulate_trial", "SCENARIOS", "EXPECTED"]

KEL = 273.15


def _arr(t_c: np.ndarray, ea: float) -> np.ndarray:
    return np.exp(ea / 8.314 * (1.0 / (t_c + KEL) - 1.0 / 298.15))


def _ar1(rng: np.random.Generator, n: int, phi: float, scale: float, df: float | None = None) -> np.ndarray:
    e = rng.standard_t(df, n) * scale / np.sqrt(df / (df - 2)) if df else rng.normal(0, scale, n)
    x = np.zeros(n)
    for k in range(1, n):
        x[k] = phi * x[k - 1] + np.sqrt(1 - phi * phi) * e[k]
    x[0] = e[0]
    return x


def simulate_trial(scenario: str, seed: int, n_days: int = 730, n_cells: int = 12, chemistry: str = "nmc",
                   severity: float | None = None, pack_id: str = "PB") -> tuple[pd.DataFrame, list[CellTruth]]:
    rng = np.random.default_rng(seed + 7_000_003)
    d = np.arange(n_days, dtype=float)
    amp = rng.uniform(4, 14)
    t_amb = 18 + amp * np.sin(2 * np.pi * (d - rng.uniform(0, 365)) / 365) + _ar1(rng, n_days, 0.8, 2.5)
    for _ in range(int(rng.integers(0, 3))):  # heatwaves
        c = int(rng.integers(0, n_days - 10))
        t_amb[c : c + int(rng.integers(4, 10))] += rng.uniform(5, 9)
    drift_pack = np.cumsum(rng.normal(0, 0.03, n_days))  # shared temperature-sensor drift (K)
    ea_o = rng.uniform(12e3, 22e3)
    ea_ct = rng.uniform(40e3, 60e3)
    na = chemistry == "na_ion"
    cap = 100.0 if na else 2.5
    keep = rng.random(n_days) > rng.uniform(0.05, 0.15)
    o0 = int(rng.integers(30, n_days - 60))
    keep[o0 : o0 + int(rng.integers(10, 40))] = False
    keep[:20] = True  # commissioning period always logged
    current = np.abs(rng.normal(0.4, 0.15, n_days)) * cap
    frames, truths = [], []
    for k in range(n_cells):
        scen = scenario if k == 0 else "healthy"
        sev = (severity if severity is not None else rng.uniform(0, 1)) if k == 0 else 0.0
        r0 = rng.uniform(0.5e-3, 1.5e-3) if na else rng.uniform(0.030, 0.075)
        f_ct = rng.uniform(0.4, 0.6)
        heat = rng.uniform(0.3, 1.2)
        sig_n = rng.uniform(0.008, 0.025)
        t_cell = t_amb + heat * (current / cap / 0.4) ** 2 + rng.normal(0, 0.4, n_days)
        soc = np.clip(rng.uniform(0.2, 0.9, n_days), 0, 1)
        soc_f = 1 + 0.6 * np.clip(0.3 - soc, 0, None) / 0.3 + 0.1 * np.clip(soc - 0.85, 0, None) / 0.15
        life = d / (n_days - 1)
        age = 1 + rng.uniform(0.02, 0.06) * life ** 0.75
        ro = r0 * (1 - f_ct) * np.ones(n_days)
        rct = r0 * f_ct * np.ones(n_days)
        b_ind = np.zeros(n_days)
        soh = 1 - rng.uniform(0.005, 0.02) * life
        ocv = rng.uniform(0, 2.5) * life
        cold = np.zeros(n_days)
        params: dict[str, Any] = {"severity_draw": sev}
        if scen == "cold":
            L = int(rng.integers(45, 76))
            snap = rng.uniform(10, 20)
            t_cell[-L:] -= snap
            params.update(snap=snap)
        elif scen == "cooling":
            on = int(rng.uniform(0.5, 0.85) * n_days)
            extra = 7 + 5 * sev
            ramp = np.clip((d - on) / 60.0, 0, 1) ** 1.5
            t_cell = t_cell + extra * ramp * (0.7 + 0.3 * current / current.mean())
        elif scen == "plating":
            on = int(rng.uniform(0.5, 0.8) * n_days)
            n_ev = int(rng.integers(4, 14))
            ev = np.sort(rng.choice(np.arange(on, min(n_days - 40, on + 80)), n_ev, replace=False))
            cold[ev] = 1.0
            t_cell[ev] = np.minimum(t_cell[ev], rng.uniform(-5, 7, n_ev))
            total = 0.13 + 0.14 * sev
            add = np.zeros(n_days)
            for e in ev:
                dd = np.clip(d - e, 0, None)
                add += np.where(d > e, total / n_ev * (0.7 + 0.3 * np.exp(-dd / 20.0)), 0)
            rct = rct * (1 + add * r0 / rct)
            soh = soh - (0.025 + 0.035 * sev) * add / total
            ocv = ocv - (8 + 12 * sev) * add / total
        elif scen == "internal_damage":
            on = int(rng.uniform(0.55, 0.9) * n_days)
            pre = np.clip((d - (on - 30)) / 30, 0, 1) * 0.05
            step = np.where(d >= on, 0.15 + 0.2 * sev, 0)
            ro = ro + r0 * (pre + step) * 0.5
            rct = rct + r0 * (pre + step) * 0.5
            soh = soh - np.where(d >= on, 0.04 + 0.06 * sev, 0)
            ocv = ocv + np.where(d >= on, 15 + 20 * sev, 0)
        elif scen == "weld":
            on = int(rng.uniform(0.5, 0.8) * n_days)
            s1 = 0.10 + 0.05 * sev
            s2 = 0.10 + 0.10 * sev
            gap = int(rng.integers(30, 90))
            b_ind = r0 * (np.where(d >= on, s1, 0) + np.where(d >= on + gap, s2, 0))
        elif scen == "corrosion":
            on = int(rng.uniform(0.1, 0.35) * n_days)
            final = 0.18 + 0.22 * sev
            kk = rng.uniform(2, 5)
            x = np.clip((d - on) / (n_days - 1 - on), 0, 1)
            b_ind = r0 * final * (np.exp(kk * x) - 1) / (np.exp(kk) - 1)
        elif scen == "sei":
            ea_sei = rng.uniform(30e3, 50e3)
            teff = np.cumsum(1.0 / _arr(t_cell, ea_sei))  # Arrhenius-weighted time: faster when hot
            q = (teff / teff[-1]) ** 0.6
            grow = 0.18 + 0.22 * sev
            age = age + grow * q
            soh = soh - (0.05 + 0.07 * sev) * q ** 0.85
            ocv = ocv + (8 + 12 * sev) * q
        r_true = (ro * age * _arr(t_cell, ea_o) + rct * age * _arr(t_cell, ea_ct)) * soc_f + b_ind
        noise = _ar1(rng, n_days, 0.5, sig_n, df=4.0)
        r_meas = r_true * (1 + noise)
        t_meas = t_cell + drift_pack + rng.normal(0, 0.5)  # per-cell offset + shared drift
        t_pack = t_amb + drift_pack * 0.5 + rng.normal(0, 0.3, n_days)
        df = pd.DataFrame({
            "t": d, "cell_id": f"C{k:02d}", "pack_id": pack_id, "string_id": "S0", "resistance_ohm": r_meas,
            "temp_cell_c": t_meas, "temp_pack_c": t_pack, "soc": soc, "soh": soh + rng.normal(0, 0.003, n_days),
            "ocv_mv": ocv + rng.normal(0, 1.0, n_days), "current_a": current, "cold_charge": cold,
            "ah_throughput": np.cumsum(current) * 2, "cycle_count": d,
        })[keep]
        frames.append(df)
        truths.append(CellTruth(f"C{k:02d}", scen, EXPECTED[scen], float(r0 * age[0] * np.median(soc_f)),
                                float((1 - f_ct) * ea_o + f_ct * ea_ct), params))
    return pd.concat(frames, ignore_index=True), truths
