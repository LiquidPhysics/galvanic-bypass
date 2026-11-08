"""Render 100 Hz pulse-mode BMS data from a daily sample frame (pulse_mode testing).

For each cell and day, ``pulses_per_day`` current steps are rendered, each as a 100 Hz burst:
``pre_s`` of rest, then a step of ``di_c_rate * capacity`` held for ``hold_s``. Voltage:

``V = OCV - I * R_ohmic - I * R_slow * (1 - exp(-(t - t0) / tau))  +  N(0, sigma_v)``

with ``R_ohmic = f_ohm * R_day`` and ``R_slow = (1 - f_ohm) * R_day / (1 - exp(-1 s / tau))``
so that the 1-second resistance equals the daily value ``R_day``. A steady-current window
of ``noise_s`` follows each day's first pulse for the voltage-noise metric.

Injectable pulse-only faults (on cell ``C00`` from ``onset_frac`` of the record):
``intermittent_contact`` (voltage-noise bursts x ``noise_gain`` and +``r_gain`` resistance),
``bms_r_bias`` (BMS-reported resistance biased by ``bias``), ``temp_sensor_offset`` (cell
thermistor reads ``offset_k`` low).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def render_pulses(daily: pd.DataFrame, *, capacity_ah: float = 2.5, pulses_per_day: int = 2, hz: float = 100.0,
                  pre_s: float = 0.2, hold_s: float = 1.2, noise_s: float = 1.0, f_ohm: float = 0.6, tau_s: float = 0.25,
                  di_c_rate: float = 1.0, sigma_v: float = 2e-4, seed: int = 0, fault: str | None = None,
                  onset_frac: float = 0.7, noise_gain: float = 8.0, r_gain: float = 0.05, bias: float = 0.4,
                  offset_k: float = 8.0, bms_bias_all: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt = 1.0 / hz
    n_pre, n_hold, n_noise = int(pre_s * hz), int(hold_s * hz), int(noise_s * hz)
    tl = np.arange(n_pre + n_hold) * dt
    t0 = n_pre * dt - dt / 2
    shape = np.where(tl > t0, 1.0, 0.0)
    rc = np.where(tl > t0, 1.0 - np.exp(-(tl - t0) / tau_s), 0.0)
    k1 = 1.0 - np.exp(-1.0 / tau_s)
    out = []
    days = np.sort(daily["t"].unique())
    onset_day = days[int(onset_frac * (len(days) - 1))] if len(days) else 0
    for cid, g in daily.groupby("cell_id", sort=True):
        g = g.sort_values("t")
        for _, row in g.iterrows():
            day = float(row["t"])
            r_day = float(row["resistance_ohm"])
            if not np.isfinite(r_day):
                continue
            target = cid == "C00" and fault is not None and day >= onset_day
            if target and fault == "intermittent_contact":
                r_day *= 1 + r_gain
            r_o = f_ohm * r_day
            r_s = (1 - f_ohm) * r_day / k1
            di = di_c_rate * capacity_ah * (1 if rng.random() < 0.5 else -1)
            ocv = 3.7 + 0.001 * float(row.get("ocv_mv", 0.0) or 0.0)
            temp = float(row["temp_cell_c"])
            if target and fault == "temp_sensor_offset":
                temp -= offset_k
            bms = r_day * (1 + bms_bias_all) * (1 + (bias if (target and fault == "bms_r_bias") else 0.0))
            for p in range(pulses_per_day):
                base_t = day * 86400.0 + 3600.0 * (6 + 6 * p) + rng.uniform(0, 600)
                i = di * shape
                sv = sigma_v * (noise_gain if (target and fault == "intermittent_contact") else 1.0)
                v = ocv - i * r_o - i * r_s * rc + rng.normal(0, sigma_v, tl.size)
                if target and fault == "intermittent_contact":
                    spikes = rng.random(tl.size) < 0.05
                    v = v + spikes * rng.normal(0, sv, tl.size)
                seg = {"t": base_t + tl, "voltage_v": v, "current_a": i}
                if p == 0:
                    tn = base_t + tl[-1] + dt * (1 + np.arange(n_noise))
                    i_n = np.full(n_noise, di)
                    v_n = ocv - di * (r_o + r_s * (1 - np.exp(-(tn - base_t - t0) / tau_s))) + rng.normal(0, sv, n_noise)
                    seg = {"t": np.concatenate([seg["t"], tn]), "voltage_v": np.concatenate([seg["voltage_v"], v_n]),
                           "current_a": np.concatenate([seg["current_a"], i_n])}
                m = seg["t"].size
                frame = pd.DataFrame(seg)
                frame["cell_id"] = cid
                frame["pack_id"] = row.get("pack_id", "P0")
                frame["string_id"] = row.get("string_id", "S0")
                frame["temp_cell_c"] = temp
                frame["temp_pack_c"] = row.get("temp_pack_c", np.nan)
                frame["soc"] = row.get("soc", np.nan)
                frame["resistance_ohm"] = bms
                frame["soh"] = np.nan
                frame["ocv_mv"] = np.nan
                frame["cold_charge"] = 0.0
                if p == 0:  # one sparse row per day carries SOH / OCV / cold-charge
                    frame.loc[0, "soh"] = row.get("soh", np.nan)
                    frame.loc[0, "ocv_mv"] = row.get("ocv_mv", np.nan)
                    frame.loc[0, "cold_charge"] = row.get("cold_charge", 0.0)
                frame["ah_throughput"] = row.get("ah_throughput", np.nan)
                frame["cycle_count"] = row.get("cycle_count", np.nan)
                assert len(frame) == m
                out.append(frame)
    return pd.concat(out, ignore_index=True)
