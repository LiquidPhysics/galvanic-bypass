"""Pulse-mode resistance extraction (100 Hz BMS data) and daily aggregation.

For a current step at ``t0`` (|dI| above threshold, current stable before and after):

* ``R_ohmic = |V(t0 + 10 ms) - V(t0 - 10 ms)| / |dI|``
* ``R_1s    = |V(t0 + 1 s)  - V(t0 - 10 ms)| / |dI|``
* ``R_slow  = R_1s - R_ohmic``

all in ohm. Voltages at the offsets are linearly interpolated. Pulses are then aggregated
per day with robust medians into the per-cell *sample table* the diagnosis works on.

Steady-current voltage noise: the signal is cut into windows of ``noise.window_s``; a
window counts when its current range is below ``noise.steady_max_di_a``; the voltage is
linearly detrended in each window and its standard deviation (mV) is taken. The daily value
is the median over windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config


def _segments(t: np.ndarray, gap_s: float) -> list[tuple[int, int]]:
    """Index ranges [a, b) of continuous segments (no gap longer than gap_s)."""
    if t.size == 0:
        return []
    cut = np.where(np.diff(t) > gap_s)[0] + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [t.size]])
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def extract_pulses(t: np.ndarray, v: np.ndarray, i: np.ndarray, cfg: Config, capacity_ah: float | None = None,
                   extra: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """Detect current steps in one cell's 100 Hz record and return one row per pulse.

    Columns: ``t0, di_a, r_ohmic, r_1s, r_slow`` plus, for each array in ``extra`` (e.g.
    temperature, SOC, BMS resistance), its value at the sample just before the step.
    """
    p = cfg.pulse
    di_thr = max(float(p.di_min_a), float(p.di_min_c_rate) * (capacity_ah or 0.0))
    rows: list[dict[str, float]] = []
    extra = extra or {}
    for a, b in _segments(t, float(p.segment_gap_s)):
        ts, vs, is_ = t[a:b], v[a:b], i[a:b]
        if ts.size < 10:
            continue
        dt = float(np.median(np.diff(ts)))
        if dt <= 0:
            continue
        di = np.diff(is_)
        cand = np.where(np.abs(di) >= di_thr)[0]
        if cand.size == 0:
            continue
        # merge adjacent candidates (a step spread over two samples)
        keep = [cand[0]]
        for c in cand[1:]:
            if c - keep[-1] > 2:
                keep.append(c)
        n_pre = max(2, int(round(float(p.pre_stable_s) / dt)))
        n_post = int(round(float(p.t_long_s) / dt)) + 1
        for k in keep:
            # step occurs between sample k and k+1; t0 is the midpoint
            if k - n_pre < 0 or k + 1 + n_post >= ts.size:
                continue
            pre = is_[k - n_pre + 1 : k + 1]
            post = is_[k + 1 : k + 1 + n_post]
            step = float(np.median(post) - np.median(pre))
            if abs(step) < di_thr:
                continue
            tol = float(p.stable_rel_di) * abs(step)
            if np.ptp(pre) > tol or np.ptp(post) > tol:
                continue
            t0 = 0.5 * (ts[k] + ts[k + 1])
            v_pre = float(np.interp(t0 - float(p.t_pre_s), ts, vs))
            v_ohm = float(np.interp(t0 + float(p.t_ohmic_s), ts, vs))
            v_long = float(np.interp(t0 + float(p.t_long_s), ts, vs))
            r_ohm = abs(v_ohm - v_pre) / abs(step)
            r_1s = abs(v_long - v_pre) / abs(step)
            row = {"t0": float(t0), "di_a": step, "r_ohmic": r_ohm, "r_1s": r_1s, "r_slow": r_1s - r_ohm}
            for name, arr in extra.items():
                row[name] = float(arr[a + k])
            rows.append(row)
    return pd.DataFrame(rows, columns=["t0", "di_a", "r_ohmic", "r_1s", "r_slow", *extra.keys()])


def steady_noise(t: np.ndarray, v: np.ndarray, i: np.ndarray, cfg: Config) -> pd.DataFrame:
    """Voltage noise (mV) in steady-current windows; one row per window: ``t, noise_mv``."""
    n = cfg.noise
    out: list[tuple[float, float]] = []
    for a, b in _segments(t, float(cfg.pulse.segment_gap_s)):
        ts, vs, is_ = t[a:b], v[a:b], i[a:b]
        if ts.size < 10:
            continue
        dt = float(np.median(np.diff(ts)))
        w = max(5, int(round(float(n.window_s) / dt)))
        nwin = ts.size // w
        if nwin == 0:
            continue
        T = ts[: nwin * w].reshape(nwin, w)
        V = vs[: nwin * w].reshape(nwin, w)
        I = is_[: nwin * w].reshape(nwin, w)
        steady = np.ptp(I, axis=1) <= float(n.steady_max_di_a)
        if not steady.any():
            continue
        T, V = T[steady], V[steady]
        x = T - T.mean(axis=1, keepdims=True)
        y = V - V.mean(axis=1, keepdims=True)
        slope = (x * y).sum(axis=1) / np.maximum((x * x).sum(axis=1), 1e-12)
        resid = y - slope[:, None] * x
        sd = resid.std(axis=1, ddof=1) * 1000.0
        out.extend(zip(T.mean(axis=1).tolist(), sd.tolist(), strict=True))
    return pd.DataFrame(out, columns=["t", "noise_mv"])


def pulse_cell_to_samples(g: pd.DataFrame, cfg: Config, capacity_ah: float | None) -> tuple[pd.DataFrame, dict[str, float]]:
    """Turn one cell's raw 100 Hz frame into the daily sample table plus extraction stats."""
    g = g[g["qc_ok"]] if "qc_ok" in g else g
    t = g["t"].to_numpy(float)
    v = g["voltage_v"].to_numpy(float)
    i = g["current_a"].to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(v) & np.isfinite(i)
    extra = {
        "temp_c": g["temp_cell_c"].to_numpy(float)[ok],
        "temp_pack_c": g["temp_pack_c"].to_numpy(float)[ok],
        "soc": g["soc"].to_numpy(float)[ok],
        "bms_r": g["resistance_ohm"].to_numpy(float)[ok],
    }
    pulses = extract_pulses(t[ok], v[ok], i[ok], cfg, capacity_ah, extra)
    noise = steady_noise(t[ok], v[ok], i[ok], cfg)
    day_s = float(cfg.pulse.day_s)
    stats = {"n_pulses": float(len(pulses)), "n_noise_windows": float(len(noise))}
    if pulses.empty:
        return pd.DataFrame(), stats
    pulses["day"] = np.floor(pulses["t0"] / day_s)
    agg = pulses.groupby("day").agg(
        t=("t0", "median"), r_ohmic=("r_ohmic", "median"), r_1s=("r_1s", "median"), r_slow=("r_slow", "median"),
        temp_c=("temp_c", "median"), temp_pack_c=("temp_pack_c", "median"), soc=("soc", "median"),
        bms_r=("bms_r", "median"), n_pulses=("r_1s", "size"),
    )
    agg["r_total"] = agg["r_1s"]
    # Per-sample daily context from the raw stream (sparse columns: SOH, OCV, throughput).
    gg = g.loc[ok].assign(day=np.floor(t[ok] / day_s))
    cap = capacity_ah or np.nan
    gg = gg.assign(c_rate=-gg["current_a"] / cap if capacity_ah else np.nan)
    ctx = gg.groupby("day").agg(
        soh=("soh", "median"), ocv_mv=("ocv_mv", "median"), ah_throughput=("ah_throughput", "max"),
        cycle_count=("cycle_count", "max"), current_rms=("current_a", lambda x: float(np.sqrt(np.nanmean(np.square(x))))),
        cold_charge=("cold_charge", "max"),
    )
    if not noise.empty:
        noise["day"] = np.floor(noise["t"] / day_s)
        ctx = ctx.join(noise.groupby("day")["noise_mv"].median(), how="left")
    else:
        ctx["noise_mv"] = np.nan
    samples = agg.join(ctx, how="left").reset_index(drop=True)
    stats["pulse_r_vs_bms_rel"] = float(np.nanmedian(np.abs(samples["r_total"] / samples["bms_r"] - 1))) \
        if samples["bms_r"].notna().any() else float("nan")
    return samples, stats
