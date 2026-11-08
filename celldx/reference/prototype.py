"""Reference reconstruction of the prototype ``celldx.py`` (single cell, daily samples).

The original prototype file was not supplied with the brief. This file re-implements the
behaviour the final brief specifies (sections 6.3 to 6.5), deliberately written as plain,
loop-based Python with no imports from the package, so that the package can be regression
tested against an independent implementation. When the real ``celldx.py`` is available,
``tests/test_regression_prototype.py`` should be pointed at it as well.

Inputs are plain arrays of one cell's samples in time order:
``r`` (ohm), ``temp`` (C), ``soh``, ``ocv`` (mV deviation), ``temp_pack`` (C),
``cold`` (0/1) and the baseline ``r0`` (ohm at 25 C) and ``ea`` (J/mol).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

RG, T25 = 8.314, 298.15
CAUSES = ["healthy", "temperature_effect", "cooling_fault", "lithium_plating", "internal_damage",
          "mechanical_weld_or_tab", "mechanical_contact_corrosion", "sei_growth", "ambiguous_ageing"]


def sig(x: float, c: float, w: float) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    z = max(-500.0, min(500.0, -(x - c) / w))
    return 1.0 / (1.0 + math.exp(z))


def arr(t_c: float, ea: float) -> float:
    return math.exp(ea / RG * (1.0 / (t_c + 273.15) - 1.0 / T25))


def nanmedian(xs: list[float]) -> float:
    v = [x for x in xs if x is not None and not math.isnan(x)]
    return float(np.median(v)) if v else float("nan")


def win(n: int, frac: float, nmin: int) -> int:
    return min(n, max(nmin, int(round(frac * n))))


def step(x: list[float], w: int) -> tuple[float, float, float]:
    """(jump, rel, z) of the largest window-median step; trend-proof noise from first differences."""
    x = [v for v in x if not math.isnan(v)]
    n = len(x)
    if n < 2 * w:
        return float("nan"), float("nan"), float("nan")
    d = [x[i + 1] - x[i] for i in range(n - 1)]
    md = float(np.median(d))
    noise = 1.4826 * float(np.median([abs(v - md) for v in d])) / math.sqrt(2)
    noise = max(noise, 1e-4 * float(np.median([abs(v) for v in x])), 1e-15)
    best, bk = 0.0, -1
    for k in range(w, n - w + 1):
        j = float(np.median(x[k:k + w])) - float(np.median(x[k - w:k]))
        if bk < 0 or abs(j) > abs(best):
            best, bk = j, k
    before = float(np.median(x[bk - w:bk]))
    return best, best / before, abs(best) / (1.25 * noise * math.sqrt(2.0 / w))


def diagnose_cell(r: list[float], temp: list[float], soh: list[float], ocv: list[float], temp_pack: list[float],
                  cold: list[float], r0: float, ea: float, t: list[float] | None = None,
                  baseline_trust: float = 1.0, n_neighbours: int = 0, cycle_mode: bool = True) -> dict[str, Any]:
    n = len(r)
    t = list(range(n)) if t is None else t
    r25 = [r[i] / arr(temp[i], ea) for i in range(n)]
    wr, wc = win(n, 0.03, 5), win(n, 0.07, 20)
    raw_excess = nanmedian(r[-wr:]) / r0 - 1
    corrected_excess = nanmedian(r25[-wc:]) / r0 - 1
    ws = win(n, 0.033, 5)
    _, step_rel, step_z = step(r25, ws)
    soh_step = step(soh, ws)[0] if any(not math.isnan(v) for v in soh) else float("nan")
    ocv_step = step(ocv, ws)[0] if any(not math.isnan(v) for v in ocv) else float("nan")
    # growth fit R25 = a + b sqrt(t)
    x = np.sqrt(np.array(t, float) - t[0])
    A = np.column_stack([np.ones(n), x])
    coef = np.linalg.lstsq(A, np.array(r25), rcond=None)[0]
    pred = A @ coef
    y = np.array(r25)
    r2 = 1 - float(((y - pred) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
    # temperature-independent share over the last 365 samples
    tt, rr, TT = t[-365:], r25[-365:], temp[-365:]
    lo, hi = np.quantile(TT, [0.05, 0.95])
    b_rel = float("nan")
    if len(tt) >= 60 and hi - lo >= 8.0:
        inv = np.array([1.0 / arr(v, ea) for v in TT])
        tc = np.array(tt, float) - np.mean(tt)
        c = np.linalg.lstsq(np.column_stack([np.ones(len(tt)), tc, inv]), np.array(rr), rcond=None)[0]
        b_rel = float(c[2]) * float(np.median(inv)) / r0
    has_ocv = any(not math.isnan(v) for v in ocv)
    ocv_recent = nanmedian(ocv[-wc:])
    soh_recent = nanmedian(soh[-wc:])
    cmp_ = nanmedian([temp[i] - temp_pack[i] for i in range(n - wc, n)])
    # plating: median R25 after the first cold charge vs the 30 samples before
    ev = [i for i in range(n) if cold[i] > 0.5]
    rise = 0.0
    if ev:
        e0 = ev[0]
        before = [v for v in r25[max(0, e0 - 30):e0] if not math.isnan(v)]
        after = [v for v in r25[e0 + 1:] if not math.isnan(v)]
        rise = float(np.median(after) / np.median(before) - 1) if len(before) >= 3 and after else float("nan")
    fault = sig(corrected_excess, 0.15, 0.04)
    rawsig = sig(raw_excess, 0.15, 0.04)
    st = sig(step_rel, 0.12, 0.03) * sig(step_z, 5, 1.5)
    jump = max(sig(-soh_step, 0.04, 0.015), sig(abs(ocv_step) if not math.isnan(ocv_step) else float("nan"), 15, 5))
    ocv_drift = sig(abs(ocv_recent) if not math.isnan(ocv_recent) else float("nan"), 8, 3)
    soh_fall = sig(1 - soh_recent if not math.isnan(soh_recent) else float("nan"), 0.03, 0.012)
    lith = ocv_drift * (0.6 + 0.4 * soh_fall) if has_ocv else soh_fall
    weak = sig(b_rel, 0.08, 0.03) if not math.isnan(b_rel) else 0.3
    cool = sig(cmp_, 6, 1.5)
    plat = sig(rise, 0.10, 0.03)
    s = {
        "healthy": (1 - rawsig) * (1 - fault) * (1 - cool) * (1 - plat),
        "temperature_effect": rawsig * (1 - fault) * (1 - cool),
        "cooling_fault": cool,
        "lithium_plating": plat,
        "internal_damage": fault * st * jump,
        "mechanical_weld_or_tab": fault * st * (1 - jump),
        "mechanical_contact_corrosion": fault * (1 - st) * weak * (1 - ocv_drift) * (1 - sig(r2, 0.9, 0.04) * lith),
        "sei_growth": fault * (1 - st) * sig(r2, 0.8, 0.07) * lith * (1 - 0.5 * weak),
        "ambiguous_ageing": 0.06,
    }
    tot = sum(s.values())
    p = {k: v / tot for k, v in s.items()}
    verdict = max(CAUSES, key=lambda k: p[k])
    ranked = sorted(p.values(), reverse=True)
    margin = min(1.0, (ranked[0] - ranked[1]) / 0.5)
    spread = float(hi - lo)
    dq = min(1.0, n / 150) * (1.0 if spread >= 8 else 0.85)
    conf = dq * baseline_trust * (0.35 + 0.65 * margin)
    conf_precap = conf
    top2 = {sorted(p, key=lambda k: -p[k])[0], sorted(p, key=lambda k: -p[k])[1]}
    if top2 == {"sei_growth", "mechanical_contact_corrosion"} and ranked[0] - ranked[1] < 0.3:
        conf = min(conf, 0.55)
    return {"verdict": verdict, "probabilities": p, "fault_probability": 1 - p["healthy"] - p["temperature_effect"],
            "confidence_base": conf_precap, "confidence_capped": max(0.0, min(1.0, conf)),
            "features": {"raw_excess": raw_excess, "corrected_excess": corrected_excess, "step_rel": step_rel,
                         "step_z": step_z, "soh_step": soh_step, "ocv_step_mv": ocv_step, "growth_r2": r2,
                         "b_rel": b_rel, "post_cold_charge_rise": rise, "cell_minus_pack_k": cmp_}}
