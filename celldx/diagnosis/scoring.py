"""Cause scores, probabilities and the deterministic decision tree (section 6.4).

All functions accept scalars or numpy arrays (bootstrap replicates) and broadcast.
Every centre and width comes from ``cfg.scoring`` (and ``cfg.fixes`` for the v3 fixes).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .. import CAUSES
from ..config import Config
from ..physics import sig


def _s(x: Any, p: Any) -> np.ndarray:
    """Logistic of a feature that may be NaN (NaN -> 0, i.e. no evidence)."""
    return np.nan_to_num(sig(x, float(p.c), float(p.w)), nan=0.0)


def sub_scores(f: Mapping[str, Any], cfg: Config, has_ocv: bool) -> dict[str, np.ndarray]:
    """Intermediate scores: fault, raw, step, jump, ocv_drift, soh_fall, lith, weak, cool, plat, noise.

    * fault = sig(corrected_excess, 0.15, 0.04)
    * step  = sig(step_rel, 0.12, 0.03) * sig(step_z, 5, 1.5)
    * jump  = max(sig(-soh_step, 0.04, 0.015), sig(|ocv_step_mv|, 15, 5))
    * ocv_drift = sig(|ocv_recent_mv|, 8, 3); soh_fall = sig(1 - soh_recent, 0.03, 0.012)
    * lith = ocv_drift * (0.6 + 0.4 soh_fall) if OCV data exists else soh_fall
    * weak = sig(b_rel, 0.08, 0.03), or 0.3 if b_rel unavailable
    * cool = sig(cell_minus_pack, 6, 1.5); plat = sig(post_cold_charge_rise, 0.10, 0.03)
    """
    sc = cfg.scoring
    fx = cfg.fixes
    g = lambda k: np.asarray(f.get(k, np.nan), dtype=float)  # noqa: E731
    fault = _s(g("corrected_excess"), sc.fault)
    raw = _s(g("raw_excess"), sc.raw)
    step = _s(g("step_rel"), sc.step_rel) * _s(g("step_z"), sc.step_z)
    if bool(fx.temp_step_guard):
        step = step * (1.0 - _s(np.abs(g("temp_step_at_r_step_k")), fx.temp_step_guard_sig))
    jump = np.maximum(_s(-g("soh_step"), sc.soh_jump), _s(np.abs(g("ocv_step_mv")), sc.ocv_jump))
    ocv_drift = _s(np.abs(g("ocv_recent_mv")), sc.ocv_drift)
    soh_fall = _s(1.0 - g("soh_recent"), sc.soh_fall)
    if has_ocv:
        lith = ocv_drift * (float(sc.lith_ocv_base) + float(sc.lith_ocv_soh_gain) * soh_fall)
    else:
        lith = soh_fall
    b_rel = g("b_rel")
    weak = np.where(np.isfinite(b_rel), _s(b_rel, sc.weak), float(sc.weak_unavailable))
    cool_x = g("cell_minus_pack_k") if str(sc.cooling_signal) == "raw" else g("unexplained_heat_k")
    cool = _s(cool_x, sc.cool)
    plat = _s(g("post_cold_charge_rise"), sc.plat)
    noise = _s(g("noise_ratio"), sc.noise_boost)
    fault_raw = fault
    if bool(fx.fault_floor):
        floor_step = step * float(fx.fault_floor_step_gain)
        fb = fx.fault_floor_brel
        floor_b = np.where(np.isfinite(b_rel), _s(b_rel, fb) * float(fb.gain), 0.0)
        fault = np.maximum(fault, np.maximum(floor_step, floor_b))
    return {"fault": fault, "fault_prototype": fault_raw, "raw": raw, "step": step, "jump": jump,
            "ocv_drift": ocv_drift, "soh_fall": soh_fall, "lith": lith, "weak": weak, "cool": cool,
            "plat": plat, "noise": noise, "r2": np.nan_to_num(g("growth_r2"), nan=0.0),
            "b_rel_available": np.isfinite(b_rel).astype(float)}


def cause_scores(s: Mapping[str, np.ndarray], cfg: Config, has_ocv: bool) -> dict[str, np.ndarray]:
    """Unnormalised cause scores from sub-scores (prototype formulas)."""
    sc = cfg.scoring
    fault, raw, step, jump = s["fault"], s["raw"], s["step"], s["jump"]
    weak, cool, plat, lith, ocv_drift, r2 = s["weak"], s["cool"], s["plat"], s["lith"], s["ocv_drift"], s["r2"]
    noise = s["noise"]
    healthy = (1 - raw) * (1 - fault) * (1 - cool) * (1 - plat) * (1 - noise)
    temperature = raw * (1 - fault) * (1 - cool) * (1 - noise)
    weld = fault * step * (1 - jump)
    weld = weld + (1 - weld) * float(sc.noise_boost.gain) * noise * (1 - jump)
    corrosion = fault * (1 - step) * weak * (1 - ocv_drift) * (1 - _s(r2, sc.r2_corrosion) * lith)
    sei = fault * (1 - step) * _s(r2, sc.r2_sei) * lith * (1 - float(sc.sei_weak_damping) * weak)
    amb = np.full_like(np.asarray(fault, float), float(sc.ambiguous_floor))
    if bool(cfg.fixes.flat_history_corrosion_guard):
        unavailable = s["b_rel_available"] < 0.5
        if not has_ocv:
            amb = np.where(unavailable, np.maximum(amb, 0.5 * np.maximum(corrosion, sei)), amb)
    out = {
        "healthy": healthy,
        "temperature_effect": temperature,
        "cooling_fault": cool * np.ones_like(fault),
        "lithium_plating": plat * np.ones_like(fault),
        "internal_damage": fault * step * jump,
        "mechanical_weld_or_tab": weld,
        "mechanical_contact_corrosion": corrosion,
        "sei_growth": sei,
        "ambiguous_ageing": amb,
    }
    return {k: np.asarray(out[k], float) for k in CAUSES}


def probabilities(scores: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Normalise scores to probabilities summing to one."""
    total = sum(np.asarray(scores[k], float) for k in CAUSES)
    return {k: np.asarray(scores[k], float) / total for k in CAUSES}


def fault_probability(p: Mapping[str, np.ndarray]) -> np.ndarray:
    """P(genuine cell fault) = 1 - P(healthy) - P(temperature_effect)."""
    return 1.0 - p["healthy"] - p["temperature_effect"]


def decision_tree(f: Mapping[str, float], cfg: Config, has_ocv: bool) -> tuple[str, list[dict[str, Any]]]:
    """Deterministic fallback, in the prototype's order, with each test at the sigmoid centre.

    Returns the verdict and the path of tests (name, value, threshold, passed).
    """
    sc = cfg.scoring
    path: list[dict[str, Any]] = []

    def val(k: str) -> float:
        x = f.get(k, float("nan"))
        return float(x) if x is not None else float("nan")

    def test(name: str, value: float, thr: float, ge: bool = True) -> bool:
        ok = bool(np.isfinite(value) and (value >= thr if ge else value < thr))
        path.append({"test": name, "value": None if not np.isfinite(value) else round(value, 6), "threshold": thr,
                     "passed": ok})
        return ok

    corr, raw = val("corrected_excess"), val("raw_excess")
    cmp_ = val("cell_minus_pack_k")
    rise = val("post_cold_charge_rise")
    cool_hit = np.isfinite(cmp_) and cmp_ >= float(sc.cool.c)
    plat_hit = np.isfinite(rise) and rise >= float(sc.plat.c)
    fault_hit = np.isfinite(corr) and corr >= float(sc.fault.c)
    if not fault_hit and not cool_hit and not plat_hit:
        if test("raw_excess below threshold (healthy)", raw, float(sc.raw.c), ge=False) or not np.isfinite(raw):
            return "healthy", path
        test("corrected_excess below threshold (temperature explains the rise)", corr, float(sc.fault.c), ge=False)
        return "temperature_effect", path
    if test("cell_minus_pack_k (cooling)", cmp_, float(sc.cool.c)):
        return "cooling_fault", path
    if test("post_cold_charge_rise (plating)", rise, float(sc.plat.c)):
        return "lithium_plating", path
    step_hit = test("step_rel", val("step_rel"), float(sc.step_rel.c)) and test("step_z", val("step_z"), float(sc.step_z.c))
    if step_hit:
        soh_hit = test("-soh_step (capacity jump)", -val("soh_step"), float(sc.soh_jump.c))
        ocv_hit = test("|ocv_step_mv| (OCV jump)", abs(val("ocv_step_mv")), float(sc.ocv_jump.c))
        return ("internal_damage" if (soh_hit or ocv_hit) else "mechanical_weld_or_tab"), path
    b_rel = val("b_rel")
    drift_hit = abs(val("ocv_recent_mv")) >= float(sc.ocv_drift.c) if has_ocv else (1 - val("soh_recent")) >= float(sc.soh_fall.c)
    path.append({"test": "lithium-loss signal (OCV drift, or SOH fall without OCV)", "value": None,
                 "threshold": None, "passed": bool(drift_hit)})
    if np.isfinite(b_rel) and test("b_rel (temperature-independent share)", b_rel, float(sc.weak.c)) and not drift_hit:
        return "mechanical_contact_corrosion", path
    if test("growth_r2 (sqrt-t growth)", val("growth_r2"), float(sc.r2_sei.c)) and drift_hit:
        return "sei_growth", path
    return "ambiguous_ageing", path
