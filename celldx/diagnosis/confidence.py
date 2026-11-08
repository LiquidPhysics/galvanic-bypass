"""Confidence (section 6.5) with every penalty reported.

``confidence = data_quality * baseline_trust * (margin_base + margin_gain * margin)``, with
``margin = min(1, (p_top - p_second) / margin_scale)`` and
``data_quality = min(1, n / n_full) * (1 if temperature_spread >= spread_full_k else spread_short_factor)``.
Then each penalty multiplies in, the SEI/corrosion cap applies, and the value is clipped to
0..1. Levels: High >= ``high``, Medium >= ``medium``, otherwise Low.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..config import Config


def level(c: float, cfg: Config) -> str:
    if c >= float(cfg.confidence.high):
        return "High"
    if c >= float(cfg.confidence.medium):
        return "Medium"
    return "Low"


def compute_confidence(
    probs: Mapping[str, float],
    feats: Mapping[str, float],
    cfg: Config,
    *,
    baseline_trust: float,
    trust_reasons: list[str],
    mode: str,
    has_pack_temp: bool,
    n_neighbours: int,
    sensor_fault: bool,
    chemistry_mismatch: bool,
    sub: Mapping[str, float],
    verdict: str,
    fault_interval: tuple[float, float] | None,
    calibrator: Any | None = None,
) -> dict[str, Any]:
    cc = cfg.confidence
    fx = cfg.fixes
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    (top, p1), (second, p2) = ranked[0], ranked[1]
    margin = min(1.0, (p1 - p2) / float(cc.margin_scale))
    n = float(feats.get("n_samples", 0.0) or 0.0)
    spread = float(feats.get("temp_spread_k", 0.0) or 0.0)
    dq = min(1.0, n / float(cc.n_full)) * (1.0 if spread >= float(cc.spread_full_k) else float(cc.spread_short_factor))
    base = dq * baseline_trust * (float(cc.margin_base) + float(cc.margin_gain) * margin)
    penalties: list[dict[str, Any]] = []

    def pen(name: str, factor: float, reason: str) -> None:
        penalties.append({"name": name, "factor": float(factor), "reason": reason})

    if not has_pack_temp:
        pen("no_pack_temperature", cc.penalty_no_pack_temp, "no pack temperature: cooling faults and sensor checks are blind")
    if n_neighbours < int(cfg.features.neighbour_min_cells) - 1:
        pen("no_neighbours", cc.penalty_no_neighbours, "fewer than "
            f"{int(cfg.features.neighbour_min_cells) - 1} neighbours: no pack comparison")
    if mode == "cycle_mode":
        pen("cycle_mode", cc.penalty_cycle_mode, "cycle_mode data: no pulse split or voltage noise")
    if sensor_fault:
        pen("sensor_fault_suspected", cc.penalty_sensor_fault, "a sensor disagrees with its cross-check")
    if chemistry_mismatch:
        pen("baseline_chemistry_mismatch", cc.penalty_chemistry_mismatch, "baseline from a different chemistry or format")
    if float(cc.penalty_tind_unavailable) != 1.0 and not np.isfinite(feats.get("b_rel", np.nan)):
        pen("tind_unavailable", cc.penalty_tind_unavailable, "temperature-independent share unavailable")
    if bool(fx.tind_unavailable_penalty) and not np.isfinite(feats.get("b_rel", np.nan)):
        pen("tind_unavailable", fx.tind_unavailable_factor,
            "temperature-independent share unavailable (short or temperature-flat history): contact faults "
            "and SEI cannot be separated by temperature behaviour")
    if bool(fx.conflict_penalty) and verdict in ("temperature_effect", "healthy"):
        conflict = []
        if sub.get("step", 0.0) >= float(fx.conflict_step_score):
            conflict.append(f"a resistance step was detected (step score {sub.get('step', 0.0):.2f})")
        if sub.get("b_rel_available", 0.0) > 0.5 and sub.get("weak", 0.0) >= float(fx.conflict_weak):
            conflict.append(f"part of the resistance does not follow temperature (weak score {sub.get('weak', 0.0):.2f})")
        if conflict:
            pen("verdict_conflicts_with_evidence", fx.conflict_penalty_factor,
                "non-fault verdict conflicts with fault evidence: " + "; ".join(conflict))
    if bool(fx.bootstrap_width_penalty) and fault_interval is not None:
        width = max(0.0, fault_interval[1] - fault_interval[0])
        if width > 0:
            pen("wide_fault_interval", max(0.0, 1.0 - width * float(fx.bootstrap_width_scale)),
                f"bootstrap fault-probability interval is {width:.2f} wide")
    value = base
    for p in penalties:
        value *= p["factor"]
    capped = False
    if {top, second} == {"sei_growth", "mechanical_contact_corrosion"} and (p1 - p2) < float(cc.sei_corrosion_gap):
        if value > float(cc.sei_corrosion_cap):
            value = float(cc.sei_corrosion_cap)
            capped = True
    value = float(np.clip(value, 0.0, 1.0))
    out: dict[str, Any] = {
        "value": value,
        "level": level(value, cfg),
        "status": "uncalibrated",
        "components": {"data_quality": dq, "baseline_trust": baseline_trust, "margin": margin,
                       "n_samples": n, "temperature_spread_k": spread, "base": base,
                       "top_two": [top, second], "p_top": p1, "p_second": p2},
        "penalties": penalties,
        "sei_corrosion_cap_applied": capped,
        "trust_reasons": trust_reasons,
    }
    if calibrator is not None and bool(fx.confidence_calibration):
        cal = float(calibrator(value))
        out["value_uncalibrated"] = value
        out["value"] = cal
        out["level"] = level(cal, cfg)
        out["status"] = "calibrated"
    return out
