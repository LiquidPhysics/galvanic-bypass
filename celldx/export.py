"""Dashboard export: the compact ``celldx/v1`` JSON the Heat Atlas frontend reads.

Shape (one object per pack snapshot)::

    {"schema": "celldx/v1", "generated_at": ISO-8601, "pack_id": "S-12", "t": 68400,
     "probability_status": "uncalibrated_scores" | "calibrated",
     "faults": ["contact", "weld", "sei", "plating", "short", "cooling", "ageing"],
     "cells": [{"id": "C20", "p_healthy": 0.08, "p_fault": 0.84, "p_fault_ci": [0.80, 0.87],
                "verdict": "contact", "confidence": {"value": 0.9, "level": "High"},
                "headline": "...",
                "diagnoses": [{"fault": "contact", "p": 0.81, "ci": [0.70, 0.90],
                               "evidence": [{"signal": "R_mOhm", "value": 2.735, "llr": null,
                                             "weight": 0.99, "note": "R +21% vs baseline at 25 C"}],
                               "action": "clean and re-torque the contacts"}],
                "flags": {...}, "thermal": {...optional controller state...}}]}

Notes on honesty:

* ``p`` values are celldx's cause probabilities (normalised score products). They are
  ``uncalibrated_scores`` until a calibration is fitted; the field ``probability_status`` says
  which. ``ci`` is the bootstrap interval of the *fault* probability; per-cause intervals are
  not computed, so ``ci`` is only set on the top fault and on ``p_fault_ci``.
* celldx does not compute likelihood ratios, so ``llr`` is always ``null``; ``weight`` is the
  0-1 membership score of that evidence term in the scoring (see docs/ALGORITHM.md).
* ``lam`` (loss of active material) and ``thermistor`` as a *cause* are not modelled. A
  suspected sensor fault is reported in ``flags.sensor_fault_suspected`` and as a
  ``thermistor`` diagnosis with ``p: null``.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping, Sequence
from typing import Any

from . import CAUSES

DASHBOARD_SCHEMA = "celldx/v1"

#: celldx cause -> dashboard fault code
FAULT_CODE = {
    "mechanical_contact_corrosion": "contact",
    "mechanical_weld_or_tab": "weld",
    "sei_growth": "sei",
    "lithium_plating": "plating",
    "internal_damage": "short",
    "cooling_fault": "cooling",
    "ambiguous_ageing": "ageing",
    "temperature_effect": "cold",
    "healthy": "healthy",
}
FAULT_CODES = ["contact", "weld", "sei", "plating", "short", "cooling", "ageing"]
FAULT_LABEL = {
    "contact": "Contact corrosion (busbar / terminal)", "weld": "Loose weld or tab", "sei": "SEI growth",
    "plating": "Lithium plating", "short": "Internal damage (short)", "cooling": "Cooling fault (self-heating)",
    "ageing": "Ambiguous ageing", "cold": "Temperature effect (cold cell, not a fault)", "healthy": "Healthy",
    "thermistor": "Temperature sensor suspected",
}

# which sub-scores / features support each cause (signal name, feature key, unit transform, note template)
_EVIDENCE: dict[str, list[tuple[str, str | None, str]]] = {
    "mechanical_contact_corrosion": [("corrected_excess", "raw", "corrected R {v:+.0%} vs baseline at 25 C"),
                                     ("b_rel", "weak", "temperature-independent share b_rel {v:.3f} of R0"),
                                     ("growth_r2", "r2", "sqrt-time growth fit R2 {v:.2f}")],
    "mechanical_weld_or_tab": [("step_rel", "step", "resistance step {v:+.0%}"),
                               ("step_z", None, "step z = {v:.1f}"),
                               ("b_rel", "weak", "temperature-independent share b_rel {v:.3f} of R0")],
    "sei_growth": [("corrected_excess", "raw", "corrected R {v:+.0%} vs baseline at 25 C"),
                   ("soh_fall", "soh_fall", "SOH fell {v:.1%}"),
                   ("growth_r2", "r2", "sqrt-time growth fit R2 {v:.2f}")],
    "lithium_plating": [("post_cold_charge_rise", "plat", "R rise after cold charging {v:+.1%}"),
                        ("n_cold_charge", None, "{v:.0f} cold-charge events")],
    "internal_damage": [("step_rel", "step", "resistance step {v:+.0%}"),
                        ("soh_step", "jump", "SOH jump {v:+.3f}"),
                        ("ocv_step_mv", None, "OCV jump {v:+.1f} mV")],
    "cooling_fault": [("cell_minus_pack_k", "cool", "cell {v:+.1f} K above pack"),
                      ("unexplained_heat_k", None, "{v:+.1f} K not explained by I2R")],
    "ambiguous_ageing": [("corrected_excess", "raw", "corrected R {v:+.0%} vs baseline at 25 C"),
                         ("soh_recent", None, "SOH {v:.3f}")],
    "temperature_effect": [("raw_excess", None, "raw R {v:+.0%} at {t:.1f} C"),
                           ("corrected_excess", None, "corrected R {v:+.0%} (temperature explains the rise)")],
}


def _num(x: Any) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def _evidence(cell: Mapping[str, Any], cause: str) -> list[dict[str, Any]]:
    feats = cell.get("features", {}) or {}
    subs = cell.get("sub_scores", {}) or {}
    out: list[dict[str, Any]] = []
    for key, sub, tmpl in _EVIDENCE.get(cause, []):
        v = _num(feats.get(key))
        if v is None:
            continue
        note = tmpl.format(v=v, t=_num(feats.get("temp_recent_c")) or float("nan"))
        out.append({"signal": key, "value": round(v, 6), "llr": None,
                    "weight": (round(float(subs[sub]), 4) if sub and _num(subs.get(sub)) is not None else None),
                    "note": note})
    r0 = _num((cell.get("baseline_ref") or {}).get("r0_25c_ohm"))
    ce = _num(feats.get("corrected_excess"))
    if r0 is not None and ce is not None and cause not in ("healthy",):
        out.insert(0, {"signal": "R_mOhm", "value": round(r0 * (1 + ce) * 1000, 4), "llr": None, "weight": None,
                       "note": f"corrected R {r0 * (1 + ce) * 1000:.3f} mOhm vs baseline {r0 * 1000:.3f} mOhm at 25 C"})
    return out


def to_dashboard(report: Mapping[str, Any], *, pack_id: str | None = None, t: float | None = None,
                 plan: Mapping[str, Any] | None = None, id_map: Mapping[str, str] | None = None,
                 thermal: Mapping[str, Any] | None = None, top_k: int = 3) -> dict[str, Any]:
    """Convert a ``celldx-report/1`` (and optionally a ``celldx-plan/1``) to ``celldx/v1``.

    ``id_map`` renames cells (e.g. simulator ``C00`` -> layout ``C01``). ``thermal`` may hold a
    :meth:`celldx.thermal.controller.Decision.to_dict` whose per-cell state is attached.
    """
    recs: dict[str, Mapping[str, Any]] = {}
    if plan:
        for s in plan.get("strings", []):
            for r in s.get("recommendations", []):
                recs[str(r["cell_id"])] = r
    th_cells = {c["id"]: c for c in (thermal or {}).get("cells", [])}
    cells_out = []
    for c in report["cells"]:
        cid = str(c["cell_id"])
        oid = (id_map or {}).get(cid, cid)
        cp = c.get("cause_probabilities", {}) or {}
        ranked = sorted(((k, float(v)) for k, v in cp.items() if k not in ("healthy", "temperature_effect")),
                        key=lambda kv: -kv[1])
        rec = recs.get(cid)
        action_text = rec.get("text") if rec else None
        ci = c.get("fault_probability_interval")
        diags = []
        for i, (cause, p) in enumerate(ranked[:top_k]):
            code = FAULT_CODE[cause]
            diags.append({"fault": code, "cause": cause, "label": FAULT_LABEL[code], "p": round(p, 4),
                          "ci": ([round(float(x), 4) for x in ci] if (i == 0 and ci and cause == c.get("verdict")) else None),
                          "evidence": _evidence(c, cause),
                          "action": action_text if (i == 0 and cause == c.get("verdict")) else None})
        flags = dict(c.get("flags", {}) or {})
        if flags.get("sensor_fault_suspected"):
            diags.append({"fault": "thermistor", "cause": "sensor_fault_suspected", "label": FAULT_LABEL["thermistor"],
                          "p": None, "ci": None,
                          "evidence": [{"signal": "sensor_check", "value": None, "llr": None, "weight": None, "note": r}
                                       for r in flags.get("sensor_reasons", [])],
                          "action": "check the temperature sensor before acting on this cell"})
        conf = c.get("confidence", {}) or {}
        entry: dict[str, Any] = {
            "id": oid, "source_id": cid, "pack_id": c.get("pack_id"), "string_id": c.get("string_id"),
            "p_healthy": round(float(cp.get("healthy", 0.0)), 4),
            "p_temperature_effect": round(float(cp.get("temperature_effect", 0.0)), 4),
            "p_fault": _r(c.get("fault_probability")), "p_fault_ci": [_r(x) for x in ci] if ci else None,
            "verdict": FAULT_CODE.get(str(c.get("verdict")), str(c.get("verdict"))), "verdict_cause": c.get("verdict"),
            "is_fault": bool(c.get("is_fault")),
            "confidence": {"value": _r(conf.get("value")), "level": conf.get("level")},
            "headline": c.get("headline"), "diagnoses": diags, "flags": flags,
            "recommendation": rec,
        }
        if oid in th_cells or cid in th_cells:
            entry["thermal"] = th_cells.get(oid) or th_cells.get(cid)
        cells_out.append(entry)
    return {
        "schema": DASHBOARD_SCHEMA,
        "generated_at": report.get("generated_at") or _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "pack_id": pack_id or (cells_out[0]["pack_id"] if cells_out else None),
        "t": t,
        "celldx_version": report.get("celldx_version"),
        "config_hash": report.get("config_hash"),
        "probability_status": report.get("probability_status"),
        "faults": FAULT_CODES,
        "fault_labels": {k: FAULT_LABEL[k] for k in [*FAULT_CODES, "cold", "healthy", "thermistor"]},
        "n_cells": len(cells_out),
        "n_faults": int(sum(1 for c in cells_out if c["is_fault"])),
        "cells": cells_out,
        "notes": ["p values are celldx cause probabilities; see probability_status",
                  "llr is null: celldx scores are logistic memberships, not likelihood ratios; see weight",
                  "per-cause intervals are not computed; ci is the fault-probability bootstrap interval"],
    }


def _r(x: Any, nd: int = 4) -> float | None:
    v = _num(x)
    return round(v, nd) if v is not None else None


def layout_id_map(source_ids: Sequence[str], layout_ids: Sequence[str]) -> dict[str, str]:
    """Map sorted source ids onto layout ids one to one (e.g. C00..C31 -> C01..C32)."""
    src = sorted(source_ids)
    if len(src) > len(layout_ids):
        raise ValueError("more cells than layout slots")
    return dict(zip(src, layout_ids, strict=False))


__all__ = ["DASHBOARD_SCHEMA", "FAULT_CODE", "FAULT_CODES", "FAULT_LABEL", "layout_id_map", "to_dashboard", "CAUSES"]
