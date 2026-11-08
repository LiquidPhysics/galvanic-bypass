"""Plain-language explanations that cite the numbers behind each verdict.

The report always leads with the fault probability, so a healthy cell that is merely cold
reads as "no cell fault" even though its label is "temperature effect".
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .. import CAUSE_LABELS


def _pct(x: float | None, signed: bool = True) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x:+.0%}" if signed else f"{x:.0%}"


def _num(x: float | None, fmt: str = ".2f") -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return format(x, fmt)


def headline(verdict: str, fault_p: float, interval: tuple[float, float] | None, status: str, conf: Mapping[str, Any]) -> str:
    iv = f" (interval {interval[0]:.0%} to {interval[1]:.0%})" if interval else ""
    tag = "uncalibrated score" if status == "uncalibrated_scores" else "calibrated"
    if verdict in ("healthy", "temperature_effect") and fault_p < 0.5:
        lead = f"No cell fault: fault probability {fault_p:.0%}{iv}, {tag}."
        if verdict == "temperature_effect":
            lead += " Resistance is high only because the cell is cold."
    elif verdict in ("healthy", "temperature_effect"):
        lead = (f"Fault probability {fault_p:.0%}{iv}, {tag}, but spread over several causes: no single fault cause is "
                f"more likely than {CAUSE_LABELS[verdict]}. Treat as unresolved; collect more data.")
    else:
        lead = f"Fault probability {fault_p:.0%}{iv}, {tag}. Most likely cause: {CAUSE_LABELS[verdict]}."
    return f"{lead} Confidence {conf['level']} ({conf['value']:.2f})."


def explain(verdict: str, probs: Mapping[str, float], f: Mapping[str, float], flags: Mapping[str, Any],
            conf: Mapping[str, Any], r0_ohm: float, availability: Mapping[str, str], neighbour: Mapping[str, Any]) -> list[str]:
    """Evidence sentences, most decisive first."""
    s: list[str] = []
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    s.append("Cause probabilities: " + ", ".join(f"{CAUSE_LABELS[k]} {v:.2f}" for k, v in ranked[:3]) + ".")
    s.append(
        f"Corrected resistance (25 C, SOC-referred) is {_pct(f.get('corrected_excess'))} versus baseline R0 "
        f"{r0_ohm * 1000:.3g} mOhm; raw resistance is {_pct(f.get('raw_excess'))} at a recent median "
        f"{_num(f.get('temp_recent_c'), '.1f')} C. The fault band starts at +15% corrected."
    )
    if verdict == "temperature_effect":
        s.append("The rise disappears after Arrhenius correction, so the cell is cold, not damaged. "
                 "Fix the heating or enclosure and derate while cold.")
    step_rel, step_z = f.get("step_rel"), f.get("step_z")
    if step_rel is not None and math.isfinite(step_rel):
        s.append(f"Largest resistance step: {_pct(step_rel)} with z = {_num(step_z, '.1f')} (a step needs about +12% and z above 5).")
    if verdict in ("internal_damage", "mechanical_weld_or_tab"):
        s.append(f"At the same time SOH changed by {_num((f.get('soh_step') or float('nan')) * 100, '+.1f')} points and OCV by "
                 f"{_num(f.get('ocv_step_mv'), '+.1f')} mV; "
                 + ("a capacity or OCV jump means the electrochemistry was damaged." if verdict == "internal_damage"
                    else "no capacity or OCV jump, so the added resistance is outside the electrochemistry (weld, tab, busbar)."))
    b_rel = f.get("b_rel")
    if b_rel is not None and math.isfinite(b_rel):
        s.append(f"Temperature-independent share b_rel = {_num(b_rel, '.3f')} of R0 (contact faults sit above about 0.08; "
                 f"apparent Ea ratio {_num(f.get('ea_ratio'), '.2f')}).")
    elif "b_rel" in availability:
        s.append("Temperature-independent share unavailable: " + availability["b_rel"] + ".")
    if verdict in ("sei_growth", "mechanical_contact_corrosion", "ambiguous_ageing"):
        s.append(f"Growth fit R25 = a + b sqrt(t): R^2 = {_num(f.get('growth_r2'), '.2f')}. Median OCV deviation "
                 f"{_num(f.get('ocv_recent_mv'), '+.1f')} mV, SOH {_num(f.get('soh_recent'), '.3f')}.")
        if verdict == "sei_growth":
            s.append("Gradual sqrt-time growth that follows temperature, with lithium-inventory loss (OCV drift or SOH fall), is the SEI signature.")
        if verdict == "mechanical_contact_corrosion":
            s.append("Gradual growth that does not follow temperature and without OCV drift points to contact corrosion.")
    cmp_ = f.get("cell_minus_pack_k")
    if cmp_ is not None and math.isfinite(cmp_) and (verdict == "cooling_fault" or abs(cmp_) > 3):
        s.append(f"Cell runs {_num(cmp_, '+.1f')} K relative to the pack (expected I^2R heating {_num(f.get('dT_expected_k'), '.1f')} K); "
                 "a cooling fault starts at about +6 K.")
    rise = f.get("post_cold_charge_rise")
    if verdict == "lithium_plating" or (rise is not None and math.isfinite(rise) and rise > 0.05):
        s.append(f"Resistance rose {_pct(rise)} after {int(f.get('n_cold_charge') or 0)} cold hard-charge samples (plating starts at about +10%).")
    nr = f.get("noise_ratio")
    if nr is not None and math.isfinite(nr) and nr > 2:
        s.append(f"Steady-current voltage noise is {_num(nr, '.1f')} x the reference: intermittent contact.")
    if neighbour.get("available"):
        if neighbour.get("diverges_from_neighbours"):
            s.append(f"Diverges from its {neighbour['n_neighbours']} neighbours (robust z = {_num(neighbour.get('z'), '+.1f')}).")
        if neighbour.get("pack_wide_drift"):
            s.append("Most cells in this pack have drifted together: check pack-level causes (temperature, ageing, sensor).")
    for r in flags.get("sensor_reasons", []) or []:
        s.append("Sensor check: " + r + ".")
    if conf.get("sei_corrosion_cap_applied"):
        s.append("SEI and contact corrosion are close here and hard to separate, so confidence is capped.")
    for p in conf.get("penalties", []):
        s.append(f"Confidence x{p['factor']:.2f}: {p['reason']}.")
    return s


def text_report(report: Mapping[str, Any]) -> str:
    """Readable plain-text (Markdown) report of a ``celldx-report/1`` dict: every cell leads with
    its fault probability, then cause probabilities, confidence and its reasons, the
    explanation lines, warnings and notes; followed by validation and baseline provenance."""
    from .. import CAUSE_LABELS

    out: list[str] = []
    out.append(f"# celldx report ({report.get('celldx_version')}, profile {report.get('profile')}, config {report.get('config_hash')})")
    out.append("")
    out.append(f"Generated {report.get('generated_at')}. Probabilities are **{report.get('probability_status')}** "
               f"(calibration: {report.get('calibration_source')}). {report.get('n_cells')} cells, "
               f"{report.get('n_faults')} with fault probability >= 0.5.")
    for bid, warns in (report.get("baseline_warnings") or {}).items():
        for w in warns:
            out.append(f"> **Baseline warning ({bid})**: {w}")
    out.append("")
    for c in report.get("cells", []):
        conf = c.get("confidence", {}) or {}
        out.append(f"## {c.get('cell_id')} ({c.get('pack_id')}/{c.get('string_id')}, {c.get('mode')})")
        out.append("")
        out.append(f"**{c.get('headline')}**")
        out.append("")
        cp = c.get("cause_probabilities", {}) or {}
        ranked = sorted(cp.items(), key=lambda kv: -kv[1])
        out.append("Cause probabilities: " + ", ".join(f"{CAUSE_LABELS.get(k, k)} {v:.2f}" for k, v in ranked if v >= 0.01))
        out.append("")
        out.append(f"Confidence {conf.get('level')} ({_num(conf.get('value'))})"
                   + ("; " + "; ".join(p.get("reason", p.get("name", "")) for p in conf.get("penalties", [])) if conf.get("penalties") else ""))
        for r in conf.get("trust_reasons", []) or []:
            out.append(f"- baseline: {r}")
        out.append("")
        for line in c.get("explanation", []) or []:
            out.append(f"- {line}")
        for w in c.get("warnings", []) or []:
            out.append(f"- warning: {w}")
        for n in c.get("notes", []) or []:
            out.append(f"- note: {n}")
        if c.get("tree_verdict") and not c.get("tree_agrees", True):
            out.append(f"- decision-tree fallback says {CAUSE_LABELS.get(c['tree_verdict'], c['tree_verdict'])} (disagrees)")
        out.append("")
    v = report.get("validation", {}) or {}
    out.append("## Data validation")
    out.append("")
    for k, val in v.items():
        if isinstance(val, (int, float, str)) or val is None:
            out.append(f"- {k}: {val}")
        elif isinstance(val, list) and len(val) <= 20:
            out.append(f"- {k}: {val}")
        elif isinstance(val, dict) and len(val) <= 20:
            out.append(f"- {k}: {val}")
    out.append("")
    out.append("## Baselines")
    out.append("")
    for bid, b in (report.get("baselines") or {}).items():
        prov = b.get("provenance", {}) if isinstance(b, Mapping) else {}
        out.append(f"- {bid}: kind {b.get('kind')}, R0(25 C) {b.get('r0_25c_ohm') if not isinstance(b.get('r0_25c_ohm'), Mapping) else b['r0_25c_ohm'].get('value')}, "
                   f"dataset {prov.get('dataset', b.get('dataset', '-'))}")
    return "\n".join(out) + "\n"
