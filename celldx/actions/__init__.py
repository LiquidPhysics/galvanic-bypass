"""Action layer: verdict -> recommendation, bypass economics, transfers, and safe actuation.

Safety rules (section 8)
------------------------
* Default is recommend-only (``actions.dry_run: true``). Real actuation needs
  ``dry_run=False`` **and** a human-confirmation hook that returns True (unless
  ``require_human_confirmation`` is switched off in config).
* Safety verdicts (internal damage, severe plating) are always bypassed, regardless of the
  economics, as long as the string constraints allow it; if they do not, the string is
  flagged for derating and escalation.
* With Low confidence, irreversible actions (``actions.irreversible``) are never recommended
  automatically: they become ``inspect`` + ``collect_more_data``.
* Charge transfers are only planned between safe cells: never to or from a cell with internal
  damage or severe plating.
* Every recommendation is logged with its evidence (JSON lines).

Bypass economics for weak cells
-------------------------------
A series string delivers the capacity of its weakest cell. Keeping all N cells gives
``N * SOH_weak`` (in units of one new cell); bypassing the weakest gives ``(N - 1) * SOH_next``
where SOH_next is the weakest remaining cell. Bypass when the second is larger by more than
``balancing_margin`` (when active balancing exists, since balancing already recovers part of
the weak cell's limit). Applied greedily while the constraints hold: minimum string voltage,
maximum bypassed cells per string, switch response time.
"""

from __future__ import annotations

import abc
import datetime as _dt
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config


@dataclass
class Recommendation:
    cell_id: str
    verdict: str
    confidence_level: str
    actions: list[str]
    bypass: bool = False
    bypass_reason: str | None = None
    safety: bool = False
    transfer_allowed: bool = True
    blocked_irreversible: list[str] = field(default_factory=list)
    text: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Transfer:
    from_cell: str
    to_cell: str
    soc_delta: float
    reason: str


@dataclass
class StringPlan:
    string_id: str
    n_cells: int
    recommendations: list[Recommendation]
    bypass: list[str]
    blocked_bypass: list[dict[str, Any]]
    transfers: list[Transfer]
    economics: list[dict[str, Any]]
    constraints: dict[str, Any]
    escalations: list[str]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


ACTION_TEXT = {
    "none": "no action",
    "fix_ventilation_heating_or_cooling": "fix ventilation, heating or cooling",
    "derate": "derate the cell / string",
    "check_airflow_or_coolant": "check airflow or coolant",
    "lower_cold_charge_current_or_preheat": "lower the cold-charge current or preheat",
    "move_charge_away": "move charge away from the cell",
    "bypass_immediately": "bypass immediately",
    "flag_for_replacement": "flag for replacement",
    "no_charge_transfer": "no charge transfer to or from this cell",
    "rework_weld": "schedule re-weld or re-torque",
    "clean_and_retorque": "clean and re-torque the contacts",
    "cell_to_cell_transfer": "use cell-to-cell transfer",
    "reduce_stress": "reduce stress (current, depth of discharge, temperature)",
    "plan_replacement": "plan replacement",
    "keep_monitoring": "keep monitoring",
    "inspect": "inspect the cell",
    "collect_more_data": "collect more data before acting",
    "bypass": "bypass",
}


class ActionPlanner:
    """Maps a cell report (from :func:`celldx.diagnosis.engine.diagnose`) to recommendations."""

    def __init__(self, cfg: Config, mapping: Mapping[str, Sequence[str]] | None = None) -> None:
        self.cfg = cfg
        self.a = cfg.actions
        self.mapping = {k: list(v) for k, v in (mapping or self.a.mapping.to_dict()).items()}

    # --------------------------------------------------------------- per cell
    def plan_cell(self, rep: Mapping[str, Any]) -> Recommendation:
        a = self.a
        verdict = str(rep["verdict"])
        conf = rep.get("confidence", {}) or {}
        level = str(conf.get("level", "Low"))
        f = rep.get("features", {}) or {}
        probs = rep.get("cause_probabilities", {}) or {}
        acts = list(self.mapping.get(verdict, ["keep_monitoring"]))
        bypass, reason, safety = False, None, False
        transfer_ok = True
        if rep.get("insufficient_data"):
            acts = ["collect_more_data"]
        t_cell = _f(f.get("temp_recent_c"))
        if verdict == "internal_damage":
            bypass, reason, safety, transfer_ok = True, "safety verdict: internal damage", True, False
        elif verdict == "lithium_plating":
            p = float(probs.get("lithium_plating", 0.0))
            if p >= float(a.severe_plating_probability):
                bypass, reason, safety, transfer_ok = True, f"severe plating (p={p:.2f})", True, False
                acts.append("bypass")
        elif verdict == "cooling_fault":
            if math.isfinite(t_cell) and t_cell >= float(a.temp_limit_c) - float(a.cooling_bypass_temp_margin_k):
                bypass, reason = True, f"cell at {t_cell:.1f} C, within {a.cooling_bypass_temp_margin_k} K of the {a.temp_limit_c} C limit"
                acts.append("bypass")
        elif verdict == "mechanical_weld_or_tab":
            heat = _f(f.get("cell_minus_pack_k"))
            if math.isfinite(heat) and heat >= float(a.bypass_if_heating_k):
                bypass, reason = True, f"loose contact heating the cell ({heat:+.1f} K vs pack)"
                acts.append("bypass")
        elif verdict == "mechanical_contact_corrosion":
            b = _f(f.get("growth_b"))
            if math.isfinite(b) and b > float(a.corrosion_still_rising_b):
                bypass, reason = True, "contact resistance still rising"
                acts.append("bypass")
        if verdict in list(a.safety_verdicts):
            safety = True
        blocked: list[str] = []
        if level == "Low":
            irr = set(a.irreversible)
            blocked = [x for x in acts if x in irr]
            if blocked:
                acts = [x for x in acts if x not in irr] + ["inspect", "collect_more_data"]
        evidence = {
            "fault_probability": rep.get("fault_probability"),
            "fault_probability_interval": rep.get("fault_probability_interval"),
            "cause_probabilities": probs, "confidence": conf.get("value"), "confidence_level": level,
            "probability_status": rep.get("probability_status"),
            "key_features": {k: f.get(k) for k in ("corrected_excess", "raw_excess", "step_rel", "step_z", "b_rel",
                                                    "growth_r2", "cell_minus_pack_k", "post_cold_charge_rise", "soh_recent",
                                                    "ocv_recent_mv")},
        }
        txt = "; ".join(ACTION_TEXT.get(x, x) for x in dict.fromkeys(acts))
        if blocked:
            txt += f" (confidence Low: {', '.join(ACTION_TEXT.get(x, x) for x in blocked)} held back)"
        return Recommendation(cell_id=str(rep["cell_id"]), verdict=verdict, confidence_level=level,
                              actions=list(dict.fromkeys(acts)), bypass=bypass, bypass_reason=reason, safety=safety,
                              transfer_allowed=transfer_ok, blocked_irreversible=blocked, text=txt, evidence=evidence)

    # --------------------------------------------------------------- per string
    def plan_string(self, reports: Sequence[Mapping[str, Any]], *, string_id: str = "string0",
                    min_string_voltage_v: float | None = None, v_nominal_cell_v: float | None = None,
                    max_bypassed: int | None = None, active_balancing: bool | None = None,
                    safety_deadline_s: float | None = None) -> StringPlan:
        a = self.a
        n = len(reports)
        vnom = float(v_nominal_cell_v if v_nominal_cell_v is not None else a.v_nominal_cell_v)
        vmin = float(min_string_voltage_v if min_string_voltage_v is not None else a.min_string_voltage_v)
        kmax = int(max_bypassed if max_bypassed is not None else a.max_bypassed_per_string)
        bal = bool(active_balancing if active_balancing is not None else a.active_balancing)
        n_min = max(int(a.min_cells_in_string), math.ceil(vmin / vnom - 1e-9) if vmin > 0 else int(a.min_cells_in_string))
        recs = [self.plan_cell(r) for r in reports]
        by_id = {r.cell_id: r for r in recs}
        soh = {str(r["cell_id"]): _f((r.get("features") or {}).get("soh_recent")) for r in reports}
        bypass: list[str] = []
        blocked: list[dict[str, Any]] = []
        escal: list[str] = []

        def room() -> bool:
            return len(bypass) < kmax and (n - len(bypass) - 1) >= n_min

        # 1) safety first, then other bypass-worthy verdicts, highest fault probability first
        order = sorted(recs, key=lambda r: (not r.safety, -(r.evidence.get("fault_probability") or 0.0)))
        for r in order:
            if not r.bypass:
                continue
            if room():
                bypass.append(r.cell_id)
            else:
                blocked.append({"cell_id": r.cell_id, "reason": "string constraints (max bypassed / minimum voltage)",
                                "safety": r.safety})
                if r.safety:
                    escal.append(f"{r.cell_id}: safety bypass blocked by string constraints -> derate or stop string {string_id}")
                r.bypass = False
        # 2) economics on weak cells
        econ: list[dict[str, Any]] = []
        margin = float(a.balancing_margin) if bal else 0.0
        while True:
            active = [c for c in soh if c not in bypass and math.isfinite(soh[c])]
            if len(active) < 2:
                break
            ranked = sorted(active, key=lambda c: soh[c])
            weak, nxt = ranked[0], ranked[1]
            m = len(active)
            keep = m * soh[weak]
            byp = (m - 1) * soh[nxt]
            rec = {"weak_cell": weak, "soh_weak": soh[weak], "soh_next": soh[nxt], "keep_capacity": keep,
                   "bypass_capacity": byp, "threshold": keep * (1 + margin), "bypass": False}
            if byp > keep * (1 + margin):
                if room():
                    bypass.append(weak)
                    rec["bypass"] = True
                    br = by_id.get(weak)
                    if br is not None:
                        br.bypass = True
                        br.bypass_reason = (f"bypass economics: (N-1) x SOH_next = {byp:.3f} > N x SOH_weak = {keep:.3f}"
                                            + (f" x (1 + {margin})" if margin else ""))
                        br.actions.append("bypass")
                    econ.append(rec)
                    continue
                rec["blocked"] = "string constraints"
            econ.append(rec)
            break
        # 3) transfers between safe cells only
        unsafe = {r.cell_id for r in recs if not r.transfer_allowed}
        socs = {str(r["cell_id"]): _f((r.get("features") or {}).get("soc_recent")) for r in reports}
        safe = [c for c in socs if c not in unsafe and c not in bypass and math.isfinite(socs[c])]
        transfers: list[Transfer] = []
        if len(safe) >= 2:
            mean = sum(socs[c] for c in safe) / len(safe)
            tol = float(a.transfer_soc_tolerance)
            donors = sorted([c for c in safe if socs[c] > mean + tol], key=lambda c: -socs[c])
            recv = sorted([c for c in safe if socs[c] < mean - tol], key=lambda c: socs[c])
            for dcell, rcell in zip(donors, recv, strict=False):
                delta = min(socs[dcell] - mean, mean - socs[rcell])
                transfers.append(Transfer(dcell, rcell, float(delta), "equalise SOC between safe cells"))
        # move charge away from mild-plating cells
        for r in recs:
            if r.verdict == "lithium_plating" and r.transfer_allowed and safe:
                tgt = min((c for c in safe if c != r.cell_id), key=lambda c: socs[c], default=None)
                if tgt:
                    transfers.append(Transfer(r.cell_id, tgt, float("nan"), "move charge away from plating cell"))
        resp = float(a.switch_response_s)
        cons = {"n_cells": n, "n_min_in_string": n_min, "max_bypassed": kmax, "min_string_voltage_v": vmin,
                "v_nominal_cell_v": vnom, "string_voltage_after_v": (n - len(bypass)) * vnom,
                "switch_response_s": resp, "active_balancing": bal}
        if safety_deadline_s is not None and resp > safety_deadline_s:
            escal.append(f"switch response {resp} s exceeds the safety deadline {safety_deadline_s} s")
        return StringPlan(string_id, n, recs, bypass, blocked, transfers, econ, cons, escal)


def plan_report(report: Mapping[str, Any], cfg: Config, **kw: Any) -> dict[str, Any]:
    """Plan every string in a diagnosis report. Returns a JSON-serialisable dict."""
    planner = ActionPlanner(cfg)
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for c in report["cells"]:
        groups.setdefault((c.get("pack_id", "pack0"), c.get("string_id", "string0")), []).append(c)
    plans = [planner.plan_string(cells, string_id=f"{p}/{s}", **kw).to_dict() for (p, s), cells in groups.items()]
    return {"schema": "celldx-plan/1", "dry_run": bool(cfg.actions.dry_run),
            "probability_status": report.get("probability_status"), "strings": plans}


# ------------------------------------------------------------------- actuation
@dataclass
class ActuationResult:
    cell_id: str
    requested: bool
    executed: bool
    dry_run: bool
    reason: str


class BypassController(abc.ABC):
    """Abstract switch controller. Implementations talk to real hardware or a simulator."""

    @abc.abstractmethod
    def set_bypass(self, cell_id: str, bypass: bool) -> None: ...

    @abc.abstractmethod
    def state(self) -> dict[str, bool]: ...


class SimulatedBypassController(BypassController):
    def __init__(self, cells: Sequence[str]) -> None:
        self._state = {c: False for c in cells}
        self.log: list[tuple[str, bool]] = []

    def set_bypass(self, cell_id: str, bypass: bool) -> None:
        if cell_id not in self._state:
            raise KeyError(cell_id)
        self._state[cell_id] = bypass
        self.log.append((cell_id, bypass))

    def state(self) -> dict[str, bool]:
        return dict(self._state)


class RecommendationLog:
    """Append-only JSON-lines log of every recommendation with its evidence."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self.records: list[dict[str, Any]] = []

    def write(self, rec: Mapping[str, Any]) -> None:
        row = {"logged_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"), **rec}
        self.records.append(row)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")


def execute(plan: Mapping[str, Any], controller: BypassController, cfg: Config, *, dry_run: bool | None = None,
            confirm: Callable[[Mapping[str, Any]], bool] | None = None, log: RecommendationLog | None = None) -> list[ActuationResult]:
    """Apply a plan's bypass decisions through ``controller``.

    Dry run (default from config) only logs. Real actuation requires ``dry_run=False`` and, when
    ``require_human_confirmation`` is on, a ``confirm`` hook that approves each switch.
    """
    dry = bool(cfg.actions.dry_run) if dry_run is None else bool(dry_run)
    need_conf = bool(cfg.actions.require_human_confirmation)
    log = log or RecommendationLog(None)
    out: list[ActuationResult] = []
    for s in plan["strings"]:
        recs = {r["cell_id"]: r for r in s["recommendations"]}
        for cid in s["bypass"]:
            r = recs.get(cid, {"cell_id": cid})
            if dry:
                res = ActuationResult(cid, True, False, True, "dry run: recommendation only")
            elif need_conf and (confirm is None or not confirm(r)):
                res = ActuationResult(cid, True, False, False, "human confirmation missing or refused")
            else:
                controller.set_bypass(cid, True)
                res = ActuationResult(cid, True, True, False, "executed")
            out.append(res)
            log.write({"event": "bypass", "string": s["string_id"], "result": asdict(res), "recommendation": r})
        for r in s["recommendations"]:
            log.write({"event": "recommendation", "string": s["string_id"], "recommendation": r})
    return out


def _f(x: Any) -> float:
    try:
        return float(x) if x is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")
