"""Calibration maps and grouped cross-validation; the action planner and its safety rules."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from celldx import CAUSES
from celldx import calibration as cal
from celldx.actions import ActionPlanner, RecommendationLog, SimulatedBypassController, execute, plan_report

# ----------------------------------------------------------------------------- calibration


def _miscalibrated(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    p_true = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p_true).astype(float)
    return p_true ** 3, y   # reported scores far too low in the middle


def test_isotonic_and_platt_reduce_ece():
    p, y = _miscalibrated()
    before = cal.ece(p, y)
    iso = cal.IsotonicMap.fit(p, y)
    platt = cal.PlattMap.fit(p, y)
    assert cal.ece(iso(p), y) < 0.3 * before
    assert cal.ece(platt(p), y) < 0.5 * before
    q = np.linspace(0, 1, 50)
    assert np.all(np.diff(iso(q)) >= -1e-12), "isotonic map is monotone"
    assert cal.brier(iso(p), y) < cal.brier(p, y)


def test_platt_degenerate_labels():
    m = cal.PlattMap.fit(np.array([0.2, 0.4]), np.array([1.0, 1.0]))
    assert (m.a, m.b) == (1.0, 0.0)


def test_temperature_scaling_recovers_overconfidence():
    rng = np.random.default_rng(1)
    n, k = 3000, len(CAUSES)
    logits = rng.normal(0, 1.5, (n, k))
    P_true = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(k, p=row) for row in P_true])
    P_over = cal.apply_temperature(P_true, 1 / 2.5)    # sharpen: overconfident scores
    t = cal.fit_temperature(P_over, y)
    assert t == pytest.approx(2.5, rel=0.15)
    assert cal.multiclass_log_loss(cal.apply_temperature(P_over, t), y) < cal.multiclass_log_loss(P_over, y)


def _labelled_table(n_groups=200, seed=2):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        label = CAUSES[g % len(CAUSES)]
        P = rng.dirichlet(np.ones(len(CAUSES)) * 0.5)
        P[CAUSES.index(label)] += 1.5
        P /= P.sum()
        pred = CAUSES[int(np.argmax(P))]
        fp = 1 - P[0] - P[1]
        conf = float(rng.uniform(0.3, 0.95))
        rows.append({"group": f"g{g}", "label": label, "fault_probability": fp ** 2, **{f"p_{k}": P[i] for i, k in enumerate(CAUSES)},
                     "confidence": conf, "class_correct": (pred == label) or (label in ("healthy", "temperature_effect")
                                                                             and pred in ("healthy", "temperature_effect"))})
    return pd.DataFrame(rows)


def test_grouped_cross_validation_has_no_leakage_and_helps():
    df = _labelled_table()
    cv = cal.cross_validate(df, "isotonic", n_splits=5)
    assert cv["group_overlap_last_fold"] == 0
    assert cv["after"]["fault_ece"] < cv["before"]["fault_ece"]
    assert cv["after"]["confidence_ece"] < cv["before"]["confidence_ece"]
    assert set(cv["before"]) >= {"fault_brier", "fault_log_loss", "multiclass_log_loss", "per_class", "confusion"}


def test_fitted_calibration_round_trip(tmp_path):
    df = _labelled_table()
    for method in ("isotonic", "platt"):
        fc = cal.fit(df, method, source="unit test")
        p = tmp_path / f"cal_{method}.json"
        fc.save(p)
        back = cal.FittedCalibration.load(p)
        x = np.linspace(0, 1, 11)
        assert np.allclose(back.fault(x), fc.fault(x))
        assert back.temperature == pytest.approx(fc.temperature)
        eng = back.to_engine()
        assert eng.status == "calibrated"
        assert json.loads(p.read_text())["schema"] == "celldx-calibration/1"


def test_reliability_png(tmp_path):
    p, y = _miscalibrated(500)
    out = tmp_path / "rel.png"
    cal.reliability_png(p, y, cal.IsotonicMap.fit(p, y)(p), out, "test")
    assert out.stat().st_size > 1000


# ----------------------------------------------------------------------------- actions


def _rep(cid, verdict, *, level="High", fp=0.9, probs=None, **features):
    probs = probs or {k: (0.9 if k == verdict else 0.1 / (len(CAUSES) - 1)) for k in CAUSES}
    return {"cell_id": cid, "verdict": verdict, "confidence": {"value": 0.8 if level == "High" else 0.3, "level": level},
            "fault_probability": fp, "cause_probabilities": probs, "features": features,
            "pack_id": "P", "string_id": "S", "probability_status": "uncalibrated_scores"}


def test_internal_damage_is_a_safety_bypass(cfg):
    r = ActionPlanner(cfg).plan_cell(_rep("C1", "internal_damage"))
    assert r.bypass and r.safety and not r.transfer_allowed
    assert "bypass_immediately" in r.actions


def test_low_confidence_never_recommends_irreversible_actions(cfg):
    planner = ActionPlanner(cfg)
    irr = set(cfg.actions.irreversible)
    for verdict in CAUSES:
        r = planner.plan_cell(_rep("C1", verdict, level="Low"))
        assert not (set(r.actions) & irr), (verdict, r.actions)
        if r.blocked_irreversible:
            assert "inspect" in r.actions and "collect_more_data" in r.actions
            assert "held back" in r.text


def test_severe_plating_bypass_and_mild_plating_transfer(cfg):
    p = cfg.actions.severe_plating_probability
    severe = {k: 0.0 for k in CAUSES} | {"lithium_plating": p + 0.05, "healthy": 1 - p - 0.05}
    mild = {k: 0.0 for k in CAUSES} | {"lithium_plating": p - 0.2, "healthy": 1 - p + 0.2}
    planner = ActionPlanner(cfg)
    assert planner.plan_cell(_rep("C1", "lithium_plating", probs=severe)).bypass
    assert not planner.plan_cell(_rep("C1", "lithium_plating", probs=mild)).bypass


def test_conditional_bypasses(cfg):
    planner = ActionPlanner(cfg)
    assert planner.plan_cell(_rep("C1", "mechanical_weld_or_tab", cell_minus_pack_k=4.0)).bypass
    assert not planner.plan_cell(_rep("C1", "mechanical_weld_or_tab", cell_minus_pack_k=1.0)).bypass
    assert planner.plan_cell(_rep("C1", "cooling_fault", temp_recent_c=52.0)).bypass
    assert not planner.plan_cell(_rep("C1", "cooling_fault", temp_recent_c=35.0)).bypass
    assert planner.plan_cell(_rep("C1", "mechanical_contact_corrosion", growth_b=1e-4)).bypass


def test_string_constraints_block_and_escalate(cfg):
    reps = [_rep(f"C{i}", "internal_damage", soh_recent=0.95) for i in range(3)] + \
           [_rep(f"C{i}", "healthy", fp=0.05, soh_recent=0.95) for i in range(3, 12)]
    plan = ActionPlanner(cfg).plan_string(reps, max_bypassed=2)
    assert len(plan.bypass) == 2
    assert len(plan.blocked_bypass) == 1 and plan.blocked_bypass[0]["safety"]
    assert any("derate or stop" in e for e in plan.escalations)


def test_min_string_voltage_constraint(cfg):
    reps = [_rep(f"C{i}", "internal_damage", soh_recent=0.95) for i in range(2)] + \
           [_rep(f"C{i}", "healthy", fp=0.05, soh_recent=0.95) for i in range(2, 10)]
    plan = ActionPlanner(cfg).plan_string(reps, min_string_voltage_v=10 * 3.1, max_bypassed=5)
    assert len(plan.bypass) == 0          # all 10 cells are needed for the minimum string voltage
    assert len(plan.escalations) == 2     # both safety bypasses are blocked and escalated
    plan2 = ActionPlanner(cfg).plan_string(reps, min_string_voltage_v=9 * 3.1, max_bypassed=5)
    assert len(plan2.bypass) == 1         # 9 x 3.1 V exactly (no float off-by-one)
    assert plan2.constraints["string_voltage_after_v"] >= 9 * 3.1 - 1e-9


def test_bypass_economics_for_a_weak_cell(cfg):
    reps = [_rep("W", "ambiguous_ageing", fp=0.3, soh_recent=0.70)] + \
           [_rep(f"C{i}", "healthy", fp=0.05, soh_recent=0.95) for i in range(7)]
    plan = ActionPlanner(cfg).plan_string(reps, active_balancing=True)
    assert plan.bypass == ["W"]
    e = plan.economics[0]
    assert e["bypass_capacity"] == pytest.approx(7 * 0.95)
    assert e["keep_capacity"] == pytest.approx(8 * 0.70)
    # a marginal cell is kept
    reps[0] = _rep("W", "ambiguous_ageing", fp=0.3, soh_recent=0.90)
    assert ActionPlanner(cfg).plan_string(reps, active_balancing=True).bypass == []


def test_transfers_only_between_safe_cells(cfg):
    reps = [_rep("D", "internal_damage", soc_recent=0.9, soh_recent=0.95),
            _rep("H1", "healthy", fp=0.05, soc_recent=0.8, soh_recent=0.95),
            _rep("H2", "healthy", fp=0.05, soc_recent=0.4, soh_recent=0.95),
            _rep("H3", "healthy", fp=0.05, soc_recent=0.6, soh_recent=0.95)]
    plan = ActionPlanner(cfg).plan_string(reps)
    for tr in plan.transfers:
        assert "D" not in (tr.from_cell, tr.to_cell)
    assert any(tr.from_cell == "H1" and tr.to_cell == "H2" for tr in plan.transfers)


def test_execute_is_dry_run_by_default_and_needs_confirmation(cfg, tmp_path):
    report = {"cells": [_rep("C1", "internal_damage", soh_recent=0.9)] +
              [_rep(f"C{i}", "healthy", fp=0.05, soh_recent=0.95) for i in range(2, 8)], "probability_status": "x"}
    plan = plan_report(report, cfg)
    assert plan["dry_run"] is True
    ctl = SimulatedBypassController([c["cell_id"] for c in report["cells"]])
    res = execute(plan, ctl, cfg)
    assert res and all(not r.executed and r.dry_run for r in res)
    assert ctl.log == []
    res = execute(plan, ctl, cfg, dry_run=False)
    assert all(not r.executed for r in res) and ctl.log == []
    res = execute(plan, ctl, cfg, dry_run=False, confirm=lambda r: False)
    assert all(not r.executed for r in res)
    log = RecommendationLog(tmp_path / "log.jsonl")
    res = execute(plan, ctl, cfg, dry_run=False, confirm=lambda r: True, log=log)
    assert all(r.executed for r in res) and ctl.state()["C1"] is True
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert any(json.loads(x)["event"] == "bypass" for x in lines)
    assert all("logged_at" in json.loads(x) for x in lines)
