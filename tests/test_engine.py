"""Diagnosis engine: report contract, verdicts on simulated packs, determinism, streaming call,
pulse mode, and exact regression against the independent reference prototype."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from celldx import CAUSES, NON_FAULT_CAUSES
from celldx.diagnosis.engine import REPORT_SCHEMA, Calibration, detect_mode, diagnose, diagnose_latest
from celldx.reference.prototype import diagnose_cell
from celldx.sim.evaluate import population_baseline, run_trial
from celldx.sim.harness_a import EXPECTED, SCENARIOS

# ----------------------------------------------------------------------------- report contract


@pytest.fixture(scope="module")
def weld_report(trial, cfg):
    df, truths = trial("A", "weld", 1)
    return diagnose(df, None, cfg, seed=1), truths


def test_report_contract(weld_report):
    rep, truths = weld_report
    assert rep["schema"] == REPORT_SCHEMA
    assert rep["probability_status"] == "uncalibrated_scores"
    assert rep["n_cells"] == len(truths) == len(rep["cells"])
    fps = [c["fault_probability"] for c in rep["cells"]]
    assert fps == sorted(fps, reverse=True), "cells are sorted by fault probability"
    for c in rep["cells"]:
        p = c["cause_probabilities"]
        assert set(p) == set(CAUSES)
        assert sum(p.values()) == pytest.approx(1.0)
        assert all(0.0 <= v <= 1.0 for v in p.values())
        assert c["fault_probability"] == pytest.approx(1 - p["healthy"] - p["temperature_effect"])
        assert c["verdict"] == max(p, key=p.get)
        assert c["is_fault"] == (c["verdict"] not in NON_FAULT_CAUSES)
        lo, hi = c["fault_probability_interval"]
        assert 0.0 <= lo <= hi <= 1.0
        assert c["confidence"]["level"] in ("High", "Medium", "Low")
        assert 0.0 <= c["confidence"]["value"] <= 1.0
        assert c["headline"] and c["explanation"]
        assert c["baseline_ref"]["kind"] == "own"
    json.dumps(rep)  # serialisable as is


def test_report_names_what_the_data_cannot_support(weld_report):
    rep, _ = weld_report
    c = rep["cells"][0]
    assert "feature_availability" in c and isinstance(c["feature_availability"], dict)
    # no OCV in simulator A's healthy-neighbour stream is reported, not silently assumed
    assert "has_ocv" in c["flags"]


def test_determinism(trial, cfg):
    df, _ = trial("A", "plating", 0)
    a = diagnose(df, None, cfg, seed=3)
    b = diagnose(df, None, cfg, seed=3)

    def strip(x):  # time stamps are the only thing allowed to differ
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in ("generated_at", "fit_date")}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x

    assert json.dumps(strip(a), sort_keys=True) == json.dumps(strip(b), sort_keys=True)


def test_parallel_feature_extraction_matches_serial(trial, cfg):
    df, _ = trial("A", "sei", 0)
    a = diagnose(df, None, cfg, bootstrap=False, n_jobs=1)
    b = diagnose(df, None, cfg, bootstrap=False, n_jobs=2)
    pa = {c["cell_id"]: c["cause_probabilities"] for c in a["cells"]}
    pb = {c["cell_id"]: c["cause_probabilities"] for c in b["cells"]}
    for cid in pa:
        for k in CAUSES:
            assert pa[cid][k] == pytest.approx(pb[cid][k], abs=1e-12)


def test_insufficient_data_is_reported_not_guessed(cfg):
    df = pd.DataFrame({"t": np.arange(5.0), "cell_id": "C1", "resistance_ohm": 0.05, "temp_cell_c": 25.0})
    c = diagnose(df, None, cfg)["cells"][0]
    assert c["insufficient_data"] is True
    assert c["fault_probability"] is None
    assert c["confidence"]["level"] == "Low"
    assert "Insufficient data" in c["headline"]


def test_calibration_changes_status(trial, cfg):
    df, _ = trial("A", "healthy", 0)
    cal = Calibration(fault=lambda p: np.clip(np.asarray(p) * 0.5, 0, 1), multiclass_t=1.0, source="test")
    rep = diagnose(df, None, cfg, calibration=cal, bootstrap=False)
    assert rep["probability_status"] == "calibrated"
    for c in rep["cells"]:
        assert c["fault_probability"] == pytest.approx(0.5 * c["fault_probability_uncalibrated"])


def test_diagnose_latest_scores_one_cell(trial, cfg):
    df, _ = trial("A", "cooling", 2)
    c = diagnose_latest(df[df.cell_id == "C00"], None, cfg, neighbours=df, bootstrap=False)
    assert c["cell_id"] == "C00"
    assert c["verdict"] == "cooling_fault"
    with pytest.raises(ValueError):
        diagnose_latest(df, None, cfg)


def test_pack_baseline_mode(trial, cfg):
    """A pack-median baseline only makes sense for matched cells (simulator A draws R0 from
    35-70 mOhm, so the cells are first rescaled to one R0, as in a matched production pack)."""
    df, truths = trial("A", "corrosion", 0)
    r0 = {t.cell_id: t.r0_ohm for t in truths}
    med = float(np.median(list(r0.values())))
    df["resistance_ohm"] = df["resistance_ohm"] * med / df["cell_id"].map(r0)
    rep = diagnose(df, "pack", cfg, bootstrap=False)
    assert {c["baseline_ref"]["kind"] for c in rep["cells"]} == {"population"}
    assert {c["baseline_ref"]["baseline_id"] for c in rep["cells"]} == {"pack-median-P0"}
    by = {c["cell_id"]: c for c in rep["cells"]}
    assert by["C00"]["verdict"] == "mechanical_contact_corrosion"
    assert not any(by[c]["is_fault"] for c in by if c != "C00")


# ----------------------------------------------------------------------------- simulator A verdicts


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("seed", [0, 1])
def test_sim_a_verdicts(scenario, seed):
    recs = run_trial({"sim": "A", "scenario": scenario, "seed": seed, "bootstrap": False})
    target = next(r for r in recs if r["target"])
    assert target["class_correct"], (scenario, target["verdict"])
    if EXPECTED[scenario] not in NON_FAULT_CAUSES:
        assert target["verdict"] == EXPECTED[scenario]
        assert target["fault_probability"] >= 0.5
    # no healthy neighbour raises a false alarm
    assert not any(r["false_alarm"] for r in recs if not r["target"])


@pytest.mark.parametrize("scenario", ["internal_damage", "cooling"])
def test_sim_b_safety_and_cooling_without_retuning(scenario):
    for seed in range(3):
        recs = run_trial({"sim": "B", "scenario": scenario, "seed": seed, "bootstrap": False})
        target = next(r for r in recs if r["target"])
        assert target["verdict"] == EXPECTED[scenario]


# ----------------------------------------------------------------------------- reference prototype


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_matches_reference_prototype_exactly(trial, cfg, scenario):
    """Prototype profile, one cell, given baseline: the package and the independent loop-based
    reference implementation agree to floating-point precision on features, probabilities,
    verdict and base confidence."""
    for seed in range(3):
        df, tr = trial("A", scenario, seed)
        c0 = df[df.cell_id == "C00"].sort_values("t")
        bl = population_baseline(tr[0].r0_ohm, 0.0, tr[0].ea_j_per_mol, "C00")
        c = diagnose(c0, bl, cfg, bootstrap=False)["cells"][0]
        ref = diagnose_cell(c0.resistance_ohm.tolist(), c0.temp_cell_c.tolist(), c0.soh.tolist(), c0.ocv_mv.tolist(),
                            c0.temp_pack_c.tolist(), c0.cold_charge.tolist(), tr[0].r0_ohm, tr[0].ea_j_per_mol,
                            t=c0.t.tolist(), baseline_trust=0.85)
        assert c["verdict"] == ref["verdict"]
        for k, v in ref["probabilities"].items():
            assert c["cause_probabilities"][k] == pytest.approx(v, abs=1e-12)
        assert c["confidence"]["components"]["base"] == pytest.approx(ref["confidence_base"], abs=1e-12)
        for k, v in ref["features"].items():
            got = c["features"].get(k)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                assert got is None or np.isnan(got)
            else:
                assert got == pytest.approx(v, rel=1e-9, abs=1e-12), k


# ----------------------------------------------------------------------------- pulse mode


@pytest.fixture(scope="module")
def pulse_frame():
    from celldx.sim.harness_a import simulate_trial
    from celldx.sim.pulse import render_pulses

    daily, truths = simulate_trial("weld", seed=4, n_days=240, n_cells=4)
    return daily, render_pulses(daily, seed=4), truths


def test_pulse_mode_detected_and_resistance_extracted(pulse_frame, cfg):
    from celldx.schema import coerce

    daily, raw, _ = pulse_frame
    assert detect_mode(coerce(raw[raw.cell_id == "C01"]), cfg) == "pulse_mode"
    assert detect_mode(coerce(daily[daily.cell_id == "C01"]), cfg) == "cycle_mode"
    rep = diagnose(raw, None, cfg, bootstrap=False)
    c = next(x for x in rep["cells"] if x["cell_id"] == "C01")
    assert c["mode"] == "pulse_mode"
    assert c["extraction"]["n_pulses"] >= 2 * 200
    # the 1 s pulse resistance equals the daily resistance the pulses were rendered from
    assert c["extraction"]["pulse_r_vs_bms_rel"] < 0.02


def test_pulse_mode_weld_verdict(pulse_frame, cfg):
    _, raw, _ = pulse_frame
    rep = diagnose(raw, None, cfg, bootstrap=False)
    c = next(x for x in rep["cells"] if x["cell_id"] == "C00")
    assert c["verdict"] == "mechanical_weld_or_tab"
