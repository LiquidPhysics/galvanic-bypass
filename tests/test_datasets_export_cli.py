"""Dataset registry, checksummed fetch, NASA and REON adapters, dashboard export, and the CLI."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from celldx.cli.main import app
from celldx.datasets import ADAPTER_STATUS, REGISTRY, fetch, load_manifest, nasa
from celldx.datasets.others import daily_from_reon, load_reon_csv
from celldx.datasets.registry import ChecksumError
from celldx.export import FAULT_CODE, layout_id_map, to_dashboard

from .conftest import write_nasa_mat

FIX = Path(__file__).parent / "fixtures"
runner = CliRunner()

# ----------------------------------------------------------------------------- registry and fetch


def test_registry_entries_carry_licence_and_status():
    for name, info in REGISTRY.items():
        assert info.licence and info.landing_page and info.citation, name
        assert info.status in ("tested_against_real_files", "untested_against_real_files")
        assert name in ADAPTER_STATUS
    assert REGISTRY["nasa"].files[0].sha256 and len(REGISTRY["nasa"].files[0].sha256) == 64


def test_fetch_rejects_a_bad_checksum(tmp_path):
    bogus = tmp_path / "fake.zip"
    with zipfile.ZipFile(bogus, "w") as z:
        z.writestr("x.txt", "not the NASA archive")
    with pytest.raises(ChecksumError):
        fetch("nasa", tmp_path / "cache", from_file=bogus)


def test_fetch_without_url_needs_a_manual_file(tmp_path):
    with pytest.raises(RuntimeError, match="Download manually"):
        fetch("calce", tmp_path)


def test_fetch_from_manual_file_writes_manifest(tmp_path):
    arc = tmp_path / "calce_manual.zip"
    with zipfile.ZipFile(arc, "w") as z:
        z.writestr("CS2_35/a.csv", "Cycle_Index\n1\n")
    man = fetch("calce", tmp_path / "cache", from_file=arc)
    assert man["files"][0]["verified"] is False          # no published checksum: recorded, flagged
    assert len(man["files"][0]["sha256"]) == 64
    assert man["extracted"] and all(len(e["sha256"]) == 64 for e in man["extracted"])
    assert load_manifest("calce", tmp_path / "cache")["licence"] == REGISTRY["calce"].licence


# ----------------------------------------------------------------------------- NASA adapter (synthetic .mat)


def test_nasa_adapter_on_a_synthetic_mat(tmp_path):
    p = write_nasa_mat(tmp_path / "B0099.mat", complex_part=0.01)
    name, cycles = nasa.load_cycles(p)
    assert name == "B0099" and len(cycles) == 16
    eis, meta = nasa.cell_table(p, "eis")
    assert len(eis) == 8
    assert np.allclose(eis["resistance_ohm"], eis["r_ohmic_ohm"] + eis["r_slow_ohm"])
    assert eis["r_ohmic_ohm"].iloc[0] == pytest.approx(0.05)
    assert any("imaginary" in n for n in meta["notes"])         # complex Re reported, real part kept
    assert meta["soh0_ah"] == pytest.approx(np.median([2.0, 1.99, 1.98]))
    assert np.all(np.diff(eis["soh"].to_numpy()) <= 1e-12)        # capacity fade -> SOH never rises
    dcir, _ = nasa.cell_table(p, "dcir")
    assert dcir["resistance_ohm"].to_numpy() == pytest.approx(0.05)   # dV/dI of the synthetic step
    assert nasa.find_files(tmp_path) == {"B0099": p}
    with pytest.raises(FileNotFoundError):
        nasa.load(tmp_path, ["B0005"])


@pytest.mark.nasa
def test_nasa_real_files(nasa_path, cfg):
    df, metas = nasa.load(nasa_path, ["B0005"], "eis")
    assert len(df) > 200 and metas["B0005"]["soh0_ah"] == pytest.approx(1.86, abs=0.05)
    assert df["soh"].iloc[-1] < 0.8
    bl, _, _ = nasa.fit_nasa_baseline(nasa_path, ["B0005", "B0006", "B0007", "B0018"], cfg, source="eis")
    assert bl.r0_25c_ohm.value and 0.05 < bl.r0_25c_ohm.value < 0.3
    # all four cells were measured at one ambient: the data cannot support an activation energy
    assert bl.ea_j_per_mol.source == "literature_default"
    assert any("activation energy" in x for x in bl.provenance.data_cannot_support)
    assert all(len(f.sha256) == 64 for f in bl.provenance.files)


# ----------------------------------------------------------------------------- REON / Firebreak


def test_reon_loader_maps_firebreak_columns():
    df = load_reon_csv(FIX / "firebreak_hot_day_small.csv")
    assert {"t", "cell_id", "voltage_v", "current_a", "temp_cell_c", "temp_pack_c", "soc", "soh",
            "resistance_ohm"} <= set(df.columns)
    assert df["resistance_ohm"].median() == pytest.approx(1e-3, rel=0.5)   # mOhm -> ohm
    assert df["cell_id"].nunique() == 32
    daily = daily_from_reon(df)
    assert set(daily["cell_id"]) == set(df["cell_id"]) and len(daily) == 32


# ----------------------------------------------------------------------------- dashboard export


@pytest.fixture(scope="module")
def fb_report(cfg):
    from celldx.diagnosis.engine import diagnose

    df = load_reon_csv(FIX / "firebreak_hot_day_small.csv")
    return diagnose(df, "pack", cfg, bootstrap=True)


def test_dashboard_export_contract(fb_report, cfg):
    from celldx.actions import plan_report

    lay = json.loads((FIX / "firebreak_layout.json").read_text())
    idm = layout_id_map([c["cell_id"] for c in fb_report["cells"]], lay["cell_ids"])
    d = to_dashboard(fb_report, pack_id="S-12", t=68400, plan=plan_report(fb_report, cfg), id_map=idm)
    assert d["schema"] == "celldx/v1" and d["pack_id"] == "S-12" and d["t"] == 68400
    assert d["n_cells"] == 32 and {c["id"] for c in d["cells"]} == set(lay["cell_ids"])
    assert d["faults"] == ["contact", "weld", "sei", "plating", "short", "cooling", "ageing"]
    assert d["probability_status"] == "uncalibrated_scores"
    for c in d["cells"]:
        assert 0 <= c["p_healthy"] <= 1 and c["confidence"]["level"] in ("High", "Medium", "Low")
        assert len([x for x in c["diagnoses"] if x["p"] is not None]) <= 3
        ps = [x["p"] for x in c["diagnoses"] if x["p"] is not None]
        assert ps == sorted(ps, reverse=True)
        for i, x in enumerate(c["diagnoses"]):
            assert all(e["llr"] is None for e in x["evidence"])          # celldx has no likelihood ratios
            if i > 0:
                assert x["ci"] is None                                    # no per-cause intervals
    json.dumps(d, allow_nan=False)


def test_dashboard_flags_firebreak_hidden_contact_cell(fb_report, cfg):
    from celldx.actions import plan_report

    assert to_dashboard(fb_report)["cells"][0]["diagnoses"][0]["action"] is None   # no plan given -> no action
    d = to_dashboard(fb_report, plan=plan_report(fb_report, cfg))
    c20 = next(c for c in d["cells"] if c["id"] == "C20")
    assert c20["is_fault"] and c20["verdict"] == "contact"
    top = c20["diagnoses"][0]
    assert top["fault"] == "contact" and top["action"]
    assert any(e["signal"] == "R_mOhm" for e in top["evidence"])


def test_sensor_fault_becomes_thermistor_diagnosis():
    rep = {"cells": [{"cell_id": "C1", "cause_probabilities": {"healthy": 0.9, "temperature_effect": 0.1},
                      "verdict": "healthy", "is_fault": False, "fault_probability": 0.0, "confidence": {},
                      "flags": {"sensor_fault_suspected": True, "sensor_reasons": ["cell reads 8 K below pack"]}}]}
    d = to_dashboard(rep)
    th = d["cells"][0]["diagnoses"][-1]
    assert th["fault"] == "thermistor" and th["p"] is None and th["evidence"][0]["note"].startswith("cell reads")


def test_fault_codes_cover_every_cause():
    from celldx import CAUSES

    assert set(FAULT_CODE) == set(CAUSES)


def test_layout_id_map():
    m = layout_id_map([f"C{i:02d}" for i in range(32)], [f"C{i:02d}" for i in range(1, 33)])
    assert m["C00"] == "C01" and m["C31"] == "C32"
    with pytest.raises(ValueError):
        layout_id_map(["a", "b", "c"], ["x", "y"])


# ----------------------------------------------------------------------------- CLI


def test_cli_version_and_data_list():
    r = runner.invoke(app, ["version"])
    assert r.exit_code == 0 and r.stdout.strip() == "3.0.0"
    r = runner.invoke(app, ["data", "list"])
    assert r.exit_code == 0 and "nasa" in json.loads(r.stdout)


def test_cli_end_to_end(tmp_path):
    sim = tmp_path / "weld.parquet"
    r = runner.invoke(app, ["simulate", "weld", "--seed", "1", "--out", str(sim), "--truth", str(tmp_path / "t.json")])
    assert r.exit_code == 0, r.output
    rep, txt, dash = tmp_path / "rep.json", tmp_path / "rep.md", tmp_path / "dash.json"
    r = runner.invoke(app, ["diagnose", str(sim), "--out", str(rep), "--text", str(txt), "--dashboard", str(dash),
                            "--pack-id", "P0", "--no-bootstrap"])
    assert r.exit_code == 0, r.output
    report = json.loads(rep.read_text())
    assert report["schema"] == "celldx-report/1"
    assert next(c for c in report["cells"] if c["cell_id"] == "C00")["verdict"] == "mechanical_weld_or_tab"
    assert "C00" in txt.read_text()
    assert json.loads(dash.read_text())["schema"] == "celldx/v1"
    r = runner.invoke(app, ["plan", str(rep), "--out", str(tmp_path / "plan.json")])
    assert r.exit_code == 0 and json.loads((tmp_path / "plan.json").read_text())["dry_run"] is True
    r = runner.invoke(app, ["report", str(rep)])
    assert r.exit_code == 0 and "weld" in r.stdout.lower()
    r = runner.invoke(app, ["export", "dashboard", str(rep), "--layout", str(FIX / "firebreak_layout.json"),
                            "--out", str(tmp_path / "d2.json")])
    assert r.exit_code == 0
    assert {c["id"] for c in json.loads((tmp_path / "d2.json").read_text())["cells"]} <= {f"C{i:02d}" for i in range(1, 33)}
    r = runner.invoke(app, ["diagnose", str(sim), "--latest", "C00", "--no-bootstrap"])
    assert r.exit_code == 0 and json.loads(r.stdout)["cell_id"] == "C00"


def test_cli_overrides_and_bad_keys(tmp_path):
    sim = tmp_path / "h.parquet"
    runner.invoke(app, ["simulate", "healthy", "--out", str(sim)])
    r = runner.invoke(app, ["diagnose", str(sim), "--no-bootstrap", "--set", "scoring.fault.c=0.2"])
    assert r.exit_code == 0
    r = runner.invoke(app, ["diagnose", str(sim), "--no-bootstrap", "--set", "scoring.nope=1"])
    assert r.exit_code != 0


def test_cli_reon_csv_with_pack_baseline(tmp_path):
    r = runner.invoke(app, ["diagnose", str(FIX / "firebreak_hot_day_small.csv"), "--baseline", "pack", "--no-bootstrap",
                            "--out", str(tmp_path / "fb.json")])
    assert r.exit_code == 0, r.output
    rep = json.loads((tmp_path / "fb.json").read_text())
    assert rep["n_cells"] == 32 and rep["cells"][0]["cell_id"] == "C20"


def test_cli_calibrate(tmp_path):
    from celldx.sim.evaluate import make_tasks, run_many

    df = run_many(make_tasks("A", ["healthy", "weld", "cooling", "cold"], 6), n_jobs=1)
    tab = tmp_path / "lab.parquet"
    df.to_parquet(tab)
    r = runner.invoke(app, ["calibrate", str(tab), "--out", str(tmp_path / "cal.json"), "--cv-out", str(tmp_path / "cv.json"),
                            "--png-dir", str(tmp_path / "png")])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "png" / "reliability_fault.png").exists()
    sim = tmp_path / "w.parquet"
    runner.invoke(app, ["simulate", "weld", "--out", str(sim)])
    r = runner.invoke(app, ["diagnose", str(sim), "--no-bootstrap", "--calibration", str(tmp_path / "cal.json")])
    assert r.exit_code == 0 and json.loads(r.stdout)["probability_status"] == "calibrated"


def test_cli_thermal_step_skips_bad_frames():
    good = (FIX / "firebreak_frames.jsonl").read_text().splitlines()[:3]
    stdin = "\n".join([good[0], "{not json", good[1], json.dumps({"t": 1}), good[2]]) + "\n"
    r = runner.invoke(app, ["thermal", "step", "--remaining-wh", "3000"], input=stdin)
    assert r.exit_code == 0
    outs = [json.loads(x) for x in r.stdout.splitlines() if x.strip().startswith("{")]
    decisions = [o for o in outs if "commands" in o]
    assert len(decisions) == 3
    assert all(len(d["commands"]) == 32 for d in decisions)


def test_cli_thermal_replay_and_simulate(tmp_path):
    r = runner.invoke(app, ["thermal", "replay", str(FIX / "firebreak_frames.jsonl"), "--out", str(tmp_path / "dec.jsonl")])
    assert r.exit_code == 0
    summary = json.loads(r.stdout)
    assert summary["frames"] == 12 and sum(summary["cell_steps_by_action"].values()) == 12 * 32
    r = runner.invoke(app, ["thermal", "scenarios"])
    assert "exhaust_fault" in json.loads(r.stdout)["scenarios"]


def test_parquet_round_trip_of_simulated_frame(tmp_path, trial):
    df, _ = trial("A", "healthy", 0)
    p = tmp_path / "x.parquet"
    df.to_parquet(p)
    assert pd.read_parquet(p).shape == df.shape
