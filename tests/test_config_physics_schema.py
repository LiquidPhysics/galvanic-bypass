"""Configuration, physics helpers and input validation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from celldx import physics
from celldx.config import ConfigError, load_config
from celldx.schema import ALL_COLUMNS, Observation, records_to_frame, validate

# ----------------------------------------------------------------------------- config


def test_defaults_load_and_are_immutable(cfg):
    assert cfg.profile == "prototype"
    assert cfg.thermal.band_edge_c == 30.0
    assert cfg.thermal.hard_cutoff_c == 57.0
    with pytest.raises(AttributeError):
        cfg.profile = "v3"


def test_unknown_override_raises():
    with pytest.raises(ConfigError):
        load_config(overrides={"scoring.not_a_key": 1})
    with pytest.raises(ConfigError):
        load_config(overrides={"nosuchsection.x": 1})


def test_unknown_key_in_user_file_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("thermal:\n  band_edge_cc: 31\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_user_file_and_override_merge(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("thermal:\n  band_edge_c: 31.5\n")
    c = load_config(p, overrides={"thermal.hard_cutoff_c": 55.0})
    assert c.thermal.band_edge_c == 31.5
    assert c.thermal.hard_cutoff_c == 55.0
    assert c.thermal.release_margin_k == load_config().thermal.release_margin_k


def test_v3_profile_switches_every_fix_on(cfg_v3):
    fixes = cfg_v3.fixes.to_dict()
    assert all(v for v in fixes.values() if isinstance(v, bool))
    assert not any(v for v in load_config().fixes.to_dict().values() if isinstance(v, bool))


def test_unknown_profile_raises():
    with pytest.raises(ConfigError):
        load_config(profile="v9")


def test_hash_is_stable_and_sensitive(cfg):
    assert cfg.hash() == load_config().hash()
    assert cfg.hash() != cfg.with_overrides({"thermal.band_edge_c": 31.0}).hash()
    assert len(cfg.hash()) == 16


# ----------------------------------------------------------------------------- physics


def test_arrhenius_is_one_at_25c():
    assert physics.arrhenius_factor(25.0, 30e3) == pytest.approx(1.0)
    assert physics.arrhenius_factor(0.0, 30e3) > 1.0   # resistance rises when cold
    assert physics.arrhenius_factor(45.0, 30e3) < 1.0


@given(st.floats(-30, 70), st.floats(5e3, 80e3), st.floats(1e-4, 1.0))
@settings(max_examples=200, deadline=None)
def test_r25_round_trip(temp, ea, r):
    r25 = physics.to_r25(r, temp, ea)
    assert float(physics.from_r25(r25, temp, ea)) == pytest.approx(r, rel=1e-9)


@given(st.floats(-1, 1), st.floats(-1, 1), st.floats(0.01, 2))
@settings(max_examples=200, deadline=None)
def test_sigmoid_monotone_and_centred(a, b, w):
    lo, hi = sorted((a, b))
    assert float(physics.sig(lo, 0.0, w)) <= float(physics.sig(hi, 0.0, w)) + 1e-12
    assert float(physics.sig(0.3, 0.3, w)) == pytest.approx(0.5)
    v = float(physics.sig(a, 0.0, w))
    assert 0.0 <= v <= 1.0


def test_sigmoid_nan_propagates_and_scoring_maps_it_to_no_evidence(cfg):
    from celldx.diagnosis.scoring import _s

    assert math.isnan(float(physics.sig(float("nan"), 0.1, 0.03)))
    # the prototype treats a missing feature as "no evidence" (score 0)
    assert float(_s(float("nan"), cfg.scoring.fault)) == 0.0


def test_soc_factor_interpolates():
    grid, fac = [0.0, 0.5, 1.0], [2.0, 1.0, 1.5]
    assert float(physics.soc_factor(0.5, grid, fac)) == pytest.approx(1.0)
    assert float(physics.soc_factor(0.25, grid, fac)) == pytest.approx(1.5)


def test_robust_mad_ignores_outliers():
    x = np.r_[np.random.default_rng(0).normal(0, 1, 1000), [1e6] * 10]
    # unscaled MAD: 0.6745 sigma for a normal sample, unaffected by 1% gross outliers
    assert physics.robust_mad(x) == pytest.approx(0.6745, rel=0.1)
    assert math.isnan(physics.robust_mad([]))
    assert physics.robust_spread([1.0]) == 0.0


# ----------------------------------------------------------------------------- schema / validation


def _frame(n=40, **cols):
    base = {"t": np.arange(n, dtype=float), "cell_id": "C1", "resistance_ohm": 0.05 + np.zeros(n),
            "temp_cell_c": 25 + np.zeros(n), "soc": 0.5 + np.zeros(n), "soh": 1.0 + np.zeros(n)}
    base.update(cols)
    return pd.DataFrame(base)


def test_validation_never_drops_rows_and_masks_out_of_range(cfg):
    df = _frame(temp_cell_c=np.r_[25 + np.zeros(39), [500.0]], soc=np.r_[0.5 + np.zeros(39), [1.7]])
    clean, rep = validate(df, cfg)
    assert len(clean) == len(df)
    assert np.isnan(clean["temp_cell_c"].iloc[-1])
    assert np.isnan(clean["soc"].iloc[-1])
    assert rep.masked_values["temp_cell_c"] == 1
    assert any(i.code == "sensor_range" for i in rep.issues)


def test_validation_sorts_and_reports_backwards_time(cfg):
    df = _frame().iloc[::-1].reset_index(drop=True)
    clean, rep = validate(df, cfg)
    assert np.all(np.diff(clean["t"].to_numpy()) > 0)
    assert any(i.code == "non_monotonic_time" for i in rep.issues)


def test_validation_flags_resistance_spike_but_keeps_row(cfg):
    r = 0.05 + 0.0005 * np.random.default_rng(1).standard_normal(60)
    r[30] = 0.5
    clean, rep = validate(_frame(60, resistance_ohm=r), cfg)
    assert not clean["qc_ok"].iloc[30]
    assert "spike:resistance_ohm" in clean["qc_flags"].iloc[30]
    assert clean["qc_ok"].drop(index=30).all()
    assert clean["resistance_ohm"].iloc[30] == 0.5


def test_cold_day_is_not_a_spike(cfg):
    temp = 25 + np.zeros(60)
    temp[30] = 0.0
    r = 0.05 * physics.arrhenius_factor(temp, 30e3)
    clean, _ = validate(_frame(60, resistance_ohm=r, temp_cell_c=temp), cfg)
    assert clean["qc_ok"].all()


@pytest.mark.parametrize("col,val,code", [("temp_cell_c", 298.0, "unit_temperature"), ("soc", 50.0, "unit_soc"),
                                          ("soh", 95.0, "unit_soh"), ("resistance_ohm", 50.0, "unit_resistance"),
                                          ("ocv_mv", 3300.0, "unit_ocv")])
def test_unit_heuristics(cfg, col, val, code):
    clean, rep = validate(_frame(**{col: val + np.zeros(40)}), cfg)
    assert any(i.code == code for i in rep.issues)


def test_missing_time_is_flagged(cfg):
    t = np.arange(40, dtype=float)
    t[5] = np.nan
    clean, rep = validate(_frame(t=t), cfg)
    assert any(i.code == "missing_time" for i in rep.issues)
    assert (~clean["qc_ok"]).sum() >= 1


def test_no_resistance_error(cfg):
    df = _frame().drop(columns=["resistance_ohm"])
    _, rep = validate(df, cfg)
    assert any(i.code == "no_resistance" for i in rep.issues)
    assert not rep.ok


def test_non_numeric_values_masked(cfg):
    df = _frame()
    df["soc"] = df["soc"].astype(object)
    df.loc[3, "soc"] = "n/a"
    clean, rep = validate(df, cfg)
    assert np.isnan(clean["soc"].iloc[3])
    assert any(i.code == "non_numeric" for i in rep.issues)


def test_observation_model_rejects_unknown_fields():
    with pytest.raises(ValueError):
        Observation(t=0, cell_id="C1", temperature=25)
    f = records_to_frame([{"t": 0, "cell_id": "C1", "resistance_ohm": 0.05}])
    assert list(f.columns) == ALL_COLUMNS
    assert f["pack_id"].iloc[0] == "pack0"
    assert pd.isna(f["soc"].iloc[0])
