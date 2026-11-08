"""Baseline fitting (Ea, R0, provenance, trust, compatibility) and feature primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from celldx.baseline.fit import check_compatibility, fit_ea, fit_own_baseline, literature_baseline
from celldx.baseline.model import Baseline
from celldx.features.core import sqrt_growth, step_detect, window_len
from celldx.physics import arrhenius_factor
from celldx.sim.evaluate import population_baseline


def _multi_temp(ea=35e3, n_cells=4, n=200, spread=30.0, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    cells, temps, rs = [], [], []
    for k in range(n_cells):
        r0 = rng.uniform(0.04, 0.06)
        T = rng.uniform(10, 10 + spread, n)
        R = r0 * arrhenius_factor(T, ea) * (1 + noise * rng.standard_normal(n))
        cells += [f"c{k}"] * n
        temps.append(T)
        rs.append(R)
    return np.array(cells), np.concatenate(temps), np.concatenate(rs)


def test_fit_ea_recovers_known_activation_energy(cfg):
    c, T, R = _multi_temp(ea=35e3)
    f = fit_ea(c, T, R, cfg)
    assert f.source == "fitted_within_cell"
    assert f.value == pytest.approx(35e3, rel=0.02)
    assert f.ci[0] < f.value < f.ci[1]


def test_fit_ea_interval_coverage_is_close_to_nominal(cfg):
    # cluster bootstrap over 4 cells: nominal 95%; measured 92.5% over 40 seeds (see validation)
    hits = 0
    for s in range(20):
        c, T, R = _multi_temp(ea=35e3, seed=s)
        f = fit_ea(c, T, R, cfg, seed=s)
        hits += f.ci[0] <= 35e3 <= f.ci[1]
    assert hits >= 16


def test_fit_ea_refuses_without_temperature_spread(cfg):
    c, T, R = _multi_temp(spread=3.0)
    f = fit_ea(c, T, R, cfg)
    assert f.source == "literature_default"
    assert f.value == cfg.physics.ea_literature_j_per_mol
    assert "cannot support" in f.reason


def test_fit_ea_between_cell_is_labelled_confounded(cfg):
    # every cell at its own fixed temperature: only a pooled (between-cell) slope exists
    rng = np.random.default_rng(3)
    temps = np.repeat([5.0, 15.0, 25.0, 35.0, 45.0], 30)
    cells = np.repeat([f"c{k}" for k in range(5)], 30)
    R = 0.05 * arrhenius_factor(temps, 30e3) * (1 + 0.01 * rng.standard_normal(temps.size))
    f = fit_ea(cells, temps, R, cfg)
    assert f.source == "fitted_between_cell"
    assert "confounded" in f.reason
    assert f.value == pytest.approx(30e3, rel=0.1)


def test_fit_ea_rejects_implausible_value(cfg):
    c, T, R = _multi_temp(ea=-20e3)  # resistance falling when cold: outside the plausible range
    assert fit_ea(c, T, R, cfg).source == "literature_default"


def _own_samples(n=400, ea=30e3, seed=0, spread=12.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    temp = 20 + spread / 2 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 0.3, n)
    r = 0.05 * arrhenius_factor(temp, ea) * (1 + 0.01 * rng.standard_normal(n))
    return pd.DataFrame({"t": t, "r_total": r, "r_ohmic": np.nan, "r_slow": np.nan, "temp_c": temp,
                         "soh": 1.0, "temp_pack_c": temp})


def test_own_baseline_r0_and_provenance(cfg):
    bl = fit_own_baseline(_own_samples(), cfg, cell_id="X", chemistry="na_ion", resistance_kind="sim_daily")
    assert bl.kind == "own"
    assert bl.r0_25c_ohm.value == pytest.approx(0.05, rel=0.03)
    assert bl.provenance.dataset == "own early life of X"
    assert bl.trust(1.0, 0.85, 0.7, 0.95)[0] == 1.0


def test_own_baseline_widens_ea_window_when_early_life_is_flat(cfg):
    # only the second half of the record has seasonal spread
    s = _own_samples(n=730, spread=14.0)
    s.loc[:300, "temp_c"] = 20.0 + np.random.default_rng(1).normal(0, 0.2, 301)
    s.loc[:300, "r_total"] = 0.05 * arrhenius_factor(s.loc[:300, "temp_c"].to_numpy(), 30e3)
    bl = fit_own_baseline(s, cfg, cell_id="Y")
    assert bl.ea_j_per_mol.source in ("fitted_own", "literature_default")
    if bl.ea_j_per_mol.source == "fitted_own":
        assert "widened" in (bl.ea_j_per_mol.note or "")


def test_baseline_save_load_round_trip(tmp_path, cfg):
    bl = fit_own_baseline(_own_samples(), cfg, cell_id="X")
    for ext in ("json", "yaml"):
        p = bl.save(tmp_path / f"b.{ext}")
        assert Baseline.load(p) == bl


def test_trust_levels(cfg):
    pop = population_baseline(0.05, 0.0, 30e3, "C00")
    assert pop.trust(1.0, 0.85, 0.7, 0.95)[0] == 0.85  # R0 from a population fit
    lit = literature_baseline(cfg, 0.05, chemistry="na_ion", form_factor="prismatic", nominal_capacity_ah=100,
                              resistance_kind="dcir_onset", note="datasheet")
    t, why = lit.trust(1.0, 0.85, 0.7, 0.95)
    assert t == 0.7 and "literature" in why[0]
    pop_lit_ea = pop.model_copy(update={"ea_j_per_mol": pop.ea_j_per_mol.model_copy(update={"source": "literature_default"})})
    assert pop_lit_ea.trust(1.0, 0.85, 0.7, 0.95)[0] == pytest.approx(0.85 * 0.95)


def test_compatibility_warnings(cfg):
    bl = population_baseline(0.05, 0.0, 30e3, "C00", chemistry="nmc", form_factor="18650", capacity_ah=2.5)
    w = check_compatibility(bl, cfg, chemistry="na_ion", form_factor="prismatic", capacity_ah=100,
                            r_typical_ohm=0.001, resistance_kind="pulse_1s")
    joined = " ".join(w)
    for key in ("CHEMISTRY MISMATCH", "FORM-FACTOR MISMATCH", "CAPACITY MISMATCH", "RESISTANCE-LEVEL MISMATCH",
                "RESISTANCE-DEFINITION MISMATCH"):
        assert key in joined
    assert check_compatibility(bl, cfg, chemistry="nmc", form_factor="18650", capacity_ah=2.5,
                               r_typical_ohm=0.05, resistance_kind="sim_daily") == []


# ----------------------------------------------------------------------------- features


def test_window_len():
    assert window_len(100, 0.03, 5) == 5
    assert window_len(1000, 0.03, 5) == 30
    assert window_len(3, 0.5, 5) == 3


def test_step_detect_finds_injected_step():
    rng = np.random.default_rng(0)
    x = 0.05 * (1 + 0.01 * rng.standard_normal(400))
    x[250:] *= 1.3
    r = step_detect(x, 20, 1.25, 0.0)
    assert r["rel"] == pytest.approx(0.3, abs=0.03)
    assert abs(r["index"] - 250) <= 3
    assert r["z"] > 20
    flat = step_detect(0.05 * (1 + 0.01 * rng.standard_normal(400)), 20, 1.25, 0.0)
    assert flat["z"] < 5


def test_step_detect_cusum_locates_change():
    rng = np.random.default_rng(1)
    x = np.r_[np.zeros(200), np.ones(200)] + 0.05 * rng.standard_normal(400) + 10
    r = step_detect(x, 20, 1.25, 0.0, method="cusum")
    assert abs(r["index"] - 200) <= 3


def test_step_detect_too_short_returns_nan():
    assert np.isnan(step_detect(np.ones(10), 20, 1.25, 0.0)["z"])


def test_sqrt_growth_fit():
    t = np.arange(500, dtype=float)
    y = 0.05 + 0.001 * np.sqrt(t)
    g = sqrt_growth(t, y, 10)
    assert g["b"] == pytest.approx(0.001, rel=1e-6)
    assert g["r2"] == pytest.approx(1.0)
    assert np.isnan(sqrt_growth(t[:3], y[:3], 10)["r2"])
