"""Hot-cell thermal controller: floor learning, string runtime, the three limits, charging rules,
the hard cutoff, hysteresis, determinism and closed-loop safety in the pack simulator."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from celldx.config import load_config
from celldx.thermal import CellReading, FloorEstimator, ThermalController, readings_from_frame, sim
from celldx.thermal.controller import series_runtime_ah

IDS = [f"C{i:02d}" for i in range(1, 33)]


def cells(temp=25.0, soc=0.6, v=3.3, current=0.0, r=1.0e-3, overrides=None):
    """32 identical Na-ion readings; ``overrides`` maps id -> dict of field changes."""
    out = []
    for cid in IDS:
        kw = dict(id=cid, temp_c=temp, voltage_v=v, current_a=current, soc=soc, soh=1.0, capacity_ah=100.0, r_ohm=r)
        kw.update((overrides or {}).get(cid, {}))
        out.append(CellReading(**kw))
    return out


def by_id(decision):
    return {c.id: c for c in decision.cells}


# ----------------------------------------------------------------------------- series runtime


@given(st.lists(st.floats(0.0, 100.0), min_size=1, max_size=40), st.integers(1, 40))
@settings(max_examples=300, deadline=None)
def test_series_runtime_is_the_water_filling_optimum(a, m):
    a = np.array(a)
    t = series_runtime_ah(a, m)
    if m > a.size:
        assert t == 0.0
        return
    # feasible: the cells still in can supply m cells' worth of throughput
    assert np.minimum(a, t).sum() >= m * t - 1e-6
    # maximal: a little more is infeasible (unless every cell is exhausted at t)
    t2 = t * 1.001 + 1e-6
    assert np.minimum(a, t2).sum() < m * t2 + 1e-6 or t2 > a.max()


def test_series_runtime_closed_forms():
    assert series_runtime_ah(np.full(32, 10.0), 32) == pytest.approx(10.0)
    assert series_runtime_ah(np.full(32, 10.0), 24) == pytest.approx(32 * 10.0 / 24)
    # a series string with no bypass would be limited by the weakest cell; bypass lets it drop out
    a = np.r_[np.full(31, 10.0), 2.0]
    assert series_runtime_ah(a, 32) == pytest.approx(2.0)
    # with 31 of 32 cells required the weak cell gives its 2 Ah and then drops out: t = (310 + 2) / 31
    assert series_runtime_ah(a, 31) == pytest.approx(312.0 / 31.0)
    assert series_runtime_ah(a, 0) == 0.0


# ----------------------------------------------------------------------------- floor learning


@pytest.mark.parametrize("ambient", [True, False])
def test_floor_estimator_learns_offset_and_tau(cfg, ambient):
    est = FloorEstimator(cfg)
    tau, offset, amb = 1800.0, 3.0, 24.0
    T = 34.0
    for k in range(240):          # 4 h idle, 1-minute samples, Newtonian cooling toward amb + offset
        t = 60.0 * k
        floor_true = amb + offset
        T = floor_true + (T - floor_true) * math.exp(-60.0 / tau)
        est.update(t, T + 0.01 * math.sin(k), True, amb if ambient else None)
    f, src = est.floor(amb if ambient else None, fallback_c=0.0)
    assert f == pytest.approx(amb + offset, abs=0.15)
    assert est.tau == pytest.approx(tau, rel=0.1)
    assert src.startswith("learned")


def test_floor_estimator_ignores_loaded_samples(cfg):
    est = FloorEstimator(cfg)
    for k in range(100):
        est.update(60.0 * k, 30.0 + 0.1 * k, False, 24.0)   # heating under load: not idle
    assert est.n_fits == 0
    assert est.floor(24.0, 0.0) == (24.0, "ambient sensor (not yet learned)")
    assert est.floor(None, 21.0) == (21.0, "fallback (not yet learned)")


# ----------------------------------------------------------------------------- basic contract


def test_rejects_changed_cell_set(cfg):
    c = ThermalController(cfg, IDS)
    with pytest.raises(ValueError):
        c.step(0.0, cells()[:-1], 1000.0)
    with pytest.raises(ValueError):
        ThermalController(cfg, IDS + ["C01"])


def test_cool_pack_keeps_every_cell_in(cfg):
    c = ThermalController(cfg, IDS)
    d = c.step(0.0, cells(temp=24.0), 2800.0, ambient_c=23.0, remaining_energy_wh=4000.0)
    assert d.n_in == 32 and not d.bypassed and d.checks["ok"]
    assert d.granted_w == pytest.approx(2800.0)
    assert json.dumps(d.to_dict())  # JSON-safe (no NaN)


def test_decision_to_dict_has_no_nan(cfg):
    c = ThermalController(cfg, IDS)
    d = c.step(0.0, cells(), 0.0)
    s = json.dumps(d.to_dict(), allow_nan=False)
    assert "NaN" not in s


def test_aging_rate_orders_hotter_fuller_weaker(cfg):
    c = ThermalController(cfg, IDS)
    base = c.aging_rate(30.0, 0.5, 1.0)
    assert c.aging_rate(25.0, 0.5, 1.0) == pytest.approx(1.0)
    assert c.aging_rate(35.0, 0.5, 1.0) > base
    assert c.aging_rate(30.0, 0.9, 1.0) > base
    assert c.aging_rate(30.0, 0.5, 0.8) > base


# ----------------------------------------------------------------------------- hot flag, ranking, release


def _warm_history(ctrl, t_end, temps_by_id, request=2800.0, amb=26.0, remaining=4000.0, current=28.0):
    """Feed a minute-level history ending at ``t_end`` (temps_by_id: id -> callable(t) -> T)."""
    d = None
    for t in np.arange(0.0, t_end + 1.0, 60.0):
        ov = {cid: {"temp_c": f(t), "current_a": current} for cid, f in temps_by_id.items()}
        d = ctrl.step(float(t), cells(temp=27.0, current=current, overrides=ov), request, ambient_c=amb,
                      remaining_energy_wh=remaining)
    return d


def test_predictive_flag_fires_before_the_band(cfg):
    c = ThermalController(cfg, IDS)
    # C07 heating at 0.4 K/min: flagged once T + 10 min of trend reaches 30 C, while T is still ~26 C
    first = None
    for t in np.arange(0.0, 900.0, 60.0):
        T = 24.0 + 0.4 * t / 60.0
        d = c.step(float(t), cells(temp=27.0, current=28.0, overrides={"C07": {"temp_c": T, "current_a": 28.0}}),
                   2800.0, ambient_c=26.0, remaining_energy_wh=1000.0)
        if by_id(d)["C07"].action == "BYPASS_HOT":
            first = by_id(d)["C07"]
            break
    assert first is not None
    assert first.temp_c < 30.0 - 3.0 and first.temp_pred_c >= 30.0
    assert by_id(d)["C01"].action == "IN"


def test_cooling_trend_never_moves_the_flag_earlier(cfg):
    c = ThermalController(cfg, IDS)
    d = _warm_history(c, 600.0, {"C07": lambda t: 29.5 - 0.2 * t / 60.0})
    assert by_id(d)["C07"].temp_pred_c == pytest.approx(by_id(d)["C07"].temp_c)
    assert by_id(d)["C07"].action == "IN"


def test_ranking_puts_hotter_fuller_weaker_first_when_slots_are_scarce(cfg):
    # 96 V inverter minimum with 32 x 3.3 V = 105.6 V: only a few cells can be off at once
    cfgv = cfg.with_overrides({"thermal.inverter_v_min": 96.0})
    c = ThermalController(cfgv, IDS)
    ov = {cid: {"temp_c": 33.0} for cid in IDS[:8]}
    ov["C03"] = {"temp_c": 36.0}
    ov["C05"] = {"temp_c": 33.0, "soh": 0.8}
    d = c.step(0.0, cells(temp=26.0, overrides=ov), 1500.0, ambient_c=25.0, remaining_energy_wh=None)
    assert d.rotation_active and "voltage_window" in d.binding
    assert d.string_v >= 96.0 - 1e-6
    assert 0 < len(d.bypassed) < 8
    assert "C03" in d.bypassed and "C05" in d.bypassed
    assert set(d.checks["waiting"]) == set(IDS[:8]) - set(d.bypassed)


def test_rotation_shares_off_time(cfg):
    """Closed-loop toy pack: six cells in 31 C air heat to ~34 C in the string and cool toward
    31 C when bypassed; the voltage window allows only a few off at once, so they take turns."""
    cfgv = cfg.with_overrides({"thermal.inverter_v_min": 99.0})
    c = ThermalController(cfgv, IDS)
    hot = IDS[:6]
    T = dict.fromkeys(IDS, 26.0)
    on = dict.fromkeys(IDS, True)
    off_minutes = dict.fromkeys(hot, 0)
    for k in range(480):
        for cid in IDS:
            floor = 31.0 if cid in hot else 26.0
            target = floor + (3.0 if on[cid] else 0.0)
            T[cid] += (target - T[cid]) * (1 - math.exp(-60.0 / 900.0))
        d = c.step(60.0 * k, cells(overrides={cid: {"temp_c": T[cid], "current_a": 15.0 if on[cid] else 0.0}
                                              for cid in IDS}), 1500.0, ambient_c=25.0, remaining_energy_wh=None)
        on = {x.id: x.in_string for x in d.cells}
        for cid in d.bypassed:
            if cid in off_minutes:
                off_minutes[cid] += 1
    assert all(v > 0 for v in off_minutes.values()), off_minutes
    assert max(off_minutes.values()) / min(off_minutes.values()) < 3.0


def test_release_when_cooled_below_band_margin(cfg):
    c = ThermalController(cfg, IDS)
    temps = [32.0, 32.0, 32.0] + [32.0 - 0.5 * k for k in range(1, 12)]
    seen = []
    for k, T in enumerate(temps):
        d = c.step(60.0 * k, cells(temp=27.0, current=28.0, overrides={"C07": {"temp_c": T}}), 2800.0,
                   ambient_c=26.0, remaining_energy_wh=1000.0)
        seen.append((T, by_id(d)["C07"].action, by_id(d)["C07"].reason))
    assert seen[0][1] == "BYPASS_HOT"
    released = next(x for x in seen if x[1] == "IN")
    assert released[0] <= 28.0 and released[2].startswith("released: cooled")


def test_early_flag_is_not_released_while_still_heating(cfg):
    """A cell flagged predictively at 27 C is not released the next minute just because it is
    below 28 C: the release uses the heating trend it had when it was switched off."""
    c = ThermalController(cfg, IDS)
    d = _warm_history(c, 600.0, {"C07": lambda t: 22.0 + 0.5 * t / 60.0}, remaining=1000.0)
    assert by_id(d)["C07"].action == "BYPASS_HOT"
    lit = ThermalController(cfg.with_overrides({"thermal.release_uses_heating_trend": False}), IDS)
    acts = []
    for t in np.arange(0.0, 601.0, 60.0):
        T = 22.0 + 0.5 * t / 60.0
        dd = lit.step(float(t), cells(temp=27.0, current=28.0, overrides={"C07": {"temp_c": T, "current_a": 28.0}}),
                      2800.0, ambient_c=26.0, remaining_energy_wh=1000.0)
        acts.append(by_id(dd)["C07"].in_string)
    assert True in acts[3:] and False in acts  # literal rule: off at 26 C then straight back in


def test_release_when_no_heat_left_to_shed_and_rearm(cfg):
    """A bypassed cell stuck above the band at its floor (hot air) is released, and is not
    re-flagged until it warms past its floor by the rearm margin."""
    c = ThermalController(cfg, IDS)
    # learn C07's floor: idle at 32 C in 32 C air -> floor offset +6 K vs a 26 C ambient sensor
    for k in range(60):
        T = 32.0 + 3.0 * math.exp(-k / 20.0)
        c.step(60.0 * k, cells(temp=27.0, overrides={"C07": {"temp_c": T}}), 0.0, ambient_c=26.0)
    assert c.floors()["C07"]["floor_c"] == pytest.approx(32.0, abs=0.3)
    seen_release = False
    for k in range(60, 120):
        d = c.step(60.0 * k, cells(temp=27.0, current=28.0, overrides={"C07": {"temp_c": 32.1, "current_a": 28.0}}),
                   2800.0, ambient_c=26.0, remaining_energy_wh=4000.0)
        cd = by_id(d)["C07"]
        seen_release |= cd.reason.startswith("released")
        assert cd.action != "BYPASS_HOT" or k < 61
    assert seen_release or by_id(d)["C07"].action == "IN"


# ----------------------------------------------------------------------------- the three limits


def test_power_and_voltage_window_limit(cfg):
    c = ThermalController(cfg.with_overrides({"thermal.inverter_v_min": 100.0}), IDS)
    ov = {cid: {"temp_c": 34.0} for cid in IDS[:10]}
    d = c.step(0.0, cells(temp=26.0, overrides=ov), 2800.0, ambient_c=25.0, remaining_energy_wh=None)
    assert d.string_v >= 100.0 - 1e-6
    assert d.granted_w == pytest.approx(2800.0)
    assert "voltage_window" in d.binding
    assert len(d.bypassed) < 10


def test_energy_limit(cfg):
    c = ThermalController(cfg, IDS)
    ov = {cid: {"temp_c": 34.0} for cid in IDS[:4]}
    # 32 cells x 100 Ah x ~(0.6 - 0.05) x ~3 V ~ 5.3 kWh deliverable; ask for (almost) all of it
    d_slack = c.step(0.0, cells(temp=26.0, overrides=ov), 2800.0, ambient_c=25.0, remaining_energy_wh=1000.0)
    assert len(d_slack.bypassed) == 4 and not d_slack.rotation_active
    c2 = ThermalController(cfg, IDS)
    tight = d_slack.checks["energy_in_wh"]
    d = c2.step(0.0, cells(temp=26.0, overrides=ov), 2800.0, ambient_c=25.0,
                remaining_energy_wh=0.995 * tight / float(cfg.thermal.energy_reserve_factor) * (32 / 28))
    assert "energy" in d.binding
    assert d.checks["energy_in_wh"] >= d.checks["energy_required_wh"]


def test_band_protection_limit(cfg):
    """Healthy cells at 29.9 C carrying current: taking hot cells out raises the string current
    and would push them over the band, so the hot cells wait."""
    c = ThermalController(cfg, IDS)
    ov = {cid: {"temp_c": 34.0, "current_a": 90.0} for cid in IDS[:6]}
    d = c.step(0.0, cells(temp=29.9, current=90.0, r=3e-3, overrides=ov), 9000.0, ambient_c=25.0,
               remaining_energy_wh=100.0)
    assert "band_protection" in d.binding
    off = cfg.with_overrides({"thermal.check_band_protection": False})
    d2 = ThermalController(off, IDS).step(0.0, cells(temp=29.9, current=90.0, r=3e-3, overrides=ov), 9000.0,
                                          ambient_c=25.0, remaining_energy_wh=100.0)
    assert len(d2.bypassed) > len(d.bypassed)


# ----------------------------------------------------------------------------- charging rules and cutoff


@pytest.mark.parametrize("request_w", [-3000.0, 0.0, 3000.0, 20000.0])
def test_hard_cutoff_always_switches_off(cfg, request_w):
    c = ThermalController(cfg, IDS)
    d = c.step(0.0, cells(overrides={"C09": {"temp_c": 57.0}}), request_w, ambient_c=25.0, remaining_energy_wh=1e6)
    assert by_id(d)["C09"].action == "BYPASS_CUTOFF"
    assert not by_id(d)["C09"].in_string


def test_cutoff_is_latched_until_release_temperature(cfg):
    c = ThermalController(cfg, IDS)
    seq = [58.0, 55.0, 51.0, 50.5, 49.9]
    acts = []
    for k, T in enumerate(seq):
        d = c.step(60.0 * k, cells(overrides={"C09": {"temp_c": T}}), 2000.0, ambient_c=25.0)
        acts.append(by_id(d)["C09"].action)
    assert acts[:4] == ["BYPASS_CUTOFF"] * 4
    assert acts[4] != "BYPASS_CUTOFF"


def test_cutoff_derates_power_instead_of_keeping_the_cell_in(cfg):
    c = ThermalController(cfg.with_overrides({"thermal.inverter_v_min": 104.0}), IDS)
    d = c.step(0.0, cells(overrides={"C09": {"temp_c": 60.0}}), 3000.0, ambient_c=25.0)
    assert by_id(d)["C09"].action == "BYPASS_CUTOFF"
    assert d.checks["derated"] and d.granted_w < 3000.0


def test_no_charge_below_min_or_above_max_temperature(cfg):
    c = ThermalController(cfg, IDS)
    ov = {"C02": {"temp_c": 14.0}, "C03": {"temp_c": 46.0}, "C04": {"temp_c": 40.0, "charge_max_c": 38.0}}
    d = c.step(0.0, cells(temp=25.0, overrides=ov), -1700.0, ambient_c=25.0)
    b = by_id(d)
    for cid in ("C02", "C03", "C04"):
        assert b[cid].action == "BYPASS_CHARGE_RULE", cid
    # the same cells discharge normally
    d2 = ThermalController(cfg, IDS).step(0.0, cells(temp=25.0, overrides=ov), 1700.0, ambient_c=25.0)
    assert by_id(d2)["C02"].in_string


def test_hot_air_charge_caps(cfg):
    c = ThermalController(cfg, IDS)
    assert c.soc_cap(25.0) == pytest.approx(0.95)
    assert c.soc_cap(30.0) == pytest.approx(0.80)
    assert c.soc_cap(36.0) == pytest.approx(0.70)


def test_cap_hysteresis_on_the_floor(cfg):
    c = ThermalController(cfg, IDS)
    active: set[float] = set()
    seq = [30.1, 29.8, 29.6, 29.4, 30.0]
    caps = [c.soc_cap(f, active) for f in seq]
    assert caps == [0.80, 0.80, 0.80, 0.95, 0.80]


def test_hot_air_cell_stops_charging_at_its_cap(cfg):
    c = ThermalController(cfg, IDS)
    # learn C07's floor at 31 C first (idle), then charge
    for k in range(40):
        c.step(60.0 * k, cells(temp=25.0, overrides={"C07": {"temp_c": 31.0 + 2.0 * math.exp(-k / 10)}}), 0.0,
               ambient_c=25.0)
    d = c.step(2400.0, cells(temp=25.0, soc=0.6, overrides={"C07": {"temp_c": 31.0, "soc": 0.81}}), -1700.0,
               ambient_c=25.0)
    assert by_id(d)["C07"].soc_cap == pytest.approx(0.80)
    assert by_id(d)["C07"].action == "BYPASS_CHARGE_RULE"


def test_empty_cell_bypassed_on_discharge(cfg):
    c = ThermalController(cfg, IDS)
    d = c.step(0.0, cells(soc=0.5, overrides={"C11": {"soc": 0.04}}), 2000.0, ambient_c=25.0)
    assert by_id(d)["C11"].action == "BYPASS_SOC_LIMIT"


def test_rule_bypass_is_held_against_a_flipping_request(cfg):
    c = ThermalController(cfg, IDS)
    acts = []
    for k in range(6):
        req = -1500.0 if k % 2 == 0 else 1500.0
        d = c.step(60.0 * k, cells(soc=0.6, overrides={"C02": {"temp_c": 13.0}}), req, ambient_c=20.0)
        acts.append(by_id(d)["C02"].in_string)
    assert acts == [False] * 6


def test_frame_parsing():
    fr = {"t": 120, "ambient_C": 24.5, "request_W": -900,
          "cells": [{"id": "C01", "V": 3.3, "I": -9.0, "T": 27.1, "soc": 0.5, "soh": 0.97, "R_mOhm": 1.1}]}
    t, cs, amb, req = readings_from_frame(fr)
    assert (t, amb, req) == (120.0, 24.5, -900.0)
    assert cs[0].r_ohm == pytest.approx(0.0011)


# ----------------------------------------------------------------------------- closed loop


def test_simulation_is_deterministic():
    a = sim.run("hot_pocket", "celldx", seed=2)
    b = sim.run("hot_pocket", "celldx", seed=2)
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


@pytest.mark.parametrize("scenario", ["exhaust_fault", "cold_morning"])
def test_closed_loop_safety_rules_hold(scenario):
    """Ground truth from the plant, not the controller's own view: no cell carries current at or
    above the 57 C cutoff, none charges below 15 C or above its charge limit or above its
    effective cap, and the inverter never leaves its voltage window."""
    r = sim.run(scenario, "celldx", seed=0)
    v = r["violations_truth"]
    assert v["carrying_current_at_or_above_cutoff_cell_min"] == 0
    assert v["charging_below_min_temp_cell_min"] == 0
    assert v["charging_above_charge_limit_cell_min"] == 0
    assert v["charging_above_effective_cap_cell_min"] == 0
    assert v["inverter_out_of_window_min"] == 0
    assert r["energy"]["served_discharge_frac"] == pytest.approx(1.0)


def test_fixed_string_trips_in_the_exhaust_fault():
    r = sim.run("exhaust_fault", "fixed", seed=0)
    assert r["energy"]["discharge_shortfall_min"] > 0 or r["thermal"]["max_temp_c"] >= 57.0


def test_floor_learning_beats_the_room_sensor():
    r = sim.run("hot_pocket", "celldx", seed=0)
    ctl = r["controller"]
    assert ctl["floor_mae_vs_idle_equilibrium_k"] < 0.5 * ctl["room_sensor_mae_vs_idle_equilibrium_k"]


def test_cap_hysteresis_prevents_chatter():
    r = sim.run("hot_pocket", "celldx", seed=0)
    assert r["switching"]["chatter_events_lt5min"] <= 5


def test_cli_strategy_names_are_all_runnable():
    cfg = load_config()
    for name in sim.STRATEGIES:
        sim.make_strategy(name, IDS, cfg)
    with pytest.raises((KeyError, ValueError)):
        sim.make_strategy("nope", IDS, cfg)
