"""Per-cell features computed on the sample table (one row per day or cycle).

Every function is pure; units are stated in each docstring. The feature dictionary holds
NaN for anything that cannot be computed, and ``availability`` says why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..baseline.model import Baseline
from ..config import Config
from ..physics import KELVIN, arrhenius_factor, robust_spread, soc_correct, to_r25

SAMPLE_COLUMNS = [
    "t", "r_total", "r_ohmic", "r_slow", "temp_c", "temp_pack_c", "soc", "soh", "ocv_mv", "current_rms",
    "cold_charge", "ah_throughput", "cycle_count", "noise_mv", "bms_r",
]


def cycle_cell_to_samples(g: pd.DataFrame) -> pd.DataFrame:
    """Cycle mode: the validated rows already are samples.

    ``r_total`` = ``resistance_ohm`` if given, else ``r_ohmic + r_slow`` (EIS Re + Rct).
    ``current_rms`` = |current_a| of the row (cycle-level representative current).
    """
    g = g[g["qc_ok"]] if "qc_ok" in g else g
    r_total = g["resistance_ohm"].to_numpy(float).copy()
    comp = g["r_ohmic_ohm"].to_numpy(float) + g["r_slow_ohm"].to_numpy(float)
    r_total = np.where(np.isfinite(r_total), r_total, comp)
    out = pd.DataFrame({
        "t": g["t"].to_numpy(float), "r_total": r_total, "r_ohmic": g["r_ohmic_ohm"].to_numpy(float),
        "r_slow": g["r_slow_ohm"].to_numpy(float), "temp_c": g["temp_cell_c"].to_numpy(float),
        "temp_pack_c": g["temp_pack_c"].to_numpy(float), "soc": g["soc"].to_numpy(float),
        "soh": g["soh"].to_numpy(float), "ocv_mv": g["ocv_mv"].to_numpy(float),
        "current_rms": np.abs(g["current_a"].to_numpy(float)), "cold_charge": g["cold_charge"].to_numpy(float),
        "ah_throughput": g["ah_throughput"].to_numpy(float), "cycle_count": g["cycle_count"].to_numpy(float),
        "noise_mv": np.nan, "bms_r": np.nan,
    })
    return out


def window_len(n: int, frac: float, nmin: int) -> int:
    """``max(nmin, round(frac * n))`` capped at n (samples)."""
    return int(min(n, max(nmin, int(round(frac * n)))))


@dataclass
class CellSeries:
    """Arrays a cell's features are computed from (all aligned, sorted by t, finite R and T)."""

    t: np.ndarray
    r_raw: np.ndarray        # ohm, as measured
    r25: np.ndarray          # ohm, temperature + SOC corrected (components corrected separately)
    r25_tot_ea: np.ndarray   # ohm, total R corrected with Ea_total only (for b_rel)
    r_ohm25: np.ndarray      # ohm (NaN if unknown)
    r_slow25: np.ndarray     # ohm (NaN if unknown)
    temp: np.ndarray         # C
    temp_pack: np.ndarray    # C
    soc: np.ndarray
    soh: np.ndarray
    ocv: np.ndarray          # mV deviation
    i_rms: np.ndarray        # A
    cold: np.ndarray         # 0/1
    ah: np.ndarray           # Ah
    noise: np.ndarray        # mV
    bms_r: np.ndarray        # ohm
    win_raw: int
    win_corr: int
    r0: float
    r0_ohm: float
    r0_slow: float


@dataclass
class CellFeatures:
    values: dict[str, float] = field(default_factory=dict)
    flags: dict[str, bool] = field(default_factory=dict)
    availability: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    series: CellSeries | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": {k: (None if (isinstance(v, float) and not math.isfinite(v)) else v) for k, v in self.values.items()},
            "flags": dict(self.flags),
            "availability": dict(self.availability),
            "notes": list(self.notes),
        }


def build_series(s: pd.DataFrame, bl: Baseline, cfg: Config) -> CellSeries:
    """Sort, drop rows without R or T (reported by the caller) and apply corrections.

    Corrections (section 6.2): ``R25 = R / exp(Ea/Rg (1/T - 1/T25))`` then
    ``R_corr = R25 * Rmap(soc_ref) / Rmap(soc)``. When both components exist they are
    corrected with their own Ea and summed; otherwise total R uses Ea_total.
    """
    s = s.sort_values("t", kind="stable")
    s = s[np.isfinite(s["r_total"].to_numpy(float)) & np.isfinite(s["temp_c"].to_numpy(float))]
    rg = float(cfg.physics.rg_j_per_mol_k)
    t25 = float(cfg.physics.t25_k)
    soc_ref = float(bl.soc_ref)
    ea = float(bl.ea_j_per_mol.value or cfg.physics.ea_literature_j_per_mol)
    ea_o = float(bl.ea_ohmic_j_per_mol.value or ea)
    ea_s = float(bl.ea_slow_j_per_mol.value or ea)
    T = s["temp_c"].to_numpy(float)
    soc = s["soc"].to_numpy(float)
    grid_x, grid_y = bl.r_soc_map.x, bl.r_soc_map.y
    r = s["r_total"].to_numpy(float)
    ro = s["r_ohmic"].to_numpy(float)
    rs = s["r_slow"].to_numpy(float)
    r25_tot = soc_correct(to_r25(r, T, ea, rg, t25), soc, grid_x, grid_y, soc_ref)
    ro25 = soc_correct(to_r25(ro, T, ea_o, rg, t25), soc, grid_x, grid_y, soc_ref)
    rs25 = soc_correct(to_r25(rs, T, ea_s, rg, t25), soc, grid_x, grid_y, soc_ref)
    comp_ok = np.isfinite(ro25) & np.isfinite(rs25)
    r25 = np.where(comp_ok, ro25 + rs25, r25_tot)
    n = len(s)
    w = cfg.windows
    return CellSeries(
        t=s["t"].to_numpy(float), r_raw=r, r25=r25, r25_tot_ea=r25_tot, r_ohm25=ro25, r_slow25=rs25, temp=T,
        temp_pack=s["temp_pack_c"].to_numpy(float), soc=soc, soh=s["soh"].to_numpy(float),
        ocv=s["ocv_mv"].to_numpy(float), i_rms=s["current_rms"].to_numpy(float),
        cold=np.nan_to_num(s["cold_charge"].to_numpy(float), nan=0.0), ah=s["ah_throughput"].to_numpy(float),
        noise=s["noise_mv"].to_numpy(float), bms_r=s["bms_r"].to_numpy(float),
        win_raw=window_len(n, float(w.raw_recent_frac), int(w.raw_recent_min)),
        win_corr=window_len(n, float(w.corr_recent_frac), int(w.corr_recent_min)),
        r0=float(bl.r0_25c_ohm.value) if bl.r0_25c_ohm.value else float("nan"),
        r0_ohm=float(bl.r0_ohmic_ohm.value) if bl.r0_ohmic_ohm.value else float("nan"),
        r0_slow=float(bl.r0_slow_ohm.value) if bl.r0_slow_ohm.value else float("nan"),
    )


# --------------------------------------------------------------------------- detectors
def diff_noise(x: np.ndarray) -> float:
    """Trend-proof noise of one sample: ``1.4826 * MAD(first differences) / sqrt(2)``."""
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    d = np.diff(x)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(2.0))


def resid_noise(x: np.ndarray, win: int) -> float:
    """Noise as 1.4826 * MAD of the residual from a centred rolling median (window ``win``).

    Trend-proof like the diff estimator, but it sees AR(1)-correlated noise that first
    differences cancel."""
    x = x[np.isfinite(x)]
    if x.size < 2 * win + 1:
        return float("nan")
    rm = pd.Series(x).rolling(win, center=True, min_periods=max(2, win // 2)).median().to_numpy()
    r = x - rm
    r = r[np.isfinite(r)]
    return float(1.4826 * np.median(np.abs(r - np.median(r)))) if r.size else float("nan")


def step_detect(x: np.ndarray, win: int, noise_factor: float, floor_rel: float, method: str = "window_median",
                robust_noise: bool = False) -> dict[str, float]:
    """Largest step between medians of a window before and after each point.

    Returns ``jump`` (signed, units of x), ``rel`` (jump / median before), ``z`` =
    ``|jump| / (noise_factor * noise * sqrt(2/win))`` with the diff-based noise, and the
    ``index`` of the first sample after the step. ``method='cusum'`` locates the change
    point with a two-sided CUSUM on the standardised series, then measures the jump the
    same way.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    nan = {"jump": float("nan"), "rel": float("nan"), "z": float("nan"), "index": float("nan"), "noise": float("nan")}
    if n < 2 * win or win < 1:
        return nan
    noise = diff_noise(x)
    if robust_noise:
        rn = resid_noise(x, win)
        if np.isfinite(rn):
            noise = max(noise if np.isfinite(noise) else 0.0, rn)
    med_all = float(np.median(np.abs(x))) if n else 0.0
    noise = max(noise if np.isfinite(noise) else 0.0, floor_rel * med_all, 1e-15)
    from numpy.lib.stride_tricks import sliding_window_view

    M = np.median(sliding_window_view(x, win), axis=1)  # M[j] = median(x[j:j+win])
    ks = np.arange(win, n - win + 1)
    d = M[ks] - M[ks - win]
    if method == "cusum":
        mu = np.median(x[:win])
        c = np.cumsum((x - mu) / noise)
        k = int(np.clip(np.argmax(np.abs(c - np.linspace(0, c[-1], n))), win, n - win))
        j = int(k - win)
    else:
        j = int(np.argmax(np.abs(d)))
        k = int(ks[j])
    jump = float(M[k] - M[k - win])
    before = float(M[k - win])
    rel = jump / before if before != 0 else float("nan")
    z = abs(jump) / (noise_factor * noise * math.sqrt(2.0 / win))
    return {"jump": jump, "rel": rel, "z": float(z), "index": float(k), "noise": noise}


def sqrt_growth(t: np.ndarray, y: np.ndarray, nmin: int) -> dict[str, float]:
    """Fit ``y = a + b * sqrt(t - t0)``; returns a, b (units of y per sqrt(unit of t)) and R^2."""
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if t.size < nmin:
        return {"a": float("nan"), "b": float("nan"), "r2": float("nan")}
    x = np.sqrt(np.maximum(t - t.min(), 0.0))
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"a": float(coef[0]), "b": float(coef[1]), "r2": float(r2)}


def temperature_independent_share(cs: CellSeries, bl: Baseline, cfg: Config) -> dict[str, float]:
    """Temperature-independent resistance share ``b_rel`` and apparent Ea ratio.

    Over the last ``tind_window`` samples, if temperature spread >= ``tind_min_spread_k`` and
    n >= ``tind_min_samples``: regress ``R25 = a + c1 t + c2 inv`` with
    ``inv = 1/arrhenius(T, Ea_baseline)``. Since ``R = arr X + B`` gives ``R25 = X + B inv``,
    c2 estimates B (ohm), the resistance that does not follow the temperature curve.
    ``b_rel = c2 * median(inv) / R0``. Apparent Ea ratio: slope of ln R on [1, t, 1/T]
    times Rg divided by Ea_baseline.
    """
    f = cfg.features
    w = int(f.tind_window)
    t, R25, T, R = cs.t[-w:], cs.r25_tot_ea[-w:], cs.temp[-w:], cs.r_raw[-w:]
    spread = robust_spread(T, *f.temp_spread_quantiles)
    out = {"b_rel": float("nan"), "c2_ohm": float("nan"), "ea_apparent": float("nan"), "ea_ratio": float("nan"),
           "tind_spread_k": spread, "tind_n": float(t.size)}
    if t.size < int(f.tind_min_samples) or spread < float(f.tind_min_spread_k):
        return out
    ea = float(bl.ea_j_per_mol.value or cfg.physics.ea_literature_j_per_mol)
    rg = float(cfg.physics.rg_j_per_mol_k)
    inv = 1.0 / arrhenius_factor(T, ea, rg, float(cfg.physics.t25_k))
    tc = t - t.mean()
    A = np.column_stack([np.ones_like(tc), tc, inv])
    coef, *_ = np.linalg.lstsq(A, R25, rcond=None)
    c2 = float(coef[2])
    out["c2_ohm"] = c2
    out["b_rel"] = c2 * float(np.median(inv)) / cs.r0 if cs.r0 > 0 else float("nan")
    A2 = np.column_stack([np.ones_like(tc), tc, 1.0 / (T + KELVIN)])
    coef2, *_ = np.linalg.lstsq(A2, np.log(R), rcond=None)
    out["ea_apparent"] = float(coef2[2] * rg)
    out["ea_ratio"] = out["ea_apparent"] / ea
    return out


# --------------------------------------------------------------------------- recent features
def recent_features(cs: CellSeries, idx_raw: np.ndarray, idx_corr: np.ndarray, rth: float) -> dict[str, np.ndarray]:
    """Recent-window features, vectorised over rows of index matrices (bootstrap-friendly).

    ``idx_raw`` / ``idx_corr`` are integer arrays of shape (B, w). Returns arrays of shape (B,).
    raw_excess = median(R_raw)/R0 - 1, corrected_excess = median(R25)/R0 - 1, component
    excesses, median OCV deviation (mV), median SOH, median cell-minus-pack (K), expected
    I^2 R heating (K), median steady-current noise (mV).
    """
    def med(a: np.ndarray, idx: np.ndarray) -> np.ndarray:
        v = a[idx]
        allnan = ~np.isfinite(v).any(axis=1)
        v = np.where(allnan[:, None], 0.0, v)
        r = np.nanmedian(v, axis=1)
        return np.where(allnan, np.nan, r)

    out = {
        "raw_excess": med(cs.r_raw, idx_raw) / cs.r0 - 1.0,
        "corrected_excess": med(cs.r25, idx_corr) / cs.r0 - 1.0,
        "corrected_excess_ohmic": med(cs.r_ohm25, idx_corr) / cs.r0_ohm - 1.0 if cs.r0_ohm > 0 else np.full(idx_corr.shape[0], np.nan),
        "corrected_excess_slow": med(cs.r_slow25, idx_corr) / cs.r0_slow - 1.0 if cs.r0_slow > 0 else np.full(idx_corr.shape[0], np.nan),
        "ocv_recent_mv": med(cs.ocv, idx_corr),
        "soh_recent": med(cs.soh, idx_corr),
        "cell_minus_pack_k": med(cs.temp - cs.temp_pack, idx_corr),
        "dT_expected_k": med(cs.i_rms ** 2 * cs.r_raw * rth, idx_corr),
        "noise_recent_mv": med(cs.noise, idx_corr),
    }
    return out


def plating_rise(cs: CellSeries, cfg: Config) -> dict[str, float]:
    """Resistance rise persisting after cold hard-charge samples.

    ``post_cold_charge_rise = median(R25 after the first cold-charge sample) /
    median(R25 over the ``plating_pre_samples`` samples before it) - 1``. Zero when there
    are no cold-charge samples (no plating signal), NaN when there is too little history
    before the first event.
    """
    ev = np.where(cs.cold > 0.5)[0]
    n_ev = int(ev.size)
    if n_ev < int(cfg.features.plating_min_events):
        return {"post_cold_charge_rise": 0.0, "n_cold_charge": float(n_ev), "first_cold_index": float("nan")}
    e0 = int(ev[0])
    npre = int(cfg.features.plating_pre_samples)
    before = cs.r25[max(0, e0 - npre) : e0]
    after = cs.r25[e0 + 1 :]
    before = before[np.isfinite(before)]
    after = after[np.isfinite(after)]
    if before.size < 3 or after.size < 1:
        return {"post_cold_charge_rise": float("nan"), "n_cold_charge": float(n_ev), "first_cold_index": float(e0)}
    rise = float(np.median(after) / np.median(before) - 1.0)
    return {"post_cold_charge_rise": rise, "n_cold_charge": float(n_ev), "first_cold_index": float(e0)}


def ocv_soc_shift(ocv_mv: float, soc: float, bl: Baseline) -> float:
    """Equivalent SOC shift = OCV deviation / (dOCV/dSOC at the cell's SOC), dimensionless."""
    c = bl.ocv_soc_curve
    if c is None or not np.isfinite(ocv_mv) or len(c.x) < 3:
        return float("nan")
    x = np.asarray(c.x, float)
    y = np.asarray(c.y, float) * 1000.0  # V -> mV
    s = 0.5 if not np.isfinite(soc) else float(np.clip(soc, x.min(), x.max()))
    slope = float(np.interp(s, x, np.gradient(y, x)))
    return ocv_mv / slope if slope != 0 else float("nan")


def compute_features(samples: pd.DataFrame, bl: Baseline, cfg: Config, mode: str) -> CellFeatures:
    """All per-cell features (section 6.3) except neighbour comparison (added fleet-wide).

    With ``fixes.ea_adapt`` on, if the apparent Ea over the last 365 samples exceeds the
    baseline Ea by ``ea_adapt_min_ratio``, the features are recomputed with the baseline Ea
    values scaled by that ratio (capped at ``ea_adapt_max_ratio``). Upward only.
    """
    cf = _compute_features(samples, bl, cfg, mode)
    fx = cfg.fixes
    ratio = cf.values.get("ea_ratio", float("nan"))
    if bool(fx.ea_adapt) and np.isfinite(ratio) and ratio >= float(fx.ea_adapt_min_ratio):
        k = min(float(ratio), float(fx.ea_adapt_max_ratio))
        upd = {}
        for key in ("ea_j_per_mol", "ea_ohmic_j_per_mol", "ea_slow_j_per_mol"):
            p = getattr(bl, key)
            if p.value:
                upd[key] = p.model_copy(update={"value": float(p.value) * k, "source": p.source + "+adapted"})
        cf2 = _compute_features(samples, bl.model_copy(update=upd), cfg, mode)
        cf2.values["ea_adapted_ratio"] = k
        cf2.values["ea_ratio_before_adapt"] = float(ratio)
        cf2.notes.append(f"apparent Ea is {ratio:.2f} x the baseline Ea; features recomputed with Ea x {k:.2f}")
        return cf2
    return cf


def _compute_features(samples: pd.DataFrame, bl: Baseline, cfg: Config, mode: str) -> CellFeatures:
    cf = CellFeatures()
    n_in = len(samples)
    cs = build_series(samples, bl, cfg)
    cf.series = cs
    n = cs.t.size
    v = cf.values
    v["n_samples"] = float(n)
    v["n_dropped_no_r_or_t"] = float(n_in - n)
    if n_in - n:
        cf.notes.append(f"{n_in - n} samples had no resistance or temperature and were not used")
    if n == 0:
        cf.availability["all"] = "no usable samples"
        return cf
    f = cfg.features
    rth = float(f.thermal_resistance_k_per_w)
    idx_raw = np.arange(n - cs.win_raw, n)[None, :]
    idx_corr = np.arange(n - cs.win_corr, n)[None, :]
    rec = recent_features(cs, idx_raw, idx_corr, rth)
    for k, a in rec.items():
        v[k] = float(a[0])
    v["win_raw"] = float(cs.win_raw)
    v["win_corr"] = float(cs.win_corr)
    v["temp_recent_c"] = float(np.median(cs.temp[-cs.win_corr:]))
    v["soc_recent"] = float(np.nanmedian(cs.soc[-cs.win_corr:])) if np.isfinite(cs.soc[-cs.win_corr:]).any() else float("nan")
    v["temp_spread_k"] = robust_spread(cs.temp[-int(f.tind_window):], *f.temp_spread_quantiles)
    v["temp_spread_history_k"] = robust_spread(cs.temp, *f.temp_spread_quantiles)

    # ---- availability of optional inputs
    has_ocv = bool(np.isfinite(cs.ocv).any())
    has_soh = bool(np.isfinite(cs.soh).any())
    has_pack = bool(np.isfinite(cs.temp_pack).any())
    cf.flags["has_ocv"] = has_ocv
    cf.flags["has_soh"] = has_soh
    cf.flags["has_pack_temp"] = has_pack
    cf.flags["has_components"] = bool(np.isfinite(cs.r_ohm25).any() and np.isfinite(cs.r_slow25).any())
    if not has_ocv:
        cf.availability["ocv"] = "no OCV data"
    if not has_soh:
        cf.availability["soh"] = "no SOH data"
    if not has_pack:
        cf.availability["cell_minus_pack"] = "no pack temperature"
    if not np.isfinite(v["noise_recent_mv"]):
        cf.availability["voltage_noise"] = "needs pulse_mode (100 Hz) data" if mode == "cycle_mode" else "no steady-current windows"
    if mode == "cycle_mode":
        cf.availability["pulse_split"] = "cycle_mode: no ohmic/slow pulse split"

    # ---- steps
    win = window_len(n, float(cfg.windows.step_win_frac), int(cfg.windows.step_win_min))
    v["step_win"] = float(win)
    nf, fl, meth = float(f.step_noise_factor), float(f.step_noise_floor_rel), str(f.step_method)
    rob = bool(cfg.fixes.robust_step_noise)
    st = step_detect(cs.r25, win, nf, fl, meth, rob)
    v["step_jump_ohm"], v["step_rel"], v["step_z"], v["step_index"], v["r25_noise_ohm"] = (
        st["jump"], st["rel"], st["z"], st["index"], st["noise"])
    if np.isfinite(st["index"]):
        ki = int(st["index"])
        v["step_t"] = float(cs.t[ki])
        v["temp_step_at_r_step_k"] = float(np.median(cs.temp[ki : ki + win]) - np.median(cs.temp[max(0, ki - win) : ki]))
    else:
        cf.availability["step"] = f"needs at least {2 * win} samples"
    sh = step_detect(cs.soh, win, nf, fl, meth, rob) if has_soh else {"jump": float("nan"), "z": float("nan"), "index": float("nan")}
    v["soh_step"], v["soh_step_z"] = sh["jump"], sh["z"]
    oc = step_detect(cs.ocv, win, nf, fl, meth, rob) if has_ocv else {"jump": float("nan"), "z": float("nan"), "index": float("nan")}
    v["ocv_step_mv"], v["ocv_step_z"] = oc["jump"], oc["z"]

    # ---- growth
    g = sqrt_growth(cs.t, cs.r25, int(f.growth_min_samples))
    v["growth_a_ohm"], v["growth_b"], v["growth_r2"] = g["a"], g["b"], g["r2"]
    if np.isfinite(cs.ah).sum() >= int(f.growth_min_samples):
        ga = sqrt_growth(cs.ah, cs.r25, int(f.growth_min_samples))
        v["growth_b_ah"], v["growth_r2_ah"] = ga["b"], ga["r2"]
    else:
        v["growth_b_ah"], v["growth_r2_ah"] = float("nan"), float("nan")
    slow_ok = np.isfinite(cs.r_slow25)
    if slow_ok.sum() >= int(f.growth_min_samples):
        gs = sqrt_growth(cs.t[slow_ok], cs.r_slow25[slow_ok], int(f.growth_min_samples))
        v["growth_b_slow"], v["growth_r2_slow"] = gs["b"], gs["r2"]

    # ---- temperature-independent share
    ti = temperature_independent_share(cs, bl, cfg)
    v.update({k: float(x) for k, x in ti.items()})
    if not np.isfinite(ti["b_rel"]):
        cf.availability["b_rel"] = (f"needs >= {f.tind_min_samples} samples and >= {f.tind_min_spread_k} K spread in "
                                    f"the last {f.tind_window} (have {int(ti['tind_n'])} samples, {ti['tind_spread_k']:.1f} K)")

    # ---- OCV / SOH
    v["ocv_soc_shift"] = ocv_soc_shift(v["ocv_recent_mv"], v["soc_recent"], bl)
    v["soh_fall"] = 1.0 - v["soh_recent"] if np.isfinite(v["soh_recent"]) else float("nan")

    # ---- thermal
    v["unexplained_heat_k"] = v["cell_minus_pack_k"] - v["dT_expected_k"] if np.isfinite(v["dT_expected_k"]) else v["cell_minus_pack_k"]

    # ---- plating
    v.update(plating_rise(cs, cfg))

    # ---- sensor sanity
    reasons: list[str] = []
    if has_pack:
        d = cs.temp - cs.temp_pack
        md = float(np.nanmedian(d[-cs.win_corr:]))
        if md < -float(f.sensor_temp_cold_offset_k):
            reasons.append(f"cell reads {abs(md):.1f} K colder than the pack sensor (a cell cannot be colder "
                           f"than its surroundings while carrying current)")
        if np.nanmax(np.abs(d)) > float(f.sensor_temp_abs_disagree_k):
            reasons.append(f"cell and pack temperatures differ by up to {np.nanmax(np.abs(d)):.1f} K")
    bms = cs.bms_r
    if np.isfinite(bms).sum() >= 3 and mode == "pulse_mode":
        w_ = cs.win_corr
        rel = float(np.nanmedian(np.abs(cs.r_raw[-w_:] / bms[-w_:] - 1.0)))  # recent window
        v["bms_vs_pulse_rel"] = rel
        if rel > float(cfg.pulse.bms_disagreement_rel):
            reasons.append(f"BMS-reported resistance differs from pulse-derived resistance by {rel:.0%}")
    cf.flags["sensor_fault_suspected"] = bool(reasons)
    if reasons:
        cf.notes.extend(reasons)
    v["sensor_reasons_n"] = float(len(reasons))
    cf.flags["sensor_reasons"] = reasons  # type: ignore[assignment]
    return cf
