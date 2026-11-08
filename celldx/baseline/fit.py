"""Baseline fitting: early-life resistance, activation energy with bootstrap, spreads.

Formulas
--------
Arrhenius: ``R(T) = R25 * exp(Ea/Rg * (1/T - 1/T25))``  =>  ``ln R = ln R25 + (Ea/Rg) * (1/T - 1/T25)``.
So Ea = Rg * slope of ln R against 1/T (T in kelvin).

Within-cell fit (preferred): per-cell fixed effects, i.e. ln R and 1/T are de-meaned inside
each cell before the regression, so cell-to-cell differences in R25 cannot masquerade as a
temperature effect. Between-cell fit (fallback): a pooled regression across cells that were
each tested at one temperature. It is confounded with cell-to-cell spread and is flagged
``fitted_between_cell``. If neither reaches the configured temperature spread (8 K), Ea is the
literature default and the reason is recorded.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..physics import KELVIN, robust_spread, to_r25
from .model import Baseline, Curve, Param, Provenance


@dataclass
class EaFit:
    value: float
    source: str  # fitted_within_cell | fitted_between_cell | fitted_own | literature_default
    ci: list[float] | None
    method: str
    spread_k: float
    within_cell_spread_k: float
    n_cells: int
    n_rows: int
    reason: str
    candidates: dict[str, Any] = field(default_factory=dict)

    def to_param(self) -> Param:
        return Param(value=self.value, source=self.source, ci=self.ci, n=self.n_rows, note=self.reason)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fe_slope(cells: np.ndarray, x: np.ndarray, y: np.ndarray, t: np.ndarray | None) -> float:
    """Fixed-effects slope of y on x (optionally with a per-cell linear time trend)."""
    xs, ys, ts = [], [], []
    for c in np.unique(cells):
        m = cells == c
        if m.sum() < 2:
            continue
        xs.append(x[m] - x[m].mean())
        ys.append(y[m] - y[m].mean())
        if t is not None:
            ts.append(t[m] - t[m].mean())
    if not xs:
        return float("nan")
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    if t is not None:
        A = np.column_stack([X, np.concatenate(ts)])
        if np.linalg.matrix_rank(A) < 2:
            return float("nan")
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        return float(coef[0])
    den = float((X * X).sum())
    return float((X * Y).sum() / den) if den > 0 else float("nan")


def _pooled_slope(x: np.ndarray, y: np.ndarray) -> float:
    xm = x - x.mean()
    den = float((xm * xm).sum())
    return float((xm * (y - y.mean())).sum() / den) if den > 0 else float("nan")


def fit_ea(
    cell_ids: np.ndarray | list[str],
    temp_c: np.ndarray,
    r_ohm: np.ndarray,
    cfg: Config,
    time: np.ndarray | None = None,
    seed: int = 0,
    allow_between_cell: bool = True,
    own: bool = False,
) -> EaFit:
    """Fit an activation energy (J/mol) from ln R versus 1/T, with a bootstrap interval.

    Bootstrap resamples whole cells when there are at least 3 cells, otherwise rows.
    Gates: temperature spread >= ``baseline.ea_min_spread_k``, estimate inside
    ``physics.ea_plausible_range_j_per_mol``, CI lower bound > 0 and relative CI width below
    ``baseline.ea_max_rel_ci_width``. On any failure the literature default is returned with
    the reason.
    """
    rg = float(cfg.physics.rg_j_per_mol_k)
    ea_lit = float(cfg.physics.ea_literature_j_per_mol)
    min_spread = float(cfg.baseline.ea_min_spread_k)
    lo_ok, hi_ok = cfg.physics.ea_plausible_range_j_per_mol
    cells = np.asarray(cell_ids).astype(str)
    T = np.asarray(temp_c, float)
    R = np.asarray(r_ohm, float)
    tt = None if time is None else np.asarray(time, float)
    ok = np.isfinite(T) & np.isfinite(R) & (R > 0)
    if tt is not None:
        ok &= np.isfinite(tt)
    cells, T, R = cells[ok], T[ok], R[ok]
    if tt is not None:
        tt = tt[ok]
    n = int(ok.sum())
    q = tuple(cfg.features.temp_spread_quantiles)

    def lit(reason: str, spread: float, wspread: float, cand: dict[str, Any] | None = None) -> EaFit:
        return EaFit(ea_lit, "literature_default", list(cfg.physics.ea_literature_range_j_per_mol),
                     "literature_default", spread, wspread, int(len(np.unique(cells))), n, reason, cand or {})

    if n < 3:
        return lit("fewer than 3 usable rows", 0.0, 0.0)
    x = 1.0 / (T + KELVIN)
    y = np.log(R)
    spread = robust_spread(T, *q)
    per_cell = {c: robust_spread(T[cells == c], *q) for c in np.unique(cells)}
    wspread = max(per_cell.values()) if per_cell else 0.0
    within_cells = [c for c, s in per_cell.items() if s >= min_spread]
    rng = np.random.default_rng(seed)
    nboot = int(cfg.baseline.ea_bootstrap_n)
    lo_q, hi_q = cfg.baseline.ea_ci

    def boot(slope_fn: Any, use_cells: list[str]) -> list[float]:
        m = np.isin(cells, use_cells)
        c_, x_, y_ = cells[m], x[m], y[m]
        t_ = None if tt is None else tt[m]
        uc = np.unique(c_)
        vals = []
        for _ in range(nboot):
            if len(uc) >= 3:
                pick = rng.choice(uc, size=len(uc), replace=True)
                idx = np.concatenate([np.where(c_ == p)[0] for p in pick])
                # relabel duplicates so fixed effects stay separate
                lab = np.concatenate([np.full((c_ == p).sum(), f"{p}#{k}") for k, p in enumerate(pick)])
            else:
                idx = rng.integers(0, len(c_), len(c_))
                lab = c_[idx]
            s = slope_fn(lab, x_[idx], y_[idx], None if t_ is None else t_[idx])
            if np.isfinite(s):
                vals.append(s * rg)
        return vals

    cand: dict[str, Any] = {}
    if within_cells:
        m = np.isin(cells, within_cells)
        slope = _fe_slope(cells[m], x[m], y[m], None if tt is None else tt[m])
        est = slope * rg
        vals = boot(_fe_slope, within_cells)
        ci = [float(np.quantile(vals, lo_q)), float(np.quantile(vals, hi_q))] if len(vals) > 10 else None
        source = "fitted_own" if own else "fitted_within_cell"
        method = "fixed_effects_ln_r_vs_inv_t" + ("_with_time_trend" if tt is not None else "")
        cand["within_cell"] = {"value": est, "ci": ci, "cells": within_cells}
        reason = _gate(est, ci, lo_ok, hi_ok, float(cfg.baseline.ea_max_rel_ci_width))
        if reason is None:
            return EaFit(float(est), source, ci, method, spread, wspread, len(within_cells), int(m.sum()),
                         f"within-cell spread {wspread:.1f} K >= {min_spread} K", cand)
        if not allow_between_cell:
            return lit(f"within-cell fit rejected: {reason}", spread, wspread, cand)
    if allow_between_cell and spread >= min_spread and len(np.unique(cells)) >= 2:
        slope = _pooled_slope(x, y)
        est = slope * rg
        vals = boot(lambda c_, x_, y_, t_: _pooled_slope(x_, y_), list(np.unique(cells)))
        ci = [float(np.quantile(vals, lo_q)), float(np.quantile(vals, hi_q))] if len(vals) > 10 else None
        cand["between_cell"] = {"value": est, "ci": ci}
        reason = _gate(est, ci, lo_ok, hi_ok, float(cfg.baseline.ea_max_rel_ci_width))
        if reason is None:
            return EaFit(float(est), "fitted_between_cell", ci, "pooled_ln_r_vs_inv_t_across_cells", spread,
                         wspread, len(np.unique(cells)), n,
                         f"no cell has {min_spread} K of its own spread; pooled across cells at different "
                         f"temperatures (confounded with cell-to-cell variation)", cand)
        return lit(f"between-cell fit rejected: {reason}", spread, wspread, cand)
    return lit(f"temperature spread {spread:.1f} K (within-cell max {wspread:.1f} K) is below "
               f"{min_spread} K: the data cannot support an activation energy", spread, wspread, cand)


def _gate(est: float, ci: list[float] | None, lo_ok: float, hi_ok: float, max_rel: float) -> str | None:
    if not np.isfinite(est):
        return "slope not finite"
    if not (lo_ok <= est <= hi_ok):
        return f"estimate {est / 1000:.1f} kJ/mol outside plausible range [{lo_ok / 1000:.0f}, {hi_ok / 1000:.0f}]"
    if ci is None:
        return "bootstrap failed"
    if ci[0] <= 0:
        return f"bootstrap interval includes zero ({ci[0] / 1000:.1f} to {ci[1] / 1000:.1f} kJ/mol)"
    if (ci[1] - ci[0]) > max_rel * est:
        return f"bootstrap interval too wide ({ci[0] / 1000:.1f} to {ci[1] / 1000:.1f} kJ/mol)"
    return None


def early_life_mask(order: np.ndarray, cell_ids: np.ndarray, cfg: Config) -> np.ndarray:
    """Boolean mask of each cell's first N rows: N = max(min, ceil(frac * n_cell))."""
    order = np.asarray(order, float)
    cells = np.asarray(cell_ids).astype(str)
    mask = np.zeros(len(order), dtype=bool)
    frac = float(cfg.baseline.early_life_frac)
    nmin = int(cfg.baseline.early_life_min)
    for c in np.unique(cells):
        idx = np.where(cells == c)[0]
        idx = idx[np.argsort(order[idx], kind="stable")]
        k = max(nmin, math.ceil(frac * len(idx)))
        mask[idx[:k]] = True
    return mask


def flat_soc_map(note: str = "no SOC-resolved resistance data; map assumed flat (factor 1)") -> Curve:
    return Curve(x=[0.0, 1.0], y=[1.0, 1.0], source="assumed_flat", note=note)


def fit_soc_map(soc: np.ndarray, r25: np.ndarray, soc_ref: float, bins: int = 10, source: str = "fitted") -> Curve:
    """Median resistance per SOC bin normalised to the value at ``soc_ref`` (dimensionless)."""
    s = np.asarray(soc, float)
    r = np.asarray(r25, float)
    ok = np.isfinite(s) & np.isfinite(r)
    s, r = s[ok], r[ok]
    edges = np.linspace(0, 1, bins + 1)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        m = (s >= a) & (s < b) if b < 1 else (s >= a) & (s <= b)
        if m.sum() >= 3:
            xs.append(float(np.median(s[m])))
            ys.append(float(np.median(r[m])))
    if len(xs) < 3:
        return flat_soc_map("too few SOC bins with data; map assumed flat")
    ref = float(np.interp(soc_ref, xs, ys))
    return Curve(x=xs, y=[y / ref for y in ys], source=source,
                 note=f"median R per SOC bin ({bins} bins), normalised at soc_ref={soc_ref}")


def fit_population_baseline(
    table: pd.DataFrame,
    cfg: Config,
    *,
    baseline_id: str,
    chemistry: str,
    form_factor: str,
    nominal_capacity_ah: float | None,
    resistance_kind: str,
    provenance: Provenance,
    kind: str = "population",
    soc_map: Curve | None = None,
    ocv_curve: Curve | None = None,
    allow_between_cell: bool = True,
    ea_time_trend: bool = False,
    seed: int = 0,
    ea_override: Mapping[str, EaFit] | None = None,
) -> Baseline:
    """Fit a baseline from a table with columns
    ``cell_id, order, temp_c, r_total, r_ohmic, r_slow, capacity_ah`` (r_ohmic, r_slow,
    capacity_ah may be NaN).

    Steps: (1) early-life rows of each cell, (2) Ea for total, ohmic and slow resistance,
    (3) per-cell median R25 of each component, (4) population median and standard deviation
    across cells.
    """
    t = table.copy()
    em = early_life_mask(t["order"].to_numpy(), t["cell_id"].to_numpy(), cfg)
    early = t[em]
    own = kind == "own"
    ea_fits: dict[str, EaFit] = {}
    for comp in ("r_total", "r_ohmic", "r_slow"):
        if ea_override and comp in ea_override:
            ea_fits[comp] = ea_override[comp]
            continue
        sub = early[np.isfinite(early[comp].to_numpy(float))]
        if len(sub) < 3:
            ea_fits[comp] = fit_ea([], np.array([]), np.array([]), cfg)
            continue
        ea_fits[comp] = fit_ea(sub["cell_id"].to_numpy(), sub["temp_c"].to_numpy(), sub[comp].to_numpy(), cfg,
                               time=sub["order"].to_numpy(float) if ea_time_trend else None, seed=seed,
                               allow_between_cell=allow_between_cell, own=own)
    rg = float(cfg.physics.rg_j_per_mol_k)
    t25 = float(cfg.physics.t25_k)
    per_cell: dict[str, dict[str, float]] = {}
    for c, g in early.groupby("cell_id"):
        d: dict[str, float] = {}
        for comp in ("r_total", "r_ohmic", "r_slow"):
            vals = g[comp].to_numpy(float)
            ok = np.isfinite(vals)
            if ok.sum():
                r25 = to_r25(vals[ok], g["temp_c"].to_numpy(float)[ok], ea_fits[comp].value, rg, t25)
                d[comp] = float(np.median(r25))
        cap = g["capacity_ah"].to_numpy(float) if "capacity_ah" in g else np.array([])
        if np.isfinite(cap).any():
            d["capacity_ah"] = float(np.median(cap[np.isfinite(cap)]))
        per_cell[str(c)] = d

    def pop(key: str, source: str) -> Param:
        vals = np.array([d[key] for d in per_cell.values() if key in d])
        if vals.size == 0:
            return Param(value=None, source="unavailable", note="no data for this component")
        std = float(np.std(vals, ddof=1)) if vals.size > 1 else None
        return Param(value=float(np.median(vals)), source=source, population_std=std, n=int(vals.size),
                     note=f"median of per-cell early-life medians over {vals.size} cell(s)")

    src = "fitted_own" if own else "fitted_population"
    cyc = {str(c): [float(g["order"].min()), float(g["order"].max())] for c, g in early.groupby("cell_id")}
    provenance.cycle_ranges = cyc
    provenance.cells = sorted(per_cell)
    provenance.config_hash = cfg.hash()
    provenance.fitting_method = (
        "early life = first max(" + str(cfg.baseline.early_life_min) + ", " + str(cfg.baseline.early_life_frac)
        + " x n) rows per cell; Ea from ln R vs 1/T (fixed effects within cell, else pooled, else literature); "
        "R25 = R / exp(Ea/Rg (1/T - 1/T25)); per-cell median, then median across cells"
    )
    cannot: list[str] = list(provenance.data_cannot_support)
    for comp, f in ea_fits.items():
        if f.source == "literature_default":
            cannot.append(f"activation energy ({comp}): {f.reason}; literature default "
                          f"{cfg.physics.ea_literature_j_per_mol / 1000:.0f} kJ/mol used")
        if f.source == "fitted_between_cell":
            cannot.append(f"activation energy ({comp}) is a between-cell fit: {f.reason}")
    provenance.data_cannot_support = cannot
    provenance.notes.append("ea_fits=" + str({k: {"value": round(v.value, 1), "source": v.source, "ci": v.ci,
                                                   "spread_k": round(v.spread_k, 2)} for k, v in ea_fits.items()}))
    cap = pop("capacity_ah", src)
    return Baseline(
        baseline_id=baseline_id,
        kind=kind,  # type: ignore[arg-type]
        chemistry=chemistry,
        form_factor=form_factor,
        nominal_capacity_ah=nominal_capacity_ah,
        resistance_kind=resistance_kind,
        soc_ref=float(cfg.physics.soc_ref),
        r0_25c_ohm=pop("r_total", src),
        r0_ohmic_ohm=pop("r_ohmic", src),
        r0_slow_ohm=pop("r_slow", src),
        ea_j_per_mol=ea_fits["r_total"].to_param(),
        ea_ohmic_j_per_mol=ea_fits["r_ohmic"].to_param(),
        ea_slow_j_per_mol=ea_fits["r_slow"].to_param(),
        soh0_ah=cap,
        r_soc_map=soc_map or flat_soc_map(),
        ocv_soc_curve=ocv_curve,
        provenance=provenance,
    )


def fit_own_baseline(samples: pd.DataFrame, cfg: Config, *, cell_id: str, chemistry: str = "unknown",
                     form_factor: str = "unknown", nominal_capacity_ah: float | None = None,
                     resistance_kind: str = "unknown", seed: int = 0) -> Baseline:
    """Own early-life baseline for one cell from its sample table
    (columns ``t, r_total, r_ohmic, r_slow, temp_c, soh``).

    Ea: early-life window; if its spread is below 8 K the window is widened up to
    ``baseline.own_ea_extend_frac`` of the history with a linear time-trend term so ageing
    does not bias the slope. If that still lacks spread, the literature default is used.
    """
    s = samples.sort_values("t").reset_index(drop=True)
    tab = pd.DataFrame({
        "cell_id": cell_id, "order": s["t"].to_numpy(float), "temp_c": s["temp_c"].to_numpy(float),
        "r_total": s["r_total"].to_numpy(float), "r_ohmic": s.get("r_ohmic", np.nan),
        "r_slow": s.get("r_slow", np.nan),
        "capacity_ah": (s["soh"] * nominal_capacity_ah) if (nominal_capacity_ah and "soh" in s) else np.nan,
    })
    prov = Provenance(dataset=f"own early life of {cell_id}", fitting_method="own early life")
    return fit_with_widening(tab, cfg, baseline_id=f"own-{cell_id}", chemistry=chemistry, form_factor=form_factor,
                             nominal_capacity_ah=nominal_capacity_ah, resistance_kind=resistance_kind,
                             provenance=prov, kind="own", allow_between_cell=False, seed=seed)


_EA_KEYS = {"r_total": "ea_j_per_mol", "r_ohmic": "ea_ohmic_j_per_mol", "r_slow": "ea_slow_j_per_mol"}


def fit_with_widening(table: pd.DataFrame, cfg: Config, *, provenance: Provenance, kind: str = "population",
                      allow_between_cell: bool = True, seed: int = 0, **kw: Any) -> Baseline:
    """:func:`fit_population_baseline`, then, for every component whose Ea fell back to the
    literature default because early life has too little temperature spread, refit Ea on the
    first ``baseline.own_ea_extend_frac`` of each cell's history with a linear time-trend term
    (so ageing does not bias the slope; fixed effects within cell) and refit R0 with it.
    R0 itself is still the early-life median. Used by own and pack baselines."""
    bl = fit_population_baseline(table, cfg, provenance=provenance.model_copy(deep=True), kind=kind,
                                 allow_between_cell=allow_between_cell, seed=seed, **kw)
    lit = [c for c, key in _EA_KEYS.items() if getattr(bl, key).source == "literature_default"]
    if not lit:
        return bl
    frac = float(cfg.baseline.own_ea_extend_frac)
    wide = early_life_mask(table["order"].to_numpy(), table["cell_id"].to_numpy(),
                           cfg.with_overrides({"baseline.early_life_frac": frac}))
    w = table[wide]
    override: dict[str, EaFit] = {}
    for comp in lit:
        sub = w[np.isfinite(w[comp].to_numpy(float))]
        if len(sub) < 3:
            continue
        f = fit_ea(sub["cell_id"].to_numpy(), sub["temp_c"].to_numpy(), sub[comp].to_numpy(float), cfg,
                   time=sub["order"].to_numpy(float), seed=seed, allow_between_cell=False, own=kind == "own")
        if f.source != "literature_default":
            n_cell = int(round(len(sub) / max(1, sub["cell_id"].nunique())))
            f.reason += f" (window widened to first {n_cell} samples with time trend)"
            override[comp] = f
    if not override:
        return bl
    return fit_population_baseline(table, cfg, provenance=provenance.model_copy(deep=True), kind=kind,
                                   allow_between_cell=allow_between_cell, seed=seed, ea_override=override, **kw)


def literature_baseline(cfg: Config, r0_ohm: float, *, chemistry: str, form_factor: str,
                        nominal_capacity_ah: float | None, resistance_kind: str, note: str) -> Baseline:
    """A baseline built from user-entered literature values. Trust 0.70, every value labelled."""
    ea = float(cfg.physics.ea_literature_j_per_mol)
    lit = "literature_default"
    return Baseline(
        baseline_id="literature", kind="literature", chemistry=chemistry, form_factor=form_factor,
        nominal_capacity_ah=nominal_capacity_ah, resistance_kind=resistance_kind,
        soc_ref=float(cfg.physics.soc_ref),
        r0_25c_ohm=Param(value=r0_ohm, source=lit, note=note),
        r0_ohmic_ohm=Param(value=None, source="unavailable"), r0_slow_ohm=Param(value=None, source="unavailable"),
        ea_j_per_mol=Param(value=ea, source=lit, ci=list(cfg.physics.ea_literature_range_j_per_mol)),
        ea_ohmic_j_per_mol=Param(value=ea, source=lit), ea_slow_j_per_mol=Param(value=ea, source=lit),
        soh0_ah=Param(value=nominal_capacity_ah, source=lit), r_soc_map=flat_soc_map(),
        provenance=Provenance(dataset="literature_default", fitting_method="user-entered literature values",
                              notes=[note]),
    )


def check_compatibility(bl: Baseline, cfg: Config, *, chemistry: str | None, form_factor: str | None,
                        capacity_ah: float | None, r_typical_ohm: float | None,
                        resistance_kind: str | None) -> list[str]:
    """Warnings when a baseline is applied to a materially different cell (chemistry, form
    factor, capacity, resistance level or resistance definition). Empty list = compatible."""
    w: list[str] = []
    if bl.kind == "own":
        return w
    if chemistry and bl.chemistry != "unknown" and chemistry.lower() != bl.chemistry.lower():
        w.append(f"CHEMISTRY MISMATCH: baseline is {bl.chemistry} ({bl.provenance.dataset}), target is {chemistry}. "
                 "Resistance, Ea and OCV do not transfer between chemistries; fit an own early-life baseline.")
    if form_factor and bl.form_factor != "unknown" and form_factor.lower() != bl.form_factor.lower():
        w.append(f"FORM-FACTOR MISMATCH: baseline is {bl.form_factor}, target is {form_factor}.")
    cap0 = bl.nominal_capacity_ah
    if capacity_ah and cap0 and abs(capacity_ah / cap0 - 1) > float(cfg.baseline.capacity_mismatch_rel):
        w.append(f"CAPACITY MISMATCH: baseline cells are {cap0:g} Ah, target is {capacity_ah:g} Ah.")
    r0 = bl.r0_25c_ohm.value
    if r_typical_ohm and r0 and abs(r_typical_ohm / r0 - 1) > float(cfg.baseline.resistance_mismatch_rel):
        w.append(f"RESISTANCE-LEVEL MISMATCH: target early resistance {r_typical_ohm:.4g} ohm vs baseline R0 {r0:.4g} ohm.")
    if resistance_kind and bl.resistance_kind not in ("unknown", resistance_kind):
        w.append(f"RESISTANCE-DEFINITION MISMATCH: baseline is {bl.resistance_kind}, target data is {resistance_kind}.")
    return w
