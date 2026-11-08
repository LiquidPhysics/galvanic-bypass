"""Fleet and single-cell diagnosis: validation -> samples -> features -> neighbours ->
scores -> probabilities -> confidence (+ bootstrap) -> explanation.

Public entry points: :func:`diagnose` (batch, many cells) and :func:`diagnose_latest`
(streaming-friendly: score one cell's latest window).
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .. import CAUSE_LABELS, CAUSES, NON_FAULT_CAUSES, __version__
from ..baseline.fit import check_compatibility, fit_own_baseline
from ..baseline.model import Baseline
from ..config import Config, load_config
from ..features.core import CellFeatures, compute_features, cycle_cell_to_samples, recent_features
from ..features.pulse import pulse_cell_to_samples
from ..schema import validate
from .confidence import compute_confidence
from .explain import explain, headline
from .scoring import cause_scores, decision_tree, fault_probability, probabilities, sub_scores

REPORT_SCHEMA = "celldx-report/1"


@dataclass
class TargetMeta:
    """What the diagnosed cells are, used for baseline compatibility checks."""

    chemistry: str | None = None
    form_factor: str | None = None
    capacity_ah: float | None = None
    resistance_kind: str | None = None


@dataclass
class Calibration:
    """Fitted calibration maps; ``None`` members mean that part is uncalibrated."""

    fault: Any | None = None          # callable p -> p_cal
    multiclass_t: float | None = None  # temperature for temperature scaling
    confidence: Any | None = None     # callable c -> P(correct)
    source: str = "none"

    @property
    def status(self) -> str:
        return "calibrated" if (self.fault is not None or self.multiclass_t is not None) else "uncalibrated_scores"


@dataclass
class CellContext:
    cell_id: str
    pack_id: str
    string_id: str
    mode: str
    samples: pd.DataFrame
    baseline: Baseline
    extraction: dict[str, float] = field(default_factory=dict)


def detect_mode(g: pd.DataFrame, cfg: Config) -> str:
    """pulse_mode if the median sample interval is <= ``mode.pulse_max_median_dt_s`` and
    voltage and current are present; otherwise cycle_mode."""
    t = np.sort(g["t"].dropna().to_numpy(float))
    if t.size < 3:
        return "cycle_mode"
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        return "cycle_mode"
    has_vi = g["voltage_v"].notna().mean() > 0.5 and g["current_a"].notna().mean() > 0.5
    return "pulse_mode" if (float(np.median(dt)) <= float(cfg.mode.pulse_max_median_dt_s) and has_vi) else "cycle_mode"


def derive_cold_charge(df: pd.DataFrame, cfg: Config, capacity_ah: float | None) -> pd.DataFrame:
    """Fill missing ``cold_charge``: charge C-rate >= ``plating_c_rate`` and cell temperature
    < ``plating_temp_c``. Needs a capacity; without one the column stays as given (NaN = 0)."""
    if not capacity_ah:
        return df
    need = df["cold_charge"].isna()
    if not need.any():
        return df
    c_rate = -df["current_a"] / capacity_ah
    cold = (c_rate >= float(cfg.features.plating_c_rate)) & (df["temp_cell_c"] < float(cfg.features.plating_temp_c))
    df = df.copy()
    df.loc[need, "cold_charge"] = cold[need].astype(float)
    return df


def _neighbour_stats(feats: dict[str, CellFeatures], ctxs: dict[str, CellContext], cfg: Config) -> dict[str, dict[str, Any]]:
    """Robust z of corrected excess against the same pack and string; pack drift; noise ratio.

    z = (x - median(others)) / max(1.4826 * MAD(others), neighbour_mad_floor).
    """
    f = cfg.features
    out: dict[str, dict[str, Any]] = {}
    groups: dict[tuple[str, str], list[str]] = {}
    for cid, ctx in ctxs.items():
        groups.setdefault((ctx.pack_id, ctx.string_id), []).append(cid)
    for (_pack, _string), cells in groups.items():
        xs = {c: feats[c].values.get("corrected_excess", float("nan")) for c in cells}
        noise = {c: feats[c].values.get("noise_recent_mv", float("nan")) for c in cells}
        finite = {c: x for c, x in xs.items() if math.isfinite(x)}
        pack_med = float(np.median(list(finite.values()))) if finite else float("nan")
        frac_hi = float(np.mean([x > float(f.pack_drift_excess) for x in finite.values()])) if finite else 0.0
        drift = bool(len(finite) >= int(f.neighbour_min_cells) and pack_med > float(f.pack_drift_excess)
                     and frac_hi >= float(f.pack_drift_frac))
        for c in cells:
            others = [x for k, x in finite.items() if k != c]
            onoise = [x for k, x in noise.items() if k != c and math.isfinite(x)]
            rec: dict[str, Any] = {"n_neighbours": len(others), "available": False, "z": float("nan"),
                                   "diverges_from_neighbours": False, "pack_wide_drift": drift,
                                   "pack_median_excess": pack_med}
            if len(others) >= int(f.neighbour_min_cells) - 1 and math.isfinite(xs[c]):
                med = float(np.median(others))
                mad = 1.4826 * float(np.median(np.abs(np.asarray(others) - med)))
                z = (xs[c] - med) / max(mad, float(f.neighbour_mad_floor))
                rec.update({"available": True, "z": float(z), "diverges_from_neighbours": bool(abs(z) > float(f.neighbour_z))})
            ref_mode = str(f.noise_ratio_reference)
            ref = float("nan")
            if ref_mode == "neighbours" and onoise:
                ref = float(np.median(onoise))
            if not math.isfinite(ref):
                s = feats[c].series
                if s is not None and np.isfinite(s.noise).sum() >= 10:
                    k = max(5, int(0.1 * s.noise.size))
                    ref = float(np.nanmedian(s.noise[:k]))
            if not math.isfinite(ref) or ref <= 0:
                ref = float(f.noise_floor_mv)
            rec["noise_reference_mv"] = ref
            out[c] = rec
    return out


def _bootstrap_fault(cf: CellFeatures, nb: Mapping[str, Any], cfg: Config, has_ocv: bool, seed: int,
                     calib: Calibration | None) -> tuple[float, float] | None:
    """Resample the recent windows (with replacement), recompute recent features, re-score."""
    cs = cf.series
    nboot = int(cfg.confidence.bootstrap_n)
    if cs is None or nboot <= 0 or cs.t.size < 2:
        return None
    rng = np.random.default_rng(seed)
    n = cs.t.size
    idx_raw = rng.integers(n - cs.win_raw, n, size=(nboot, cs.win_raw))
    idx_corr = rng.integers(n - cs.win_corr, n, size=(nboot, cs.win_corr))
    rec = recent_features(cs, idx_raw, idx_corr, float(cfg.features.thermal_resistance_k_per_w))
    feats: dict[str, Any] = dict(cf.values)
    feats.update(rec)
    ref = nb.get("noise_reference_mv", float("nan"))
    feats["noise_ratio"] = rec["noise_recent_mv"] / ref if ref and math.isfinite(ref) else np.full(nboot, np.nan)
    sub = sub_scores(feats, cfg, has_ocv)
    p = probabilities(cause_scores(sub, cfg, has_ocv))
    fp = fault_probability(p)
    if calib is not None and calib.fault is not None:
        fp = np.asarray(calib.fault(fp), float)
    lo, hi = cfg.confidence.bootstrap_interval
    return float(np.quantile(fp, lo)), float(np.quantile(fp, hi))


def score_cell(ctx: CellContext, cf: CellFeatures, nb: Mapping[str, Any], cfg: Config, *, chem_warnings: list[str],
               calib: Calibration | None = None, seed: int = 0, bootstrap: bool = True) -> dict[str, Any]:
    """Score one cell whose features (and neighbour stats) are computed. Returns the cell report."""
    v = dict(cf.values)
    ref = nb.get("noise_reference_mv", float("nan"))
    nr = v.get("noise_recent_mv", float("nan"))
    v["noise_ratio"] = nr / ref if (math.isfinite(nr) and ref and math.isfinite(ref)) else float("nan")
    v["neighbour_z"] = nb.get("z", float("nan"))
    has_ocv = bool(cf.flags.get("has_ocv"))
    n = int(v.get("n_samples", 0))
    status = calib.status if calib is not None else "uncalibrated_scores"
    base = {"cell_id": ctx.cell_id, "pack_id": ctx.pack_id, "string_id": ctx.string_id, "mode": ctx.mode,
            "probability_status": status, "calibration_source": calib.source if calib else "none"}
    if n < int(cfg.windows.min_samples):
        conf = {"value": 0.0, "level": "Low", "status": "uncalibrated", "components": {}, "penalties": [
            {"name": "insufficient_data", "factor": 0.0, "reason": f"only {n} usable samples (need {cfg.windows.min_samples})"}]}
        probs = {k: (1.0 if k == "ambiguous_ageing" else 0.0) for k in CAUSES}
        return {**base, "verdict": "ambiguous_ageing", "verdict_label": "Insufficient data", "insufficient_data": True,
                "fault_probability": None, "fault_probability_interval": None, "cause_probabilities": probs,
                "confidence": conf, "headline": f"Insufficient data: {n} usable samples; collect more before diagnosing.",
                "explanation": [f"Only {n} usable samples; at least {cfg.windows.min_samples} are needed."],
                "decision_path": [], "tree_verdict": None, "features": cf.to_dict()["values"],
                "feature_availability": cf.availability, "flags": {}, "neighbours": dict(nb), "warnings": list(chem_warnings),
                "baseline_ref": ctx.baseline.summary()}
    sub = sub_scores(v, cfg, has_ocv)
    scores = cause_scores(sub, cfg, has_ocv)
    probs_arr = probabilities(scores)
    probs = {k: float(probs_arr[k]) for k in CAUSES}
    if calib is not None and calib.multiclass_t is not None:
        logits = np.log(np.clip(np.array([probs[k] for k in CAUSES]), 1e-12, 1.0)) / float(calib.multiclass_t)
        e = np.exp(logits - logits.max())
        probs = {k: float(x) for k, x in zip(CAUSES, e / e.sum(), strict=True)}
    fp = 1.0 - probs["healthy"] - probs["temperature_effect"]
    fp_uncal = float(fault_probability(probs_arr))
    if calib is not None and calib.fault is not None:
        fp = float(np.asarray(calib.fault(np.array([fp_uncal])), float)[0])
    verdict = max(probs, key=lambda k: probs[k])
    tree_verdict, path = decision_tree(v, cfg, has_ocv)
    interval = _bootstrap_fault(cf, nb, cfg, has_ocv, seed, calib) if bootstrap else None
    trust, trust_reasons = ctx.baseline.trust(float(cfg.baseline.trust_own), float(cfg.baseline.trust_population),
                                              float(cfg.baseline.trust_literature),
                                              float(cfg.baseline.population_trust_ea_literature_factor))
    chem_mismatch = any(w.startswith(("CHEMISTRY", "FORM-FACTOR", "CAPACITY")) for w in chem_warnings)
    sub_scalar = {k: float(np.asarray(x)) for k, x in sub.items()}
    conf = compute_confidence(
        probs, v, cfg, baseline_trust=trust, trust_reasons=trust_reasons, mode=ctx.mode,
        has_pack_temp=bool(cf.flags.get("has_pack_temp")), n_neighbours=int(nb.get("n_neighbours", 0)),
        sensor_fault=bool(cf.flags.get("sensor_fault_suspected")), chemistry_mismatch=chem_mismatch,
        sub=sub_scalar, verdict=verdict, fault_interval=interval, calibrator=calib.confidence if calib else None)
    flags: dict[str, Any] = {"sensor_fault_suspected": bool(cf.flags.get("sensor_fault_suspected")),
             "sensor_reasons": cf.flags.get("sensor_reasons", []),
             "diverges_from_neighbours": bool(nb.get("diverges_from_neighbours")),
             "pack_wide_drift": bool(nb.get("pack_wide_drift")), "chemistry_mismatch": chem_mismatch,
             "has_ocv": has_ocv, "has_soh": bool(cf.flags.get("has_soh")),
             "has_pack_temp": bool(cf.flags.get("has_pack_temp")), "is_fault": verdict not in NON_FAULT_CAUSES}
    r0 = ctx.baseline.r0_25c_ohm.value or float("nan")
    text = explain(verdict, probs, v, flags, conf, r0, cf.availability, nb)
    report = {
        **base,
        "verdict": verdict,
        "verdict_label": CAUSE_LABELS[verdict],
        "is_fault": verdict not in NON_FAULT_CAUSES,
        "insufficient_data": False,
        "fault_probability": fp,
        "fault_probability_uncalibrated": fp_uncal,
        "fault_probability_interval": list(interval) if interval else None,
        "cause_probabilities": probs,
        "confidence": conf,
        "headline": headline(verdict, fp, interval, status, conf),
        "explanation": text,
        "decision_path": path,
        "tree_verdict": tree_verdict,
        "tree_agrees": tree_verdict == verdict,
        "sub_scores": sub_scalar,
        "features": {k: (None if isinstance(x, float) and not math.isfinite(x) else x) for k, x in v.items()},
        "feature_availability": dict(cf.availability),
        "flags": flags,
        "neighbours": {k: (None if isinstance(x, float) and not math.isfinite(x) else x) for k, x in nb.items()},
        "notes": list(cf.notes),
        "extraction": ctx.extraction,
        "warnings": list(chem_warnings),
        "baseline_ref": {"baseline_id": ctx.baseline.baseline_id, "kind": ctx.baseline.kind,
                         "r0_25c_ohm": ctx.baseline.r0_25c_ohm.value, "ea_j_per_mol": ctx.baseline.ea_j_per_mol.value,
                         "ea_source": ctx.baseline.ea_j_per_mol.source, "trust": trust},
    }
    return report


def _features_worker(args: tuple[pd.DataFrame, dict[str, Any], dict[str, Any], str]) -> CellFeatures:
    samples, bl_json, cfg_dict, mode = args
    return compute_features(samples, Baseline(**bl_json), Config(cfg_dict), mode)


def pack_baseline(samples: Mapping[str, pd.DataFrame], cfg: Config, meta: TargetMeta, rkind: str, seed: int = 0) -> Baseline:
    """Population baseline from the early life of every cell in the pack (median across cells).

    Useful when no open dataset matches the chemistry (e.g. sodium-ion): a cell that was bad
    from day one shows up against its pack, which an own baseline cannot see. Only meaningful
    for matched cells (same model and batch). Ea is fitted within cells; if early life lacks
    temperature spread the Ea window is widened as for own baselines.
    """
    from ..baseline.fit import fit_with_widening
    from ..baseline.model import Provenance

    rows = []
    for cid, s in samples.items():
        rows.append(pd.DataFrame({"cell_id": cid, "order": s["t"].to_numpy(float), "temp_c": s["temp_c"].to_numpy(float),
                                  "r_total": s["r_total"].to_numpy(float), "r_ohmic": s["r_ohmic"].to_numpy(float),
                                  "r_slow": s["r_slow"].to_numpy(float), "capacity_ah": np.nan}))
    tab = pd.concat(rows, ignore_index=True)
    prov = Provenance(dataset="pack early-life median (this data set)", fitting_method="pack median",
                      notes=["baseline built from the diagnosed pack itself; a pack-wide fault is invisible to it"])
    return fit_with_widening(tab, cfg, baseline_id="pack-median", chemistry=meta.chemistry or "unknown",
                             form_factor=meta.form_factor or "unknown", nominal_capacity_ah=meta.capacity_ah,
                             resistance_kind=rkind, provenance=prov, seed=seed)


def build_contexts(frame: pd.DataFrame, cfg: Config, baseline: Baseline | Mapping[str, Baseline] | str | None,
                   meta: TargetMeta, seed: int = 0) -> tuple[dict[str, CellContext], dict[str, Any]]:
    """Validate, derive cold-charge flags, detect modes, build sample tables and baselines.

    ``baseline``: a Baseline (applied to every cell), a mapping cell_id -> Baseline, ``None`` /
    ``"own"`` (each cell's own early life) or ``"pack"`` (pack early-life median)."""
    clean, vrep = validate(frame, cfg)
    cap = meta.capacity_ah
    if baseline == "own":
        baseline = None
    if cap is None and isinstance(baseline, Baseline):
        cap = baseline.nominal_capacity_ah
    clean = derive_cold_charge(clean, cfg, cap)
    ctxs: dict[str, CellContext] = {}
    for cid, g in clean.groupby("cell_id", sort=True):
        mode = detect_mode(g, cfg)
        extraction: dict[str, float] = {}
        if mode == "pulse_mode":
            samples, extraction = pulse_cell_to_samples(g, cfg, cap)
            rkind = "pulse_1s"
        else:
            samples = cycle_cell_to_samples(g)
            rkind = meta.resistance_kind or "cycle"
        if baseline == "pack":
            bl = None  # filled below
        elif isinstance(baseline, Baseline):
            bl = baseline
        elif isinstance(baseline, Mapping) and str(cid) in baseline:
            bl = baseline[str(cid)]
        else:
            bl = fit_own_baseline(samples, cfg, cell_id=str(cid), chemistry=meta.chemistry or "unknown",
                                  form_factor=meta.form_factor or "unknown", nominal_capacity_ah=cap,
                                  resistance_kind=rkind, seed=seed)
        ctxs[str(cid)] = CellContext(str(cid), str(g["pack_id"].iloc[0]), str(g["string_id"].iloc[0]), mode, samples, bl,  # type: ignore[arg-type]
                                     extraction)
    if baseline == "pack" and ctxs:
        by_pack: dict[str, list[str]] = {}
        for cid, ctx in ctxs.items():
            by_pack.setdefault(ctx.pack_id, []).append(cid)
        for pack, cids in by_pack.items():
            rk = "pulse_1s" if ctxs[cids[0]].mode == "pulse_mode" else (meta.resistance_kind or "cycle")
            pbl = pack_baseline({c: ctxs[c].samples for c in cids}, cfg, meta, rk, seed)
            pbl = pbl.model_copy(update={"baseline_id": f"pack-median-{pack}"})
            for c in cids:
                ctxs[c].baseline = pbl
    return ctxs, vrep.to_dict()


def diagnose(frame: pd.DataFrame, baseline: Baseline | Mapping[str, Baseline] | str | None = None, cfg: Config | None = None,
             *, meta: TargetMeta | None = None, calibration: Calibration | None = None, seed: int = 0, n_jobs: int = 1,
             bootstrap: bool = True) -> dict[str, Any]:
    """Diagnose every cell in ``frame`` (common schema). ``baseline=None`` fits each cell's
    own early-life baseline. Returns a JSON-serialisable report (schema ``celldx-report/1``)."""
    cfg = cfg or load_config()
    meta = meta or TargetMeta()
    ctxs, vrep = build_contexts(frame, cfg, baseline, meta, seed)
    feats: dict[str, CellFeatures] = {}
    if n_jobs > 1 and len(ctxs) > 1:
        args = [(c.samples, c.baseline.model_dump(mode="json"), cfg.to_dict(), c.mode) for c in ctxs.values()]
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for cid, cf in zip(ctxs, ex.map(_features_worker, args), strict=True):
                feats[cid] = cf
    else:
        for cid, c in ctxs.items():
            feats[cid] = compute_features(c.samples, c.baseline, cfg, c.mode)
    nbs = _neighbour_stats(feats, ctxs, cfg)
    warnings_by_bl: dict[str, list[str]] = {}
    cells = []
    for i, (cid, c) in enumerate(ctxs.items()):
        key = c.baseline.baseline_id
        if key not in warnings_by_bl:
            r_typ = float(np.nanmedian(c.samples["r_total"].to_numpy(float)[: max(3, len(c.samples) // 10)])) if len(c.samples) else None
            warnings_by_bl[key] = check_compatibility(c.baseline, cfg, chemistry=meta.chemistry, form_factor=meta.form_factor,
                                                      capacity_ah=meta.capacity_ah, r_typical_ohm=r_typ,
                                                      resistance_kind=meta.resistance_kind)
        cells.append(score_cell(c, feats[cid], nbs[cid], cfg, chem_warnings=warnings_by_bl[key], calib=calibration,
                                seed=seed + i, bootstrap=bootstrap))
    baselines = {c.baseline.baseline_id: c.baseline for c in ctxs.values()}
    return {
        "schema": REPORT_SCHEMA,
        "celldx_version": __version__,
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "config_hash": cfg.hash(),
        "profile": cfg.profile,
        "probability_status": calibration.status if calibration else "uncalibrated_scores",
        "calibration_source": calibration.source if calibration else "none",
        "n_cells": len(cells),
        "n_faults": int(sum(1 for c in cells if c.get("is_fault"))),
        "validation": vrep,
        "baselines": {k: (b.summary() if b.kind == "own" else b.model_dump(mode="json")) for k, b in baselines.items()},
        "baseline_warnings": {k: w for k, w in warnings_by_bl.items() if w},
        "cells": sorted(cells, key=lambda c: -(c.get("fault_probability") or 0.0)),
    }


def diagnose_latest(history: pd.DataFrame, baseline: Baseline | None = None, cfg: Config | None = None, *,
                    window: int | None = None, meta: TargetMeta | None = None, calibration: Calibration | None = None,
                    neighbours: pd.DataFrame | None = None, seed: int = 0, bootstrap: bool = True) -> dict[str, Any]:
    """Streaming-friendly call: score one cell's latest window.

    ``history`` holds one cell's samples (or raw rows). Only the last ``window`` samples are
    used (default ``features.tind_window`` = 365 for daily data). ``neighbours``, if given,
    holds other cells' rows over the same window, used only for the neighbour comparison.
    Returns the cell report dict.
    """
    cfg = cfg or load_config()
    h = history.sort_values("t")
    w = int(window or 2 * int(cfg.features.tind_window))
    if h["cell_id"].nunique() != 1:
        raise ValueError("diagnose_latest expects exactly one cell in history")
    cid = str(h["cell_id"].iloc[0])
    frame = h.iloc[-w:]
    if neighbours is not None:
        frame = pd.concat([frame, neighbours[neighbours["cell_id"] != cid]], ignore_index=True)
    rep = diagnose(frame, baseline, cfg, meta=meta, calibration=calibration, seed=seed, bootstrap=bootstrap)
    return next(c for c in rep["cells"] if c["cell_id"] == cid)
