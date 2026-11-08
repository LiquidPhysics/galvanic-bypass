"""Common data schema (one row per observation per cell) and validation.

Columns (units)
---------------
=================  ======  ===========================================================
t                  s|cycle  timestamp in seconds, or a cycle / day index (numeric)
cell_id            text     stable cell name
pack_id            text     pack name (optional, default "pack0")
string_id          text     series-string name (optional, default "string0")
voltage_v          V        terminal voltage
current_a          A        positive = discharge, negative = charge
temp_cell_c        C        cell temperature
temp_pack_c        C        pack / enclosure temperature (optional)
soc                0..1     state of charge
soh                0..1     capacity relative to the cell's new capacity
ocv_mv             mV       deviation of rest OCV from the baseline OCV curve (optional)
resistance_ohm     ohm      BMS-reported or EIS / cycle-level total resistance
r_ohmic_ohm        ohm      ohmic component when known (EIS Re, or pulse 10 ms)
r_slow_ohm         ohm      slow component when known (EIS Rct, or pulse 1 s - 10 ms)
cold_charge        0/1      derived: hard charge at low temperature in this sample
ah_throughput      Ah       cumulative charge throughput
cycle_count        -        cumulative equivalent cycles
=================  ======  ===========================================================

Validation never drops rows. Values outside sensor ranges are masked to NaN in the
cleaned frame (the original is kept in the report), resistance spikes are flagged in
``qc_flags`` and excluded from features, and every action is logged in a
:class:`ValidationReport`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .config import Config

NUMERIC_COLUMNS: list[str] = [
    "t",
    "voltage_v",
    "current_a",
    "temp_cell_c",
    "temp_pack_c",
    "soc",
    "soh",
    "ocv_mv",
    "resistance_ohm",
    "r_ohmic_ohm",
    "r_slow_ohm",
    "cold_charge",
    "ah_throughput",
    "cycle_count",
]
ID_COLUMNS: list[str] = ["cell_id", "pack_id", "string_id"]
ALL_COLUMNS: list[str] = ["t", *ID_COLUMNS, *NUMERIC_COLUMNS[1:]]

UNITS: dict[str, str] = {
    "t": "s or cycle index",
    "voltage_v": "V",
    "current_a": "A (+ discharge)",
    "temp_cell_c": "degC",
    "temp_pack_c": "degC",
    "soc": "fraction 0..1",
    "soh": "fraction of new capacity",
    "ocv_mv": "mV deviation from baseline OCV curve",
    "resistance_ohm": "ohm",
    "r_ohmic_ohm": "ohm",
    "r_slow_ohm": "ohm",
    "cold_charge": "0/1",
    "ah_throughput": "Ah",
    "cycle_count": "cycles",
}


class Observation(BaseModel):
    """One observation of one cell (pydantic model used for JSON / streaming input)."""

    model_config = ConfigDict(extra="forbid")

    t: float
    cell_id: str
    pack_id: str = "pack0"
    string_id: str = "string0"
    voltage_v: float | None = None
    current_a: float | None = None
    temp_cell_c: float | None = None
    temp_pack_c: float | None = None
    soc: float | None = Field(default=None)
    soh: float | None = None
    ocv_mv: float | None = None
    resistance_ohm: float | None = None
    r_ohmic_ohm: float | None = None
    r_slow_ohm: float | None = None
    cold_charge: float | None = None
    ah_throughput: float | None = None
    cycle_count: float | None = None


@dataclass
class ValidationIssue:
    code: str
    severity: str  # info | warning | error
    detail: str
    cell_id: str | None = None
    column: str | None = None
    n_rows: int = 0
    action: str = "reported"

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class ValidationReport:
    n_rows: int = 0
    n_cells: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    masked_values: dict[str, int] = field(default_factory=dict)
    flagged_rows: int = 0

    def add(self, *args: Any, **kwargs: Any) -> None:
        self.issues.append(ValidationIssue(*args, **kwargs))

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cells": self.n_cells,
            "ok": self.ok,
            "flagged_rows": self.flagged_rows,
            "masked_values": self.masked_values,
            "issues": [i.to_dict() for i in self.issues],
        }


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Validate a list of JSON records with :class:`Observation` and build a frame."""
    rows = [Observation(**r).model_dump() for r in records]
    return pd.DataFrame(rows, columns=ALL_COLUMNS)


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with every schema column present and numeric columns as float."""
    out = df.copy()
    for col in NUMERIC_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    if "cell_id" not in out.columns:
        raise ValueError("input must contain a cell_id column")
    out["cell_id"] = out["cell_id"].astype(str)
    out["pack_id"] = out["pack_id"].astype(str) if "pack_id" in out.columns else "pack0"
    out["string_id"] = out["string_id"].astype(str) if "string_id" in out.columns else "string0"
    out["pack_id"] = out["pack_id"].replace({"nan": "pack0", "None": "pack0"})
    out["string_id"] = out["string_id"].replace({"nan": "string0", "None": "string0"})
    return out


def validate(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate units, time order, gaps, sensor ranges and resistance spikes.

    Returns the cleaned frame (sorted by cell and t, values outside sensor ranges masked to
    NaN, a boolean ``qc_ok`` and a text ``qc_flags`` column) and a report of every action.
    """
    v = cfg.validation
    rep = ValidationReport()
    raw_numeric = {c: df[c] for c in df.columns if c in NUMERIC_COLUMNS}
    out = coerce(df)
    for col, series in raw_numeric.items():
        bad = series.notna() & out[col].isna()
        if bad.any():
            rep.add("non_numeric", "warning", f"{int(bad.sum())} non-numeric values set to NaN", column=col,
                    n_rows=int(bad.sum()), action="masked")
    rep.n_rows = len(out)
    rep.n_cells = out["cell_id"].nunique()
    if out["t"].isna().any():
        n = int(out["t"].isna().sum())
        rep.add("missing_time", "error", f"{n} rows have no time stamp; they are flagged and excluded", column="t",
                n_rows=n, action="flagged")
    flags = pd.Series([""] * len(out), index=out.index, dtype=object)

    # --- unit heuristics (reported, never auto-converted) --------------------------------
    def _med(col: str) -> float:
        s = out[col].dropna()
        return float(s.median()) if len(s) else float("nan")

    if _med("temp_cell_c") > 150:
        rep.add("unit_temperature", "error", "temp_cell_c median above 150: looks like kelvin", column="temp_cell_c")
    if out["soc"].dropna().max() > 1.5 if out["soc"].notna().any() else False:
        rep.add("unit_soc", "error", "soc above 1.5: looks like percent, expected 0..1", column="soc")
    if out["soh"].dropna().max() > 1.5 if out["soh"].notna().any() else False:
        rep.add("unit_soh", "error", "soh above 1.5: looks like percent, expected 0..1", column="soh")
    if _med("resistance_ohm") > 1.0:
        rep.add("unit_resistance", "warning", "resistance_ohm median above 1 ohm: check it is not milliohm",
                column="resistance_ohm")
    if out["ocv_mv"].notna().any() and float(out["ocv_mv"].abs().median()) > 1000:
        rep.add("unit_ocv", "error", "ocv_mv must be the deviation from the baseline OCV curve in mV, "
                "not the absolute OCV", column="ocv_mv")

    # --- sensor ranges: mask out-of-range values -----------------------------------------
    ranges = {
        "voltage_v": tuple(v.voltage_range_v),
        "temp_cell_c": tuple(v.temp_range_c),
        "temp_pack_c": tuple(v.temp_range_c),
        "soc": tuple(v.soc_range),
        "soh": tuple(v.soh_range),
        "resistance_ohm": tuple(v.resistance_range_ohm),
        "r_ohmic_ohm": tuple(v.resistance_range_ohm),
        "r_slow_ohm": tuple(v.resistance_range_ohm),
        "current_a": (-float(v.current_abs_max_a), float(v.current_abs_max_a)),
        "ocv_mv": (-float(v.ocv_deviation_abs_max_mv), float(v.ocv_deviation_abs_max_mv)),
    }
    for col, (lo, hi) in ranges.items():
        s = out[col]
        bad = s.notna() & ((s < lo) | (s > hi) | ~np.isfinite(s))
        n_bad = int(bad.sum())
        if n_bad:
            examples = ", ".join(f"{x:.4g}" for x in s[bad].head(3))
            for cid, cnt in out.loc[bad, "cell_id"].value_counts().items():
                rep.add("sensor_range", "warning", f"{cnt} values of {col} outside [{lo}, {hi}] masked "
                        f"(e.g. {examples})", cell_id=str(cid), column=col, n_rows=int(cnt), action="masked")
            out.loc[bad, col] = np.nan
            flags[bad] = flags[bad] + f"range:{col};"
            rep.masked_values[col] = rep.masked_values.get(col, 0) + n_bad

    # --- per-cell ordering, duplicates, gaps, spikes -----------------------------------
    out = out.reset_index(drop=True)
    flags = flags.reset_index(drop=True)
    order = np.lexsort((out["t"].to_numpy(), out["cell_id"].to_numpy()))
    was_sorted = bool(np.all(order == np.arange(len(out))))
    if not was_sorted:
        # Report back-steps per cell before sorting.
        for cid, g in out.groupby("cell_id", sort=False):
            back = int((np.diff(g["t"].to_numpy()) < 0).sum())
            if back:
                rep.add("non_monotonic_time", "warning", f"{back} backwards time steps; rows sorted by t",
                        cell_id=str(cid), column="t", n_rows=back, action="sorted")
    out = out.iloc[order].reset_index(drop=True)
    flags = flags.iloc[order].reset_index(drop=True)
    qc_ok = np.ones(len(out), dtype=bool)
    qc_ok[out["t"].isna().to_numpy()] = False
    for cid, idx in out.groupby("cell_id", sort=False).indices.items():
        t = out.loc[idx, "t"].to_numpy()
        dt = np.diff(t)
        dup = np.where(dt == 0)[0]
        if dup.size:
            rep.add("duplicate_time", "info", f"{dup.size} rows share a time stamp with the previous row",
                    cell_id=str(cid), column="t", n_rows=int(dup.size))
        pos = dt[dt > 0]
        if pos.size:
            med = float(np.median(pos))
            gaps = np.where(dt > float(v.gap_factor) * med)[0]
            if gaps.size:
                rep.add("time_gap", "info", f"{gaps.size} gaps longer than {v.gap_factor} x median dt "
                        f"({med:.4g}); largest {float(dt.max()):.4g}", cell_id=str(cid), column="t",
                        n_rows=int(gaps.size))
        # Resistance spikes: robust z of log R against a centred rolling median (step-safe).
        r = out.loc[idx, "resistance_ohm"].to_numpy()
        ok = np.isfinite(r) & (r > 0)
        if ok.sum() >= 11:
            # Normalise by a literature Arrhenius factor first, so a cold day is not a "spike".
            temp = out.loc[idx, "temp_cell_c"].to_numpy(float)
            ea_lit = float(cfg.physics.ea_literature_j_per_mol)
            rg = float(cfg.physics.rg_j_per_mol_k)
            tcorr = np.where(np.isfinite(temp), ea_lit / rg * (1.0 / (temp + 273.15) - 1.0 / float(cfg.physics.t25_k)), 0.0)
            lr = pd.Series(np.log(np.where(ok, r, np.nan)) - tcorr)
            roll = lr.rolling(11, center=True, min_periods=5).median().to_numpy()
            d1 = np.diff(lr.dropna().to_numpy())
            scale = 1.4826 * float(np.median(np.abs(d1 - np.median(d1)))) / np.sqrt(2) if d1.size else 0.0
            scale = max(scale, 1e-6)
            z = np.abs(lr.to_numpy() - roll) / scale
            spike = np.nan_to_num(z, nan=0.0) > float(v.outlier_mad_z)
            if spike.any():
                rows = np.asarray(idx)[spike]
                qc_ok[rows] = False
                flags.iloc[rows] = flags.iloc[rows] + "spike:resistance_ohm;"
                rep.add("resistance_spike", "warning", f"{int(spike.sum())} resistance spikes (robust z > "
                        f"{v.outlier_mad_z}) flagged and excluded from features", cell_id=str(cid),
                        column="resistance_ohm", n_rows=int(spike.sum()), action="flagged")
    out["qc_ok"] = qc_ok
    out["qc_flags"] = flags.to_numpy()
    rep.flagged_rows = int((~qc_ok).sum())
    for cid, g in out.groupby("cell_id"):
        if g["resistance_ohm"].notna().sum() == 0 and not (g["voltage_v"].notna().any() and g["current_a"].notna().any()):
            rep.add("no_resistance", "error", "no resistance and no voltage+current to extract it", cell_id=str(cid))
    return out, rep
