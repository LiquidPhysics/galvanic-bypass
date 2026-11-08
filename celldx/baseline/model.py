"""Versioned :class:`Baseline` with a provenance block.

A baseline is what "new" looks like for a cell: its resistance at 25 C and SOC_ref, the
activation energy used for temperature correction, the resistance-versus-SOC map, the
new-cell OCV curve and capacity, and the cell-to-cell spread of each parameter.

Every number carries its source. Allowed sources:
``fitted_own``, ``fitted_population``, ``fitted_within_cell``, ``fitted_between_cell``,
``literature_default``, ``assumed_flat``, ``simulation_truth`` and ``user_supplied``.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .. import __version__

BASELINE_SCHEMA = "celldx-baseline/1"

Kind = Literal["own", "population", "literature"]


class FileRecord(BaseModel):
    name: str
    sha256: str
    bytes: int | None = None


class Provenance(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset: str
    version: str | None = None
    doi: str | None = None
    url: str | None = None
    licence: str | None = None
    files: list[FileRecord] = Field(default_factory=list)
    cells: list[str] = Field(default_factory=list)
    cycle_ranges: dict[str, list[float]] = Field(default_factory=dict)
    fitting_method: str = ""
    fit_date: str = Field(default_factory=lambda: _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"))
    celldx_version: str = __version__
    config_hash: str | None = None
    notes: list[str] = Field(default_factory=list)
    data_cannot_support: list[str] = Field(default_factory=list)


class Param(BaseModel):
    """A fitted scalar with source, interval and population spread."""

    value: float | None
    source: str
    ci: list[float] | None = None
    population_std: float | None = None
    n: int | None = None
    note: str | None = None


class Curve(BaseModel):
    x: list[float]
    y: list[float]
    source: str
    note: str | None = None


class Baseline(BaseModel):
    """Stored baseline. Serialises to JSON or YAML with :meth:`save`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = BASELINE_SCHEMA
    baseline_id: str
    kind: Kind
    chemistry: str
    form_factor: str
    nominal_capacity_ah: float | None
    resistance_kind: str  # eis_re_rct | dcir_onset | pulse_1s | sim_daily
    soc_ref: float = 0.5
    r0_25c_ohm: Param
    r0_ohmic_ohm: Param
    r0_slow_ohm: Param
    ea_j_per_mol: Param
    ea_ohmic_j_per_mol: Param
    ea_slow_j_per_mol: Param
    soh0_ah: Param
    r_soc_map: Curve
    ocv_soc_curve: Curve | None = None
    provenance: Provenance
    warnings: list[str] = Field(default_factory=list)

    # ------------------------------------------------------------------ trust
    def trust(self, trust_own: float, trust_population: float, trust_literature: float,
              ea_literature_factor: float) -> tuple[float, list[str]]:
        """Baseline trust used by confidence, with the reasons.

        own 1.0, open-data population 0.85, literature default 0.70 (configurable). A
        population baseline whose Ea is a literature default is multiplied by
        ``ea_literature_factor`` because the temperature correction is then unverified.
        """
        reasons: list[str] = []
        if self.kind == "own" or self.r0_25c_ohm.source in ("fitted_own", "simulation_truth"):
            trust = trust_own
            reasons.append(f"own early-life baseline (trust {trust_own})")
        elif self.r0_25c_ohm.source == "literature_default" or self.kind == "literature":
            trust = trust_literature
            reasons.append(f"literature-default baseline (trust {trust_literature})")
        else:
            trust = trust_population
            reasons.append(f"open-data population baseline (trust {trust_population})")
            if self.ea_j_per_mol.source == "literature_default":
                trust *= ea_literature_factor
                reasons.append(f"Ea is a literature default (x{ea_literature_factor})")
        return trust, reasons

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        data = self.model_dump(mode="json")
        if p.suffix in (".yaml", ".yml"):
            p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        else:
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> Baseline:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        data: dict[str, Any] = yaml.safe_load(text) if p.suffix in (".yaml", ".yml") else json.loads(text)
        return cls(**data)

    def summary(self) -> dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "kind": self.kind,
            "chemistry": self.chemistry,
            "form_factor": self.form_factor,
            "nominal_capacity_ah": self.nominal_capacity_ah,
            "resistance_kind": self.resistance_kind,
            "r0_25c_ohm": self.r0_25c_ohm.value,
            "r0_source": self.r0_25c_ohm.source,
            "ea_j_per_mol": self.ea_j_per_mol.value,
            "ea_source": self.ea_j_per_mol.source,
            "ea_ci": self.ea_j_per_mol.ci,
            "dataset": self.provenance.dataset,
            "fit_date": self.provenance.fit_date,
            "data_cannot_support": self.provenance.data_cannot_support,
            "warnings": self.warnings,
        }
