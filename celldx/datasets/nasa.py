"""NASA Ames PCoE Li-ion aging dataset adapter (tested against the real files).

File layout (verified on the real .mat files, 2026-11-08): top-level variable ``Bxxxx`` with
field ``cycle``, a struct array whose elements have ``type`` ('charge' | 'discharge' |
'impedance'), ``ambient_temperature`` (C), ``time`` (MATLAB date vector) and ``data``.
Discharge data: Voltage_measured, Current_measured (negative = discharge in NASA's sign
convention), Temperature_measured, Current_load, Voltage_load, Time, Capacity. Impedance
data: Sense_current, Battery_current, Current_ratio, Battery_impedance, Rectified_Impedance
(complex arrays; note the capital I on the real files, the README says Rectified_impedance),
Re, Rct (real scalars; some late cycles hold corrupted values such as -9.7e14).

Two resistance sources (both cycle_mode):

* ``eis``: one row per impedance cycle, ``r_ohmic = Re``, ``r_slow = Rct``,
  ``resistance = Re + Rct``. The EIS sweep has no temperature channel, so ``temp_cell_c`` is
  the logged ambient (flagged in notes).
* ``dcir``: one row per discharge cycle, resistance from the rest-to-load voltage step at the
  start of discharge, ``(V_rest - V_loaded) / |dI|`` (about 10 to 19 s after the step at the
  NASA logging rate, so it includes slow polarisation), with the measured cell temperature.

SOH = capacity / soh0, where soh0 is the median of the first three discharge capacities of
that cell (rated 2.0 Ah; actual first capacities 1.0 to 2.04 Ah, recorded per file).
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RATED_AH = 2.0
CHEMISTRY = "li_ion_nasa_18650"  # cathode chemistry is not documented in the README
FORM_FACTOR = "18650"


def _real(x: Any) -> tuple[float, bool]:
    """Real part of a scalar that may be complex; second value True if the imaginary part was material."""
    try:
        z = complex(np.asarray(x).ravel()[0]) if np.size(x) else complex("nan")
    except (TypeError, ValueError):
        return float("nan"), False
    material = abs(z.imag) > 1e-6 * max(abs(z.real), 1e-12)
    return float(z.real), bool(material)


def _datevec(v: Any) -> float:
    """MATLAB date vector -> POSIX seconds (UTC assumed)."""
    a = np.asarray(v, float).ravel()
    if a.size < 6 or not np.all(np.isfinite(a)):
        return float("nan")
    sec = float(a[5])
    d = _dt.datetime(int(a[0]), int(a[1]), int(a[2]), int(a[3]), int(a[4]), int(sec), int((sec % 1) * 1e6),
                     tzinfo=_dt.UTC)
    return d.timestamp()


def load_cycles(path: str | Path) -> tuple[str, list[Any]]:
    """Load one NASA .mat file with scipy (struct_as_record=False, squeeze_me=True)."""
    from scipy.io import loadmat

    m = loadmat(str(path), struct_as_record=False, squeeze_me=True)
    keys = [k for k in m if not k.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"{path}: expected one top-level variable, found {keys}")
    cyc = m[keys[0]].cycle
    cyc = list(np.atleast_1d(cyc))
    return keys[0], cyc


def cell_table(path: str | Path, source: str = "eis") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Common-schema rows for one cell plus metadata (soh0, notes, counts)."""
    name, cycles = load_cycles(path)
    notes: list[str] = []
    caps: list[tuple[int, float]] = []
    n_dis = 0
    for i, c in enumerate(cycles):
        if c.type == "discharge":
            n_dis += 1
            cap = getattr(c.data, "Capacity", np.nan)
            cv, _ = _real(cap)
            caps.append((i, cv))
    good = [cv for _, cv in caps if np.isfinite(cv) and cv > 0]
    soh0 = float(np.median(good[:3])) if good else float("nan")
    if abs(soh0 - RATED_AH) / RATED_AH > 0.1:
        notes.append(f"first discharge capacities median {soh0:.3f} Ah differs from rated {RATED_AH} Ah by more than 10%")
    rows: list[dict[str, Any]] = []
    cap_idx = np.array([i for i, _ in caps])
    cap_val = np.array([v for _, v in caps])
    t0 = None
    n_complex = 0
    dis_count = 0
    ah = 0.0
    for i, c in enumerate(cycles):
        tsec = _datevec(c.time)
        if t0 is None and np.isfinite(tsec):
            t0 = tsec
        if c.type == "discharge":
            dis_count += 1
            cv = cap_val[cap_idx == i][0] if (cap_idx == i).any() else np.nan
            ah += cv if np.isfinite(cv) else 0.0
        prev = cap_val[cap_idx <= i]
        soh = (prev[np.isfinite(prev) & (prev > 0)][-1] / soh0) if np.isfinite(prev).any() and soh0 > 0 else np.nan
        base = {"cell_id": name, "pack_id": "nasa", "string_id": name, "t_unix": tsec, "op_index": i,
                "cycle_count": float(dis_count), "ah_throughput": ah, "soh": float(soh) if np.isfinite(soh) else np.nan,
                "ambient_c": float(c.ambient_temperature), "capacity_ah": float(cv) if c.type == "discharge" else np.nan}
        if source == "eis" and c.type == "impedance":
            re, m1 = _real(getattr(c.data, "Re", np.nan))
            rct, m2 = _real(getattr(c.data, "Rct", np.nan))
            n_complex += int(m1) + int(m2)
            rows.append({**base, "r_ohmic_ohm": re, "r_slow_ohm": rct, "resistance_ohm": re + rct,
                         "temp_cell_c": float(c.ambient_temperature), "temp_pack_c": float(c.ambient_temperature),
                         "current_a": np.nan, "voltage_v": np.nan})
        elif source == "dcir" and c.type == "discharge":
            d = c.data
            I = np.asarray(d.Current_measured, float).ravel()
            V = np.asarray(d.Voltage_measured, float).ravel()
            T = np.asarray(d.Temperature_measured, float).ravel()
            if I.size < 5:
                continue
            k = int(np.argmax(np.abs(I) > 0.5))
            if k == 0:
                continue
            dI = abs(I[k] - I[k - 1])
            r = (V[k - 1] - V[k]) / dI if dI > 0.3 else np.nan
            rows.append({**base, "resistance_ohm": r, "r_ohmic_ohm": np.nan, "r_slow_ohm": np.nan,
                         "temp_cell_c": float(T[k - 1]), "temp_pack_c": float(c.ambient_temperature),
                         "current_a": float(-I[k]), "voltage_v": float(V[k - 1])})
    df = pd.DataFrame(rows)
    if n_complex:
        notes.append(f"{n_complex} Re/Rct values had a material imaginary part; the real part was kept")
    if source == "eis":
        notes.append("EIS rows carry the logged ambient temperature (the sweep has no temperature channel)")
    if not df.empty:
        df["t"] = (df["t_unix"] - (t0 or 0.0)) / 86400.0  # days since the first operation
        df["t"] = df["t"].where(np.isfinite(df["t"]), df["op_index"].astype(float))
    meta = {"cell": name, "soh0_ah": soh0, "rated_ah": RATED_AH, "n_operations": len(cycles), "n_discharge": n_dis,
            "n_rows": len(df), "notes": notes, "ambient_c": sorted({float(c.ambient_temperature) for c in cycles})}
    return df, meta


def find_files(root: str | Path) -> dict[str, Path]:
    """Map cell name -> .mat path under ``root`` (first occurrence; duplicates are identical)."""
    out: dict[str, Path] = {}
    for p in sorted(Path(root).rglob("B00*.mat")):
        out.setdefault(p.stem, p)
    return out


def load(root: str | Path, cells: list[str] | None = None, source: str = "eis") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Common-schema frame for the requested cells (all if None) plus per-cell metadata."""
    files = find_files(root)
    if cells:
        missing = [c for c in cells if c not in files]
        if missing:
            raise FileNotFoundError(f"cells not found under {root}: {missing}")
        files = {c: files[c] for c in cells}
    frames, metas = [], {}
    for c, p in files.items():
        df, meta = cell_table(p, source)
        meta["file"] = str(p)
        frames.append(df)
        metas[c] = meta
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), metas


# --------------------------------------------------------------------------- baselines from NASA
def square_wave_soc_map(root: str | Path, cells: tuple[str, ...] = ("B0025", "B0026", "B0027", "B0028"),
                        n_early: int = 5, soc_ref: float = 0.5) -> tuple[Any, list[dict[str, Any]]]:
    """Resistance-versus-SOC map from the 0.05 Hz, 4 A square-wave discharges of B0025-B0028.

    Each on/off transition between consecutive samples (about 10 s apart) gives
    ``R = |dV| / |dI|``; SOC is coulomb-counted from the start of the discharge against that
    cycle's capacity. Early cycles only. Returns a Curve (normalised at ``soc_ref``) and the
    file records used. Note: about 10 s resistance (includes slow polarisation).
    """
    from ..baseline.fit import fit_soc_map
    from .registry import sha256_file

    files = find_files(root)
    socs, rs, used = [], [], []
    for c in cells:
        if c not in files:
            continue
        _, cyc = load_cycles(files[c])
        used.append({"name": files[c].name, "sha256": sha256_file(files[c])})
        dis = [x for x in cyc if x.type == "discharge"][:n_early]
        for x in dis:
            d = x.data
            I = np.asarray(d.Current_measured, float).ravel()
            V = np.asarray(d.Voltage_measured, float).ravel()
            t = np.asarray(d.Time, float).ravel()
            cap = float(np.atleast_1d(d.Capacity)[0]) if np.size(d.Capacity) else np.nan
            if not np.isfinite(cap) or cap <= 0 or I.size < 20:
                continue
            q = np.concatenate([[0.0], np.cumsum(np.abs(I[1:]) * np.diff(t)) / 3600.0])
            soc = 1.0 - q / cap
            dI = np.diff(I)
            k = np.where(np.abs(dI) > 2.0)[0]
            r = np.abs(np.diff(V)[k] / dI[k])
            socs.extend(soc[k + 1].tolist())
            rs.extend(r.tolist())
    curve = fit_soc_map(np.array(socs), np.array(rs), soc_ref, bins=10, source="fitted_nasa_square_wave")
    return curve, used


def fit_nasa_baseline(root: str | Path, cells: list[str], cfg: Any, source: str = "eis", soc_map: bool = True,
                      ea_time_trend: bool = False, seed: int = 0) -> Any:
    """Population baseline fitted on NASA cells with full provenance."""
    from ..baseline.fit import fit_population_baseline, flat_soc_map
    from ..baseline.model import FileRecord, Provenance
    from .registry import REGISTRY, load_manifest, sha256_file

    df, metas = load(root, cells, source)
    from ..schema import validate

    clean, rep = validate(df.drop(columns=[c for c in ("t_unix", "op_index", "ambient_c", "capacity_ah") if c in df]), cfg)
    clean = clean[clean["qc_ok"]]
    cap = df.set_index(["cell_id", "t"])["capacity_ah"]
    tab = pd.DataFrame({
        "cell_id": clean["cell_id"].to_numpy(), "order": clean["t"].to_numpy(float),
        "temp_c": clean["temp_cell_c"].to_numpy(float),
        "r_total": clean["resistance_ohm"].where(clean["resistance_ohm"].notna(),
                                                 clean["r_ohmic_ohm"] + clean["r_slow_ohm"]).to_numpy(float),
        "r_ohmic": clean["r_ohmic_ohm"].to_numpy(float), "r_slow": clean["r_slow_ohm"].to_numpy(float),
        "capacity_ah": np.array([cap.get((c, t), np.nan) if not isinstance(cap.get((c, t), np.nan), pd.Series) else np.nan
                                 for c, t in zip(clean["cell_id"], clean["t"], strict=True)], float),
    })
    tab["capacity_ah"] = tab["capacity_ah"].where(np.isfinite(tab["capacity_ah"]),
                                                  tab["cell_id"].map({k: m["soh0_ah"] for k, m in metas.items()}))
    info = REGISTRY["nasa"]
    man = load_manifest("nasa")
    files = [FileRecord(name=Path(m["file"]).name, sha256=sha256_file(Path(m["file"])),
                        bytes=Path(m["file"]).stat().st_size) for m in metas.values()]
    prov = Provenance(
        dataset=info.title, version="PCoE Battery Data Set 5 (phm-datasets S3 zip, last modified 2022-09-18)",
        url=info.files[0].url, licence=info.licence, files=files,
        notes=[f"resistance source: {source}", f"archive sha256 {info.files[0].sha256}",
               *(f"{k}: " + "; ".join(m["notes"]) for k, m in metas.items() if m["notes"]),
               *([f"validation: {i['code']} {i.get('cell_id')}: {i['detail']}" for i in rep.to_dict()["issues"]
                  if i["severity"] != "info"])],
        archive_manifest_present=man is not None,
    )
    ambients = sorted({a for m in metas.values() for a in m["ambient_c"]})
    if source == "eis":
        prov.data_cannot_support.append(
            "EIS temperatures are the chamber ambient (no cell temperature during the sweep); "
            f"ambients in this set: {ambients} C")
    curve = None
    used_sw: list[dict[str, Any]] = []
    if soc_map and source == "dcir":
        curve, used_sw = square_wave_soc_map(root, soc_ref=float(cfg.physics.soc_ref))
        prov.files.extend(FileRecord(name=u["name"], sha256=u["sha256"]) for u in used_sw)
    rkind = "eis_re_rct" if source == "eis" else "dcir_onset"
    bl = fit_population_baseline(
        tab, cfg, baseline_id=f"nasa-{source}-{'-'.join(cells)}", chemistry=CHEMISTRY, form_factor=FORM_FACTOR,
        nominal_capacity_ah=RATED_AH, resistance_kind=rkind, provenance=prov, soc_map=curve or flat_soc_map(
            "NASA EIS has no SOC-resolved resistance; map assumed flat" if source == "eis" else
            "square-wave SOC map not requested; map assumed flat"),
        ea_time_trend=ea_time_trend, seed=seed)
    bl.provenance.notes.append("soh0 per cell (Ah): " + ", ".join(f"{k}={m['soh0_ah']:.3f}" for k, m in metas.items()))
    return bl, df, metas
