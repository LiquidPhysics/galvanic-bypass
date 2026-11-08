"""Adapters for REON / Firebreak telemetry (tested against Firebreak's example files) and for
CALCE, Oxford and MIT-Stanford-Toyota data (``untested_against_real_files``: the hosts were
not reachable from the build sandbox, so these follow the documented layouts and must be
checked against real files before use).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

UNTESTED = "untested_against_real_files"

# ----------------------------------------------------------------------------- REON / Firebreak
FIREBREAK_ALIASES = {
    "timestamp_s": "t", "time": "t", "cell": "cell_id", "voltage_V": "voltage_v", "V": "voltage_v",
    "current_A": "current_a", "I": "current_a", "temperature_C": "temp_cell_c", "temp": "temp_cell_c",
    "ambient_C": "temp_pack_c", "soc": "soc", "soh": "soh",
}


def load_reon_csv(path: str | Path, pack_id: str = "pack0", string_id: str = "string0") -> pd.DataFrame:
    """Firebreak / REON long-format telemetry CSV -> common schema.

    ``resistance_mOhm`` is converted to ohm. Positive current = discharge in both formats.
    ``ambient_C`` (enclosure temperature) is used as the pack temperature.
    """
    raw = pd.read_csv(path, na_values=["", "nan", "NA", "null"])
    cols = {c: FIREBREAK_ALIASES.get(c, FIREBREAK_ALIASES.get(c.lower(), c)) for c in raw.columns}
    df = raw.rename(columns=cols)
    lower = {c.lower(): c for c in raw.columns}
    if "resistance_mohm" in lower:
        df["resistance_ohm"] = pd.to_numeric(raw[lower["resistance_mohm"]], errors="coerce") / 1000.0
    elif "dcir_mohm" in lower:
        df["resistance_ohm"] = pd.to_numeric(raw[lower["dcir_mohm"]], errors="coerce") / 1000.0
    df["pack_id"] = df.get("pack_id", pack_id)
    df["string_id"] = df.get("string_id", string_id)
    keep = [c for c in ("t", "cell_id", "pack_id", "string_id", "voltage_v", "current_a", "temp_cell_c", "temp_pack_c",
                        "soc", "soh", "resistance_ohm") if c in df.columns]
    return df[keep].copy()


def daily_from_reon(df: pd.DataFrame, day_s: float = 86400.0) -> pd.DataFrame:
    """Aggregate minute-level REON telemetry to daily samples (robust medians), for cycle_mode."""
    d = df.assign(day=np.floor(df["t"] / day_s))
    agg = d.groupby(["cell_id", "day"]).agg(
        t=("day", "first"), pack_id=("pack_id", "first"), string_id=("string_id", "first"),
        resistance_ohm=("resistance_ohm", "median"), temp_cell_c=("temp_cell_c", "median"),
        temp_pack_c=("temp_pack_c", "median"), soc=("soc", "median"), soh=("soh", "median"),
        current_a=("current_a", lambda x: float(np.sqrt(np.nanmean(np.square(x))))),
    ).reset_index()
    return agg.drop(columns=["day"])


# ----------------------------------------------------------------------------- CALCE (untested)
def load_calce_arbin(path: str | Path, cell_id: str | None = None) -> pd.DataFrame:
    """CALCE Arbin export (.xlsx/.csv) -> cycle-mode rows, one per cycle. ``untested_against_real_files``.

    Expected columns (documented Arbin layout): Cycle_Index, Test_Time(s), Current(A), Voltage(V),
    Internal_Resistance(Ohm), Discharge_Capacity(Ah); temperature column if the test logged one.
    Resistance = median Internal_Resistance per cycle; SOH = max discharge capacity / first-cycles median.
    """
    p = Path(path)
    raw = pd.read_excel(p, sheet_name=None) if p.suffix in (".xlsx", ".xls") else {"csv": pd.read_csv(p)}
    frames = [v for k, v in raw.items() if str(k).lower().startswith(("channel", "csv"))]
    if not frames:
        raise ValueError(f"{p}: no Channel_* sheet")
    d = pd.concat(frames, ignore_index=True)
    need = ["Cycle_Index", "Internal_Resistance(Ohm)", "Discharge_Capacity(Ah)"]
    miss = [c for c in need if c not in d.columns]
    if miss:
        raise ValueError(f"{p}: missing documented CALCE columns {miss} ({UNTESTED})")
    tcol = next((c for c in d.columns if "temperature" in c.lower()), None)
    g = d.groupby("Cycle_Index")
    cap = g["Discharge_Capacity(Ah)"].max()
    soh0 = float(np.median(cap.iloc[:3])) if len(cap) else np.nan
    out = pd.DataFrame({
        "t": cap.index.to_numpy(float), "cell_id": cell_id or p.stem,
        "resistance_ohm": g["Internal_Resistance(Ohm)"].median().to_numpy(float),
        "temp_cell_c": g[tcol].median().to_numpy(float) if tcol else np.nan,
        "soh": (cap / soh0).to_numpy(float), "cycle_count": cap.index.to_numpy(float),
    })
    out.attrs["status"] = UNTESTED
    return out


# ----------------------------------------------------------------------------- Oxford (untested)
def load_oxford(path: str | Path) -> pd.DataFrame:
    """Oxford Battery Degradation Dataset 1 .mat -> cycle-mode rows. ``untested_against_real_files``.

    Documented layout: Cell1..Cell8 -> cycNNNN (every 100 cycles) -> C1ch, C1dc, OCVch, OCVdc with
    fields t, v, q, T. Resistance is not logged; a cycle-level estimate is the voltage gap between
    the 1C and the pseudo-OCV (C/24) discharge at mid-capacity divided by the 1C current (0.74 A).
    SOH = 1C discharge capacity / first value. Temperature: T of the 1C discharge (chamber 40 C).
    """
    from scipy.io import loadmat

    m = loadmat(str(path), struct_as_record=False, squeeze_me=True)
    rows: list[dict[str, Any]] = []
    for cell in [k for k in m if k.startswith("Cell")]:
        c = m[cell]
        cyc_names = sorted([f for f in c._fieldnames if f.startswith("cyc")])
        q0 = None
        for name in cyc_names:
            cy = getattr(c, name)
            dc, ocv = cy.C1dc, cy.OCVdc
            q = np.abs(np.asarray(dc.q, float))
            qmax = float(q.max())
            q0 = q0 or qmax
            qo = np.abs(np.asarray(ocv.q, float))
            mid = 0.5 * qmax
            v1 = float(np.interp(mid, q, np.asarray(dc.v, float)))
            vo = float(np.interp(mid, qo, np.asarray(ocv.v, float)))
            rows.append({"t": float(name[3:]), "cell_id": cell, "resistance_ohm": (vo - v1) / 0.74,
                         "temp_cell_c": float(np.median(np.asarray(dc.T, float))), "soh": qmax / q0,
                         "cycle_count": float(name[3:])})
    out = pd.DataFrame(rows)
    out.attrs["status"] = UNTESTED
    return out


# ----------------------------------------------------------------------------- MATR (untested)
def load_matr_batch(path: str | Path, max_cells: int | None = None) -> pd.DataFrame:
    """MIT-Stanford-Toyota batch .mat (MATLAB v7.3 / HDF5) -> cycle-mode rows. ``untested_against_real_files``.

    Documented layout: ``batch`` struct array; ``batch(i).summary`` with ``cycle``, ``IR`` (ohm),
    ``QD`` (Ah), ``Tavg`` (C); ``batch(i).policy_readable`` gives the fast-charge policy (plating-prone).
    Requires h5py.
    """
    import h5py  # optional dependency

    rows: list[dict[str, Any]] = []
    with h5py.File(str(path), "r") as f:
        batch = f["batch"]
        n = batch["summary"].shape[0]
        for i in range(n if max_cells is None else min(n, max_cells)):
            s = f[batch["summary"][i, 0]]
            cyc = np.asarray(s["cycle"]).ravel()
            ir = np.asarray(s["IR"]).ravel()
            qd = np.asarray(s["QD"]).ravel()
            tav = np.asarray(s["Tavg"]).ravel()
            q0 = float(np.median(qd[1:4])) if qd.size > 4 else float(qd.max())
            for k in range(cyc.size):
                rows.append({"t": float(cyc[k]), "cell_id": f"matr_{i:03d}", "resistance_ohm": float(ir[k]),
                             "temp_cell_c": float(tav[k]), "soh": float(qd[k] / q0), "cycle_count": float(cyc[k])})
    out = pd.DataFrame(rows)
    out.attrs["status"] = UNTESTED
    return out
