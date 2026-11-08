"""Small, pure physics and maths helpers shared by every module.

All temperatures in degrees Celsius at the API boundary, converted to kelvin inside.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

RG = 8.314  # J/(mol K); the configured value is passed explicitly where it matters
T25_K = 298.15
KELVIN = 273.15

FloatArray: TypeAlias = NDArray[np.float64]


def sig(x: ArrayLike, c: float, w: float) -> FloatArray:
    """Logistic ``1 / (1 + exp(-(x - c) / w))``. NaN in gives NaN out.

    Overflow-safe: the exponent is clipped to +-500.
    """
    xa = np.asarray(x, dtype=float)
    z = np.clip(-(xa - c) / w, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(z))


def arrhenius_factor(temp_c: ArrayLike, ea_j_per_mol: float, rg: float = RG, t25_k: float = T25_K) -> FloatArray:
    """Multiplier ``exp(Ea/Rg * (1/T - 1/T25))`` so that ``R(T) = R25 * factor``.

    Units: temp_c in deg C, ea in J/mol. Factor is 1 at 25 C, above 1 when colder.
    """
    t_k = np.asarray(temp_c, dtype=float) + KELVIN
    return np.exp(ea_j_per_mol / rg * (1.0 / t_k - 1.0 / t25_k))


def to_r25(r_ohm: ArrayLike, temp_c: ArrayLike, ea_j_per_mol: float, rg: float = RG, t25_k: float = T25_K) -> FloatArray:
    """Arrhenius correction to 25 C: ``R25 = R / exp(Ea/Rg * (1/T - 1/T25))`` (ohm)."""
    return np.asarray(r_ohm, dtype=float) / arrhenius_factor(temp_c, ea_j_per_mol, rg, t25_k)


def from_r25(r25_ohm: ArrayLike, temp_c: ArrayLike, ea_j_per_mol: float, rg: float = RG, t25_k: float = T25_K) -> FloatArray:
    """Inverse of :func:`to_r25`: ``R(T) = R25 * exp(Ea/Rg * (1/T - 1/T25))``."""
    return np.asarray(r25_ohm, dtype=float) * arrhenius_factor(temp_c, ea_j_per_mol, rg, t25_k)


def soc_factor(soc: ArrayLike, soc_grid: ArrayLike, factor_grid: ArrayLike) -> FloatArray:
    """Interpolated resistance-vs-SOC multiplier ``Rmap(soc)`` (dimensionless).

    Outside the grid the end values are held. NaN SOC returns NaN.
    """
    s = np.asarray(soc, dtype=float)
    out = np.interp(np.nan_to_num(s, nan=0.5), np.asarray(soc_grid, float), np.asarray(factor_grid, float))
    return np.where(np.isnan(s), np.nan, out)


def soc_correct(r25: ArrayLike, soc: ArrayLike, soc_grid: ArrayLike, factor_grid: ArrayLike, soc_ref: float) -> FloatArray:
    """SOC correction ``R_corr = R25 * Rmap(soc_ref) / Rmap(soc)``.

    Rows with NaN SOC are referred to ``soc_ref`` (no correction), which is the neutral choice.
    """
    s = np.asarray(soc, dtype=float)
    s = np.where(np.isnan(s), soc_ref, s)
    ref = float(np.interp(soc_ref, np.asarray(soc_grid, float), np.asarray(factor_grid, float)))
    return np.asarray(r25, dtype=float) * ref / soc_factor(s, soc_grid, factor_grid)


def robust_mad(x: ArrayLike) -> float:
    """Median absolute deviation (unscaled), ignoring NaN. Returns NaN for empty input."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.median(np.abs(a - np.median(a))))


def robust_spread(x: ArrayLike, q_lo: float = 0.05, q_hi: float = 0.95) -> float:
    """Robust range ``P_hi - P_lo`` ignoring NaN (same units as x)."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return 0.0
    lo, hi = np.quantile(a, [q_lo, q_hi])
    return float(hi - lo)
