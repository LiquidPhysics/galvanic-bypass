"""Shared fixtures. Tests never touch the network.

The ``nasa`` marker needs the NASA PCoE .mat files; set ``CELLDX_NASA_ROOT`` to the folder
that holds them (searched recursively for ``B00*.mat``). Without it those tests are skipped.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from celldx.config import load_config

NASA_CANDIDATES = [
    os.environ.get("CELLDX_NASA_ROOT", ""),
    str(Path.home() / ".cache" / "celldx" / "nasa"),
    "/home/user/data_cache/nasa",
]


def nasa_root() -> Path | None:
    for c in NASA_CANDIDATES:
        if c and Path(c).exists() and any(Path(c).rglob("B0005.mat")):
            return Path(c)
    return None


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def cfg_v3():
    return load_config(profile="v3")


@lru_cache(maxsize=64)
def _trial(sim: str, scenario: str, seed: int) -> tuple[pd.DataFrame, list]:
    if sim == "A":
        from celldx.sim.harness_a import simulate_trial
    else:
        from celldx.sim.sim_b import simulate_trial
    return simulate_trial(scenario, seed=seed)


@pytest.fixture(scope="session")
def trial():
    """``trial(sim, scenario, seed)`` -> (frame, truths), cached across the session."""

    def get(sim: str, scenario: str, seed: int = 0):
        df, truths = _trial(sim, scenario, seed)
        return df.copy(), truths

    return get


@pytest.fixture(scope="session")
def nasa_path():
    root = nasa_root()
    if root is None:
        pytest.skip("NASA PCoE files not found (set CELLDX_NASA_ROOT)")
    return root


def write_nasa_mat(path: Path, name: str = "B0099", n_pairs: int = 8, ambient: float = 24.0,
                   re0: float = 0.05, rct0: float = 0.08, cap0: float = 2.0, fade: float = 0.01,
                   complex_part: float = 1e-9) -> Path:
    """Write a small .mat file with the NASA PCoE layout: ``name.cycle`` is a struct array of
    alternating discharge and impedance operations (fields type, ambient_temperature, time, data)."""
    from scipy.io import savemat

    dt = np.dtype([("type", "O"), ("ambient_temperature", "O"), ("time", "O"), ("data", "O")])
    n = 2 * n_pairs
    cyc = np.zeros((1, n), dtype=dt)
    for i in range(n):
        k = i // 2
        if i % 2 == 0:
            t = np.arange(0, 100.0, 10.0)
            current = np.r_[0.0, np.full(9, -2.0)]
            cyc[0, i]["type"] = "discharge"
            cyc[0, i]["data"] = {"Capacity": np.array([[cap0 - fade * k]]), "Current_measured": current,
                                 "Voltage_measured": 4.1 + 0.05 * current, "Temperature_measured": np.full(10, ambient),
                                 "Time": t}
        else:
            cyc[0, i]["type"] = "impedance"
            cyc[0, i]["data"] = {"Re": np.array([[complex(re0 + 0.001 * k, complex_part)]]),
                                 "Rct": np.array([[complex(rct0 + 0.002 * k, 0.0)]])}
        cyc[0, i]["ambient_temperature"] = ambient
        cyc[0, i]["time"] = np.array([[2008, 4, 2 + k, 13, 0, float(i)]])
    savemat(str(path), {name: {"cycle": cyc}})
    return path
