"""Paired evaluation of thermal strategies over scenarios and seeds."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from .sim import STRATEGIES, run, scenarios

KEY_METRICS = [
    "energy.served_discharge_frac", "energy.discharge_shortfall_min", "energy.accepted_charge_kwh",
    "thermal.cell_hours_above_band", "thermal.degree_hours_above_band", "thermal.hot_and_full_cell_hours",
    "thermal.healthy_cell_hours_above_band", "thermal.max_temp_c",
    "aging_proxy.sei_days_mean", "aging_proxy.sei_days_max", "aging_proxy.sei_days_cv",
    "switching.switch_events", "switching.chatter_events_lt5min",
    "violations_truth.inverter_out_of_window_min", "violations_truth.carrying_current_at_or_above_cutoff_cell_min",
    "violations_truth.charging_below_min_temp_cell_min", "violations_truth.charging_above_charge_limit_cell_min",
    "violations_truth.charging_above_soc_cap_cell_min", "violations_truth.charging_above_effective_cap_cell_min",
    "violations_truth.cap_lifted_for_energy_cell_min",
]


def _flat(res: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"scenario": res["scenario"], "strategy": res["strategy"], "seed": res["seed"]}
    for group in ("energy", "thermal", "aging_proxy", "switching", "controller", "violations_truth"):
        for k, v in res[group].items():
            row[f"{group}.{k}"] = v
    return row


def _task(args: tuple[str, str, int]) -> dict[str, Any]:
    sc, st, seed = args
    return _flat(run(sc, st, seed))


def run_matrix(scenario_names: Sequence[str] | None = None, strategies: Sequence[str] | None = None,
               seeds: Sequence[int] = (0,), n_jobs: int = 4) -> pd.DataFrame:
    scen = list(scenario_names or scenarios())
    strat = list(strategies or STRATEGIES)
    tasks = [(s, t, int(sd)) for s in scen for t in strat for sd in seeds]
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            rows = list(ex.map(_task, tasks, chunksize=1))
    else:
        rows = [_task(t) for t in tasks]
    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame, metrics: Sequence[str] = KEY_METRICS) -> pd.DataFrame:
    """Mean over seeds for each scenario x strategy."""
    cols = [m for m in metrics if m in df.columns]
    return df.groupby(["scenario", "strategy"])[cols].mean().reset_index()


def paired_change(df: pd.DataFrame, metric: str, a: str, b: str) -> pd.DataFrame:
    """Per scenario: mean of (strategy a - strategy b) over seeds, relative change, and the
    number of seeds where a is lower than b. Pairing is exact (same plant, schedule, noise)."""
    out = []
    for sc, g in df.groupby("scenario"):
        pa = g[g.strategy == a].set_index("seed")[metric]
        pb = g[g.strategy == b].set_index("seed")[metric]
        common = pa.index.intersection(pb.index)
        if not len(common):
            continue
        d = pa[common] - pb[common]
        base = float(pb[common].mean())
        out.append({"scenario": sc, "metric": metric, "a": a, "b": b, "n_seeds": len(common),
                    "mean_a": float(pa[common].mean()), "mean_b": base, "mean_diff": float(d.mean()),
                    "sd_diff": float(d.std(ddof=1)) if len(common) > 1 else float("nan"),
                    "rel_change": float(d.mean() / base) if base else float("nan"),
                    "seeds_a_lower": int((d < 0).sum())})
    return pd.DataFrame(out)


__all__ = ["KEY_METRICS", "paired_change", "run_matrix", "summary_table"]


def _main() -> None:  # pragma: no cover - manual use
    df = run_matrix(seeds=[0], n_jobs=4)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(summary_table(df).round(3).to_string())
    _ = np


if __name__ == "__main__":  # pragma: no cover
    _main()
