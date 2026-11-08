"""Labelled evaluation harness: run many simulated packs and score the verdicts.

Definitions (used in every report):

* **exact accuracy**: verdict == expected label.
* **fault-class accuracy**: like exact, but ``healthy`` and ``temperature_effect`` are one
  "no fault" class (the brief: a healthy cold cell is non-fault even though its label reads
  temperature effect).
* **missed fault**: a fault scenario whose fault probability is below 0.5.
* **false alarm**: a non-fault cell (healthy or cold target, or any healthy neighbour) whose
  fault probability is 0.5 or more.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from .. import CAUSES, NON_FAULT_CAUSES
from ..baseline.model import Baseline, Curve, Param, Provenance
from ..config import Config, load_config
from ..diagnosis.engine import diagnose

SCORING_FEATURES = ("corrected_excess", "raw_excess", "step_rel", "step_z", "soh_step", "ocv_step_mv", "ocv_recent_mv",
                    "soh_recent", "b_rel", "cell_minus_pack_k", "unexplained_heat_k", "post_cold_charge_rise",
                    "growth_r2", "noise_ratio", "temp_step_at_r_step_k")
ABLATIONS = ["none", "no_ocv", "no_soh", "no_pack_temp", "no_current", "no_neighbours"]


def population_baseline(r0_true: float, err_rel: float, ea: float, cell_id: str, chemistry: str = "nmc",
                        form_factor: str = "18650", capacity_ah: float | None = 2.5,
                        resistance_kind: str = "sim_daily") -> Baseline:
    """A population-style baseline whose R0 is off by ``err_rel`` (simulates using a fleet
    median instead of the cell's own early life). Trust 0.85 via kind=population."""
    return Baseline(
        baseline_id=f"population-{cell_id}", kind="population", chemistry=chemistry, form_factor=form_factor,
        nominal_capacity_ah=capacity_ah, resistance_kind=resistance_kind,
        r0_25c_ohm=Param(value=r0_true * (1 + err_rel), source="fitted_population",
                         note=f"simulated population baseline, R0 error {err_rel:+.1%}"),
        r0_ohmic_ohm=Param(value=None, source="unavailable"), r0_slow_ohm=Param(value=None, source="unavailable"),
        ea_j_per_mol=Param(value=ea, source="fitted_population"), ea_ohmic_j_per_mol=Param(value=ea, source="fitted_population"),
        ea_slow_j_per_mol=Param(value=ea, source="fitted_population"),
        soh0_ah=Param(value=capacity_ah, source="simulation_truth"),
        r_soc_map=Curve(x=[0.0, 1.0], y=[1.0, 1.0], source="assumed_flat"),
        provenance=Provenance(dataset="simulation (population baseline with injected R0 error)",
                              fitting_method="simulation"),
    )


def apply_ablation(df: pd.DataFrame, ablation: str) -> pd.DataFrame:
    """Remove one input. ``no_current`` removes current and the cold-charge flag derived from it."""
    df = df.copy()
    if ablation == "no_ocv":
        df["ocv_mv"] = np.nan
    elif ablation == "no_soh":
        df["soh"] = np.nan
    elif ablation == "no_pack_temp":
        df["temp_pack_c"] = np.nan
    elif ablation == "no_current":
        df["current_a"] = np.nan
        df["cold_charge"] = np.nan
    elif ablation == "no_neighbours":
        df = df[df["cell_id"] == "C00"]
    elif ablation != "none":
        raise ValueError(ablation)
    return df


def run_trial(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Run one simulated pack and return one record per cell."""
    sim = task["sim"]
    if sim == "A":
        from .harness_a import EXPECTED, simulate_trial
    else:
        from .sim_b import EXPECTED, simulate_trial  # type: ignore[no-redef,assignment]
    cfg = load_config(overrides=task.get("overrides") or {}, profile=task.get("profile", "prototype"))
    df, truths = simulate_trial(task["scenario"], seed=int(task["seed"]), **(task.get("sim_kwargs") or {}))
    df = apply_ablation(df, task.get("ablation", "none"))
    rng = np.random.default_rng(int(task["seed"]) + 99991)
    bmode = task.get("baseline", "own")
    baseline: Any = None
    if bmode == "population":
        err = float(task.get("pop_err", 0.06))
        ea_pop = float(task.get("pop_ea", 30e3))
        baseline = {t.cell_id: population_baseline(t.r0_ohm, err * rng.choice([-1, 1]), ea_pop, t.cell_id,
                                                   **(task.get("pop_kwargs") or {}))
                    for t in truths}
    rep = diagnose(df, baseline, cfg, bootstrap=bool(task.get("bootstrap", False)), seed=int(task["seed"]),
                   calibration=task.get("calibration"))
    by_id = {c["cell_id"]: c for c in rep["cells"]}
    recs = []
    for t in truths:
        c = by_id.get(t.cell_id)
        if c is None:
            continue
        expected = EXPECTED[t.scenario]
        verdict = c["verdict"]
        fp = c["fault_probability"] if c["fault_probability"] is not None else 0.0
        is_fault_truth = expected not in NON_FAULT_CAUSES
        rec = {
            "sim": sim, "scenario": t.scenario, "seed": int(task["seed"]), "cell_id": t.cell_id,
            "group": f"{sim}-{task['scenario']}-{task['seed']}-{t.cell_id}",
            "target": t.cell_id == "C00", "label": expected, "verdict": verdict, "fault_probability": fp,
            "fault_probability_interval": c.get("fault_probability_interval"),
            "confidence": c["confidence"]["value"], "confidence_level": c["confidence"]["level"],
            "exact_correct": verdict == expected,
            "class_correct": (verdict in NON_FAULT_CAUSES) if not is_fault_truth else verdict == expected,
            "truth_fault": is_fault_truth, "missed": is_fault_truth and fp < 0.5,
            "false_alarm": (not is_fault_truth) and fp >= 0.5,
            "baseline": bmode, "ablation": task.get("ablation", "none"), "profile": task.get("profile", "prototype"),
            "severity": t.params.get("severity_draw"),
            **{f"p_{k}": c["cause_probabilities"][k] for k in CAUSES},
        }
        feats = c.get("features", {})
        for k in SCORING_FEATURES:
            rec[f"f_{k}"] = feats.get(k)
        rec["has_ocv"] = bool(c.get("flags", {}).get("has_ocv"))
        rec["fault_probability_uncalibrated"] = c.get("fault_probability_uncalibrated", fp)
        for k, val in (c.get("sub_scores") or {}).items():
            rec[f"s_{k}"] = val
        recs.append(rec)
    return recs


def run_many(tasks: list[dict[str, Any]], n_jobs: int = 4) -> pd.DataFrame:
    """Run tasks in parallel (processes) and return a flat DataFrame of cell records."""
    out: list[dict[str, Any]] = []
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for recs in ex.map(run_trial, tasks, chunksize=2):
                out.extend(recs)
    else:
        for t in tasks:
            out.extend(run_trial(t))
    return pd.DataFrame(out)


def make_tasks(sim: str, scenarios: list[str], trials: int, seed0: int = 0, **common: Any) -> list[dict[str, Any]]:
    return [{"sim": sim, "scenario": s, "seed": seed0 + 1000 * i + j, **common}
            for i, s in enumerate(scenarios) for j in range(trials)]


def summarise(df: pd.DataFrame, targets_only: bool = True) -> dict[str, Any]:
    """Accuracy, misses, false alarms, per-scenario table, confusion matrix, confidence split."""
    tgt = df[df["target"]] if targets_only else df
    healthy_neigh = df[(~df["target"])]
    per = {}
    for s, g in tgt.groupby("scenario"):
        per[s] = {"n": int(len(g)), "class_accuracy": float(g["class_correct"].mean()),
                  "exact_accuracy": float(g["exact_correct"].mean()),
                  "mean_fault_probability": float(g["fault_probability"].mean()),
                  "mean_confidence": float(g["confidence"].mean()),
                  "verdicts": g["verdict"].value_counts().to_dict()}
    faults = tgt[tgt["truth_fault"]]
    nonf = tgt[~tgt["truth_fault"]]
    labels = list(CAUSES)
    cm = pd.crosstab(tgt["label"], tgt["verdict"]).reindex(index=[x for x in labels if x in set(tgt["label"])],
                                                          columns=labels, fill_value=0)
    right = tgt[tgt["class_correct"]]
    wrong = tgt[~tgt["class_correct"]]
    return {
        "n_targets": int(len(tgt)),
        "class_accuracy": float(tgt["class_correct"].mean()) if len(tgt) else float("nan"),
        "exact_accuracy": float(tgt["exact_correct"].mean()) if len(tgt) else float("nan"),
        "missed_fault_rate": float(faults["missed"].mean()) if len(faults) else float("nan"),
        "false_alarm_rate_targets": float(nonf["false_alarm"].mean()) if len(nonf) else float("nan"),
        "false_alarm_rate_healthy_neighbours": float(healthy_neigh["false_alarm"].mean()) if len(healthy_neigh) else float("nan"),
        "n_healthy_neighbours": int(len(healthy_neigh)),
        "mean_confidence_right": float(right["confidence"].mean()) if len(right) else float("nan"),
        "mean_confidence_wrong": float(wrong["confidence"].mean()) if len(wrong) else float("nan"),
        "n_wrong": int(len(wrong)),
        "wrong_high_confidence": int((wrong["confidence_level"] == "High").sum()),
        "per_scenario": per,
        "confusion": {"labels": list(cm.index), "verdicts": list(cm.columns), "matrix": cm.to_numpy().tolist()},
    }


def evaluate(sim: str, scenarios: list[str], trials: int, n_jobs: int = 4, **common: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = run_many(make_tasks(sim, scenarios, trials, **common), n_jobs=n_jobs)
    return df, summarise(df)


Runner = Callable[[Mapping[str, Any]], list[dict[str, Any]]]
__all__ = ["ABLATIONS", "apply_ablation", "evaluate", "make_tasks", "population_baseline", "run_many", "run_trial",
           "summarise", "Config"]
