"""Quick validation run: writes out/validation.json (every one-pager number comes from it)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from celldx.config import load_config
from celldx.sim.evaluate import make_tasks, run_many, summarise
from celldx.sim.harness_a import SCENARIOS

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
OUT.mkdir(parents=True, exist_ok=True)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
res: dict = {"trials_per_scenario": N, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def brief(s):
    return {k: s[k] for k in ("n_targets", "class_accuracy", "exact_accuracy", "missed_fault_rate", "false_alarm_rate_targets",
                              "false_alarm_rate_healthy_neighbours", "n_healthy_neighbours", "mean_confidence_right",
                              "mean_confidence_wrong", "n_wrong", "wrong_high_confidence")} | {
        "per_scenario_class_accuracy": {k: v["class_accuracy"] for k, v in s["per_scenario"].items()},
        "confusion": s["confusion"]}


for sim, seed0, tag in (("A", 0, "sim_A"), ("B", 100000, "sim_B_heldout")):
    for prof in ("prototype", "v3"):
        for bm in ("own", "population"):
            df = run_many(make_tasks(sim, SCENARIOS, N, seed0=seed0, baseline=bm, profile=prof), n_jobs=4)
            res[f"{tag}/{prof}/{bm}"] = brief(summarise(df))
            print(tag, prof, bm, round(res[f"{tag}/{prof}/{bm}"]["class_accuracy"], 3), flush=True)

from celldx.thermal.evaluate import run_matrix  # noqa: E402

th = run_matrix(None, ["fixed", "safety_bypass", "celldx"], seeds=[0], n_jobs=4)
keep = [c for c in th.columns if c.split(".")[0] in ("energy", "thermal", "aging_proxy", "switching", "violations_truth", "controller")]
res["thermal_seed0"] = {f"{r.scenario}/{r.strategy}": {k: (None if (isinstance(r[k], float) and not np.isfinite(r[k])) else float(r[k]))
                                                         for k in keep} for _, r in th.iterrows()}

from celldx.datasets.others import load_reon_csv  # noqa: E402
from celldx.diagnosis.engine import diagnose  # noqa: E402

fb = diagnose(load_reon_csv("tests/fixtures/firebreak_hot_day_small.csv"), "pack", load_config(), bootstrap=True)
res["firebreak_small_csv"] = [{"cell_id": c["cell_id"], "verdict": c["verdict"], "fault_probability": c["fault_probability"],
                               "interval": c["fault_probability_interval"], "confidence": c["confidence"]["level"]}
                              for c in fb["cells"][:4]]
res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
(OUT / "validation.json").write_text(json.dumps(res, indent=1, default=float))
print("done", res["finished"])
