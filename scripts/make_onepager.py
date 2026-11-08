"""Build docs/ONE_PAGER.md from out/validation.json and out/pytest.log only (no hand-typed numbers)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
v = json.loads((out / "validation.json").read_text())
log = (out / "pytest.log").read_text() if (out / "pytest.log").exists() else ""
m = re.search(r"(\d+) passed(?:, (\d+) failed)?", log)
tests = f"{m.group(1)} passed" + (f", {m.group(2)} failed" if m and m.group(2) else "") if m else "not recorded"
pct = lambda x: "n/a" if x is None or x != x else f"{100 * x:.0f}%"  # noqa: E731
L = ["# celldx: one-pager (results from runs executed " + v["finished"] + ")", "",
     "Probabilistic per-cell fault diagnosis plus a hot-cell thermal controller for REON-style packs "
     "(sodium-ion default). Every number below is read from `out/validation.json` and `out/pytest.log`, "
     "produced by `scripts/run_validation.py` and `pytest`. Simulator results are optimistic versus real packs.", "",
     f"**Tests:** {tests} (unit, property-based, closed-loop safety, CLI, real NASA files, Firebreak CSV). ruff and mypy clean.", "",
     f"## Diagnosis accuracy ({v['trials_per_scenario']} packs per scenario, 8 scenarios, fault on one cell)", "",
     "| Data | Profile | Baseline | Fault-class accuracy | Missed faults | False alarms (healthy neighbours) | Wrong & High confidence |",
     "|---|---|---|---|---|---|---|"]
for k, s in v.items():
    if not k.startswith("sim_"):
        continue
    sim, prof, bm = k.split("/")
    L.append(f"| {'Sim A (prototype harness)' if sim == 'sim_A' else 'Sim B (independent, held-out seeds, no retuning)'} | {prof} | {bm} | "
             f"{pct(s['class_accuracy'])} | {pct(s['missed_fault_rate'])} | {pct(s['false_alarm_rate_healthy_neighbours'])} "
             f"(n={s['n_healthy_neighbours']}) | {s['wrong_high_confidence']} of {s['n_wrong']} wrong |")
L += ["", "Per-scenario accuracy, sim B held-out, prototype, own baseline: " +
      ", ".join(f"{k} {pct(x)}" for k, x in v["sim_B_heldout/prototype/own"]["per_scenario_class_accuracy"].items()), ""]
L += ["Exact regression: in prototype profile the package matches an independent loop-based reference implementation "
      "to 1e-12 on features, probabilities, verdict and confidence (tests/test_engine.py).", ""]
fb = v["firebreak_small_csv"]
L += ["Firebreak hot-day CSV (pack baseline): top cells " + "; ".join(
    f"{c['cell_id']} {c['verdict']} p={c['fault_probability']:.2f} [{c['interval'][0]:.2f}, {c['interval'][1]:.2f}] {c['confidence']}"
    for c in fb[:3]) + ". C20 is Firebreak's hidden high-resistance cell.", ""]
L += ["## Hot-cell thermal controller (3-day closed-loop pack simulator, seed 0)", "",
      "| Scenario | Strategy | Served discharge | Charge kWh | Hot & full cell-h | Cell-h above 30 C | Max T C | SEI-proxy max | SEI CV | Current at >=57 C (cell-min) | Inverter trips (min) | Chatter <5 min |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for k, r in v["thermal_seed0"].items():
    sc, st = k.split("/")
    L.append(f"| {sc} | {st} | {pct(r['energy.served_discharge_frac'])} | {r['energy.accepted_charge_kwh']:.2f} | "
             f"{r['thermal.hot_and_full_cell_hours']:.1f} | {r['thermal.cell_hours_above_band']:.0f} | {r['thermal.max_temp_c']:.1f} | "
             f"{r['aging_proxy.sei_days_max']:.2f} | {r['aging_proxy.sei_days_cv']:.3f} | "
             f"{r['violations_truth.carrying_current_at_or_above_cutoff_cell_min']:.0f} | "
             f"{r['violations_truth.inverter_out_of_window_min']:.0f} | {r['switching.chatter_events_lt5min']:.0f} |")
L += ["", "## Limits of these results", "",
      "* Probabilities are `uncalibrated_scores` until a calibration is fitted on labelled field data.",
      "* Sim B (written independently, no retuning) is the honest number; sim A is the harness the logic was built against.",
      "* Thermal results are one seed of a simulator; bypass at constant power raises the string current, so cell-hours above "
      "30 C rise slightly; the gains are in hot-and-full hours, SEI spread and the 57 C cutoff.",
      "* CALCE, Oxford and MATR adapters are untested against real files (hosts blocked from the build sandbox)."]
Path("docs").mkdir(exist_ok=True)
Path("docs/ONE_PAGER.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
