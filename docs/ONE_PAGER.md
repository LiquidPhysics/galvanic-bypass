# celldx: one-pager (results from runs executed 2026-11-08T02:04:19Z)

Probabilistic per-cell fault diagnosis plus a hot-cell thermal controller for REON-style packs (sodium-ion default). Every number below is read from `out/validation.json` and `out/pytest.log`, produced by `scripts/run_validation.py` and `pytest`. Simulator results are optimistic versus real packs.

**Tests:** 151 passed (unit, property-based, closed-loop safety, CLI, real NASA files, Firebreak CSV). ruff and mypy clean.

## Diagnosis accuracy (10 packs per scenario, 8 scenarios, fault on one cell)

| Data | Profile | Baseline | Fault-class accuracy | Missed faults | False alarms (healthy neighbours) | Wrong & High confidence |
|---|---|---|---|---|---|---|
| Sim A (prototype harness) | prototype | own | 96% | 0% | 0% (n=560) | 0 of 3 wrong |
| Sim A (prototype harness) | prototype | population | 92% | 5% | 0% (n=560) | 1 of 6 wrong |
| Sim A (prototype harness) | v3 | own | 96% | 0% | 0% (n=560) | 0 of 3 wrong |
| Sim A (prototype harness) | v3 | population | 91% | 2% | 7% (n=560) | 0 of 7 wrong |
| Sim B (independent, held-out seeds, no retuning) | prototype | own | 72% | 8% | 0% (n=880) | 8 of 22 wrong |
| Sim B (independent, held-out seeds, no retuning) | prototype | population | 57% | 8% | 19% (n=880) | 8 of 34 wrong |
| Sim B (independent, held-out seeds, no retuning) | v3 | own | 76% | 8% | 4% (n=880) | 7 of 19 wrong |
| Sim B (independent, held-out seeds, no retuning) | v3 | population | 74% | 17% | 4% (n=880) | 7 of 21 wrong |

Per-scenario accuracy, sim B held-out, prototype, own baseline: cold 60%, cooling 100%, corrosion 50%, healthy 100%, internal_damage 100%, plating 90%, sei 30%, weld 50%

Exact regression: in prototype profile the package matches an independent loop-based reference implementation to 1e-12 on features, probabilities, verdict and confidence (tests/test_engine.py).

Firebreak hot-day CSV (pack baseline): top cells C20 mechanical_contact_corrosion p=1.00 [1.00, 1.00] Low; C02 healthy p=0.51 [0.40, 0.60] Low; C21 healthy p=0.26 [0.13, 0.35] Low. C20 is Firebreak's hidden high-resistance cell.

## Hot-cell thermal controller (3-day closed-loop pack simulator, seed 0)

| Scenario | Strategy | Served discharge | Charge kWh | Hot & full cell-h | Cell-h above 30 C | Max T C | SEI-proxy max | SEI CV | Current at >=57 C (cell-min) | Inverter trips (min) | Chatter <5 min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hot_pocket | fixed | 100% | 20.30 | 70.2 | 195 | 32.5 | 4.63 | 0.095 | 0 | 0 | 0 |
| hot_pocket | safety_bypass | 100% | 21.18 | 85.2 | 212 | 32.6 | 4.79 | 0.075 | 0 | 0 | 0 |
| hot_pocket | celldx | 100% | 21.08 | 85.2 | 222 | 32.8 | 4.80 | 0.082 | 0 | 0 | 2 |
| hot_pocket_regulation | fixed | 89% | 31.41 | 109.0 | 365 | 34.8 | 4.86 | 0.094 | 0 | 0 | 0 |
| hot_pocket_regulation | safety_bypass | 96% | 34.16 | 146.5 | 447 | 35.1 | 4.80 | 0.071 | 0 | 0 | 647 |
| hot_pocket_regulation | celldx | 95% | 33.99 | 151.3 | 456 | 35.1 | 4.72 | 0.057 | 0 | 0 | 674 |
| peak_shave | fixed | 100% | 15.19 | 58.0 | 215 | 34.6 | 5.07 | 0.096 | 0 | 0 | 0 |
| peak_shave | safety_bypass | 100% | 16.05 | 63.8 | 232 | 35.3 | 5.17 | 0.087 | 0 | 0 | 0 |
| peak_shave | celldx | 100% | 15.76 | 22.3 | 254 | 36.6 | 4.46 | 0.057 | 0 | 0 | 0 |
| bad_weld | fixed | 100% | 15.21 | 60.0 | 224 | 37.3 | 5.08 | 0.098 | 0 | 0 | 0 |
| bad_weld | safety_bypass | 100% | 16.05 | 65.2 | 240 | 37.9 | 5.18 | 0.090 | 0 | 0 | 0 |
| bad_weld | celldx | 100% | 15.79 | 23.0 | 261 | 40.1 | 4.49 | 0.057 | 0 | 0 | 0 |
| peak_shave_spaced | fixed | 100% | 15.19 | 77.2 | 261 | 36.5 | 5.42 | 0.126 | 0 | 0 | 0 |
| peak_shave_spaced | safety_bypass | 100% | 16.05 | 82.7 | 278 | 37.2 | 5.52 | 0.116 | 0 | 0 | 0 |
| peak_shave_spaced | celldx | 100% | 15.68 | 38.4 | 298 | 38.5 | 4.59 | 0.058 | 0 | 0 | 0 |
| heatwave | fixed | 100% | 20.25 | 410.5 | 1320 | 37.0 | 6.24 | 0.093 | 0 | 0 | 0 |
| heatwave | safety_bypass | 100% | 21.12 | 469.4 | 1322 | 37.1 | 6.48 | 0.074 | 0 | 0 | 0 |
| heatwave | celldx | 100% | 21.08 | 468.6 | 1324 | 37.3 | 6.38 | 0.068 | 0 | 0 | 0 |
| exhaust_fault | fixed | 86% | 17.29 | 46.5 | 281 | 62.2 | 6.74 | 0.218 | 0 | 0 | 0 |
| exhaust_fault | safety_bypass | 100% | 21.03 | 122.2 | 317 | 62.3 | 7.04 | 0.179 | 0 | 0 | 0 |
| exhaust_fault | celldx | 100% | 20.90 | 115.2 | 324 | 62.3 | 6.81 | 0.159 | 0 | 0 | 8 |
| cold_morning | fixed | 100% | 20.01 | 0.0 | 0 | 24.3 | 2.49 | 0.099 | 0 | 0 | 0 |
| cold_morning | safety_bypass | 100% | 20.71 | 0.0 | 0 | 24.4 | 2.54 | 0.087 | 0 | 0 | 3 |
| cold_morning | celldx | 100% | 20.71 | 0.0 | 0 | 24.4 | 2.54 | 0.087 | 0 | 0 | 3 |

## Limits of these results

* Probabilities are `uncalibrated_scores` until a calibration is fitted on labelled field data.
* Sim B (written independently, no retuning) is the honest number; sim A is the harness the logic was built against.
* Thermal results are one seed of a simulator; bypass at constant power raises the string current, so cell-hours above 30 C rise slightly; the gains are in hot-and-full hours, SEI spread and the 57 C cutoff.
* CALCE, Oxford and MATR adapters are untested against real files (hosts blocked from the build sandbox).
