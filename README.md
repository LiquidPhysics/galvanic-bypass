# celldx

**celldx is our own cell heat-fault diagnosis algorithm**, built by team **Galvanic Bypass**
(Rhythm, Aaryan, Anirudh and Sifa) for the REON challenge "Beyond the Weakest Cell: Making Every
Cell Count" at the MIT Energy & Climate Hack 2026. It looks at each cell of a pack with per-cell
sensing and bypass, works out which cells are developing a heat-related fault and why, and decides
what the pack should do about it. It feeds the Heat Atlas frontend.

In short, it provides probabilistic per-cell fault diagnosis, action planning and a hot-cell thermal controller for
REON-style packs with per-cell bypass. Sodium-ion is the default chemistry.

* **Diagnosis** (`celldx.diagnosis`): Arrhenius/SOC-corrected resistance, ten causes, a fault
  probability with a bootstrap interval, and a confidence with named penalties. Probabilities are
  labelled `uncalibrated_scores` until a calibration fitted on labelled data is applied (`celldx calibrate`).
* **Baselines** (`celldx.baseline`): own early life, pack median, population or literature. Each one
  carries provenance and a trust weight, plus compatibility warnings for chemistry, form factor,
  capacity, resistance level and resistance definition.
* **Actions** (`celldx.actions`): string-aware planner. Low confidence never recommends an
  irreversible action. Actuation is a dry run unless `execute(..., dry_run=False, confirm=...)` is
  called from Python, and it is logged.
* **Thermal** (`celldx.thermal`): a separate live controller. It uses predictive hot flags, learned
  thermal floors and aging-rate rotation. Bypass is limited by the power and voltage window, the
  energy available and band protection. It also applies charge rules and a 57 C cutoff.

## Install and run the tests

You need Python 3.11 or newer.

```bash
git clone <this repository URL> celldx && cd celldx
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # or: pip install -e . pytest hypothesis
pytest tests -q             # 151 tests in about 1.5 minutes
```

Without the NASA PCoE battery files, 150 tests pass and 1 is skipped (the real-data NASA test).
To run that one too, download the public NASA battery data set, unzip it and point
`CELLDX_NASA_ROOT` at that folder before running `pytest`.

## Use it

```bash
celldx simulate --help
celldx diagnose tests/fixtures/firebreak_hot_day_small.csv --baseline pack \
    --out report.json --text report.txt --dashboard dashboard.json --pack-id P0
celldx plan report.json --out plan.json                 # dry run
celldx thermal step < tests/fixtures/firebreak_frames.jsonl   # one decision per frame
celldx thermal simulate --help
python scripts/run_validation.py out 10 && python scripts/make_onepager.py out
```

## Frontend integration (Heat Atlas)

* `--dashboard` writes `celldx/v1` JSON (`celldx.export.to_dashboard`). It has one entry per cell,
  sorted by fault probability, with verdict, interval, confidence, evidence and the recommended
  action. `layout_id_map` maps source ids onto a layout's ids (for example C01..C32).
  Example: `out/sample/dashboard_celldx_v1.json`.
* `celldx thermal step` reads REON frames as JSON lines,
  `{"t","ambient_C","request_W","cells":[{"id","V","I","T","soc","soh","R_mOhm"}]}`, and writes one
  decision per frame: action and reason per cell, granted power, binding limits. Example:
  `out/sample/thermal_decisions.jsonl`.

## Results

Pre-computed outputs live in `out/`: `out/sample/` (diagnosis report, plan, dashboard and
thermal decisions), `out/nasa/` (NASA PCoE results) and `out/heat-atlas-pack/` (the simulated pack
shown in Heat Atlas).


See `docs/ONE_PAGER.md`, generated from `out/validation.json` and `out/pytest.log` only.

## Team and license

Built by team Galvanic Bypass (Rhythm, Aaryan, Anirudh, Sifa) at the MIT Energy & Climate Hack 2026.
Released under the MIT License (see `LICENSE`).
