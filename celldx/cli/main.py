"""``celldx`` command line.

Commands: ``data fetch|list``, ``baseline fit``, ``diagnose``, ``plan``, ``report``, ``calibrate``,
``simulate``, ``export dashboard``, ``thermal simulate|step|replay``. Every command writes JSON
(machine-readable) and, where useful, a readable text report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer

from .. import __version__
from ..config import load_config

app = typer.Typer(add_completion=False, no_args_is_help=True, help="celldx: per-cell fault diagnosis, actions and "
                  "hot-cell thermal control.")
data_app = typer.Typer(no_args_is_help=True, help="Open datasets: download with checksum verification.")
baseline_app = typer.Typer(no_args_is_help=True, help="Fit and inspect baselines.")
export_app = typer.Typer(no_args_is_help=True, help="Export reports for dashboards.")
thermal_app = typer.Typer(no_args_is_help=True, help="Hot-cell thermal controller (separate from diagnosis).")
app.add_typer(data_app, name="data")
app.add_typer(baseline_app, name="baseline")
app.add_typer(export_app, name="export")
app.add_typer(thermal_app, name="thermal")


# ------------------------------------------------------------------ helpers
def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _clean(x: Any) -> Any:
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def dump_json(obj: Any, path: Path | None) -> None:
    text = json.dumps(_clean(obj), indent=2, default=_json_default)
    if path is None:
        sys.stdout.write(text + "\n")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


def read_table(path: Path) -> pd.DataFrame:
    """CSV or Parquet in the common schema, or a Firebreak/REON telemetry CSV (auto-detected)."""
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    head = pd.read_csv(path, nrows=2)
    lower = {c.lower() for c in head.columns}
    if {"timestamp_s", "voltage_v", "temperature_c"} <= lower or "resistance_mohm" in lower:
        from ..datasets.others import daily_from_reon, load_reon_csv
        df = load_reon_csv(path)
        span_days = (df["t"].max() - df["t"].min()) / 86400.0
        # long minute-level histories are reduced to daily medians (cycle_mode); short ones are
        # used sample by sample
        return daily_from_reon(df) if span_days >= 30 else df
    return pd.read_csv(path)


def _overrides(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


Profile = Annotated[str, typer.Option(help="prototype (the brief's logic) or v3 (all proposed fixes on)")]
ConfigOpt = Annotated[Path | None, typer.Option("--config", help="YAML file overriding defaults.yaml")]
SetOpt = Annotated[list[str] | None, typer.Option("--set", help="dotted override, e.g. scoring.fault.c=0.2")]


# ------------------------------------------------------------------ version
@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


# ------------------------------------------------------------------ data
@data_app.command("list")
def data_list() -> None:
    """List registered datasets with licence, landing page and adapter status."""
    from ..datasets import ADAPTER_STATUS, REGISTRY
    dump_json({k: {**v.to_dict(), "adapter_status": ADAPTER_STATUS.get(k)} for k, v in REGISTRY.items()}, None)


@data_app.command("fetch")
def data_fetch(name: str, dest: Annotated[Path | None, typer.Option(help="cache root")] = None,
               from_file: Annotated[Path | None, typer.Option(help="use a manually downloaded archive")] = None) -> None:
    """Download a dataset with SHA-256 verification and write a manifest (checksums of every file)."""
    from ..datasets import fetch
    man = fetch(name, dest, from_file=from_file)
    typer.echo(f"{name}: {len(man['files'])} archive(s), {len(man['extracted'])} files; licence: {man['licence']}", err=True)
    dump_json(man, None)


# ------------------------------------------------------------------ baseline
@baseline_app.command("fit")
def baseline_fit(
    source: Annotated[str | None, typer.Option(help="nasa")] = None,
    cells: Annotated[list[str] | None, typer.Option(help="source cells, e.g. --cells B0005 --cells B0006")] = None,
    root: Annotated[Path | None, typer.Option(help="dataset root (default: cache)")] = None,
    resistance: Annotated[str, typer.Option(help="eis (Re+Rct) or dcir")] = "eis",
    own: Annotated[bool, typer.Option("--own", help="fit a cell's own early-life baseline")] = False,
    cell: Annotated[str | None, typer.Option(help="cell id for --own")] = None,
    input: Annotated[Path | None, typer.Option(help="telemetry for --own")] = None,
    out: Annotated[Path | None, typer.Option(help="output .json or .yaml")] = None,
    profile: Profile = "prototype", config: ConfigOpt = None, set_: SetOpt = None,
) -> None:
    """Fit a population baseline from open data, or a cell's own early-life baseline."""
    cfg = load_config(config, _overrides(set_), profile)
    if own:
        if not (cell and input):
            raise typer.BadParameter("--own needs --cell and --input")
        from ..baseline.fit import fit_own_baseline
        from ..diagnosis.engine import TargetMeta, build_contexts
        df = read_table(input)
        ctxs, _ = build_contexts(df[df["cell_id"] == cell], cfg, None, TargetMeta(), 0)
        bl = ctxs[cell].baseline
        _ = fit_own_baseline
    elif source == "nasa":
        from ..datasets import cache_dir
        from ..datasets.nasa import fit_nasa_baseline
        r = root or (cache_dir() / "nasa")
        bl, _, _ = fit_nasa_baseline(r, list(cells or ["B0005", "B0006", "B0007", "B0018"]), cfg, source=resistance)
    else:
        raise typer.BadParameter("give --source nasa or --own")
    if out:
        bl.save(out)
        typer.echo(f"wrote {out}", err=True)
    dump_json(bl.model_dump(mode="json"), None if out else None)


# ------------------------------------------------------------------ diagnose
@app.command()
def diagnose(
    input: Annotated[Path, typer.Argument(help="telemetry: CSV/Parquet in the common schema, or a REON/Firebreak CSV")],
    baseline: Annotated[str, typer.Option(help="own | pack | path to a baseline .json/.yaml")] = "own",
    out: Annotated[Path | None, typer.Option(help="report JSON (default stdout)")] = None,
    text: Annotated[Path | None, typer.Option(help="also write a readable report here")] = None,
    dashboard: Annotated[Path | None, typer.Option(help="also write celldx/v1 dashboard JSON here")] = None,
    pack_id: Annotated[str | None, typer.Option(help="pack id for the dashboard JSON")] = None,
    calibration: Annotated[Path | None, typer.Option(help="fitted calibration JSON")] = None,
    latest: Annotated[str | None, typer.Option(help="streaming call: score only this cell's latest window")] = None,
    chemistry: Annotated[str | None, typer.Option(help="target chemistry for compatibility checks")] = None,
    form_factor: Annotated[str | None, typer.Option()] = None,
    capacity_ah: Annotated[float | None, typer.Option()] = None,
    jobs: Annotated[int, typer.Option(help="parallel feature extraction across cells")] = 1,
    bootstrap: Annotated[bool, typer.Option(help="bootstrap interval on the fault probability")] = True,
    profile: Profile = "prototype", config: ConfigOpt = None, set_: SetOpt = None,
) -> None:
    """Diagnose every cell (batch) or one cell's latest window (``--latest``)."""
    from ..baseline.model import Baseline
    from ..calibration import FittedCalibration
    from ..diagnosis.engine import TargetMeta, diagnose_latest
    from ..diagnosis.engine import diagnose as run_diagnose
    from ..diagnosis.explain import text_report
    cfg = load_config(config, _overrides(set_), profile)
    df = read_table(input)
    bl: Any = baseline if baseline in ("own", "pack") else Baseline.load(Path(baseline))
    if bl == "own":
        bl = None
    cal = FittedCalibration.load(calibration).to_engine() if calibration else None
    meta = TargetMeta(chemistry=chemistry, form_factor=form_factor, capacity_ah=capacity_ah)
    if latest:
        cell = diagnose_latest(df[df["cell_id"] == latest], bl if not isinstance(bl, str) else None, cfg, meta=meta,
                               calibration=cal, neighbours=df, bootstrap=bootstrap)
        dump_json(cell, out)
        return
    rep = run_diagnose(df, bl, cfg, meta=meta, calibration=cal, n_jobs=jobs, bootstrap=bootstrap)
    dump_json(rep, out)
    if text:
        text.write_text(text_report(rep), encoding="utf-8")
    if dashboard:
        from ..actions import plan_report
        from ..export import to_dashboard
        dump_json(to_dashboard(rep, pack_id=pack_id, plan=plan_report(rep, cfg)), dashboard)


# ------------------------------------------------------------------ plan / report
@app.command()
def plan(report: Annotated[Path, typer.Argument(help="diagnosis report JSON")],
         out: Annotated[Path | None, typer.Option()] = None,
         active_balancing: Annotated[bool, typer.Option(help="raise the bypass threshold (charge can be moved)")] = False,
         profile: Profile = "prototype", config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Recommend actions (dry run). Real actuation is only available from Python with explicit
    confirmation (``celldx.actions.execute``)."""
    from ..actions import plan_report
    cfg = load_config(config, _overrides(set_), profile)
    rep = json.loads(report.read_text(encoding="utf-8"))
    dump_json(plan_report(rep, cfg, active_balancing=active_balancing), out)


@app.command()
def report(report: Annotated[Path, typer.Argument(help="diagnosis report JSON")],
           out: Annotated[Path | None, typer.Option()] = None) -> None:
    """Readable text report from a diagnosis report JSON."""
    from ..diagnosis.explain import text_report
    txt = text_report(json.loads(report.read_text(encoding="utf-8")))
    if out:
        out.write_text(txt, encoding="utf-8")
    else:
        typer.echo(txt)


# ------------------------------------------------------------------ calibrate
@app.command()
def calibrate(table: Annotated[Path, typer.Argument(help="labelled table (Parquet/CSV) from the evaluation harness or "
                                                    "field labels: label, fault_probability, p_<cause>, confidence, group")],
              method: Annotated[str, typer.Option(help="isotonic or platt")] = "isotonic",
              out: Annotated[Path | None, typer.Option(help="fitted calibration JSON")] = None,
              cv_out: Annotated[Path | None, typer.Option(help="grouped cross-validation metrics JSON")] = None,
              png_dir: Annotated[Path | None, typer.Option(help="write reliability diagrams here")] = None) -> None:
    """Fit calibration (fault probability, multi-class temperature, confidence) with grouped CV."""
    from ..calibration import cross_validate, fit, reliability_png
    df = pd.read_parquet(table) if table.suffix.lower() in (".parquet", ".pq") else pd.read_csv(table)
    cv = cross_validate(df, method)
    fc = fit(df, method, source=str(table))
    if out:
        fc.save(out)
    dump_json(cv, cv_out)
    if png_dir:
        png_dir.mkdir(parents=True, exist_ok=True)
        y = (~df["label"].isin(["healthy", "temperature_effect"])).to_numpy(int)
        p = df["fault_probability"].to_numpy(float)
        reliability_png(p, y, fc.to_engine().fault(p), png_dir / "reliability_fault.png", "Fault probability")
        corr = df["class_correct"].to_numpy(int) if "class_correct" in df else None
        if corr is not None:
            c = df["confidence"].to_numpy(float)
            reliability_png(c, corr, fc.to_engine().confidence(c) if fc.to_engine().confidence else None,
                            png_dir / "reliability_confidence.png", "Confidence vs accuracy")


# ------------------------------------------------------------------ simulate
@app.command()
def simulate(scenario: Annotated[str, typer.Argument(help="healthy, cold, cooling, plating, internal, weld, corrosion, sei, "
                                                     "ambiguous (sim A); sim B has the same names")],
             sim: Annotated[str, typer.Option(help="A (reconstructed prototype harness) or B (independent)")] = "A",
             seed: Annotated[int, typer.Option()] = 0,
             out: Annotated[Path | None, typer.Option(help="Parquet or CSV")] = None,
             pulse: Annotated[bool, typer.Option(help="render 100 Hz pulse windows (pulse_mode)")] = False,
             truth: Annotated[Path | None, typer.Option(help="write the injected truth JSON here")] = None) -> None:
    """Generate labelled telemetry with an injected fault on cell C00."""
    if sim.upper() == "A":
        from ..sim.harness_a import simulate_trial
    else:
        from ..sim.sim_b import simulate_trial  # type: ignore[no-redef,assignment]
    df, truths = simulate_trial(scenario, seed=seed)
    if pulse:
        from ..sim.pulse import render_pulses
        df = render_pulses(df)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        (df.to_parquet(out, index=False) if out.suffix.lower() in (".parquet", ".pq") else df.to_csv(out, index=False))
        typer.echo(f"wrote {len(df)} rows to {out}", err=True)
    if truth:
        dump_json([t.__dict__ for t in truths], truth)


# ------------------------------------------------------------------ export
@export_app.command("dashboard")
def export_dashboard(report: Annotated[Path, typer.Argument(help="diagnosis report JSON")],
                     out: Annotated[Path | None, typer.Option()] = None,
                     pack_id: Annotated[str | None, typer.Option()] = None,
                     t: Annotated[float | None, typer.Option(help="snapshot time (s)")] = None,
                     layout: Annotated[Path | None, typer.Option(help="Firebreak layout.json: map ids onto its cell_ids")] = None,
                     profile: Profile = "prototype", config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """celldx/v1 JSON for the Heat Atlas dashboard (cells C01..C32 with a layout)."""
    from ..actions import plan_report
    from ..export import layout_id_map, to_dashboard
    cfg = load_config(config, _overrides(set_), profile)
    rep = json.loads(report.read_text(encoding="utf-8"))
    idm = None
    if layout:
        lay = json.loads(layout.read_text(encoding="utf-8"))
        idm = layout_id_map([c["cell_id"] for c in rep["cells"]], lay["cell_ids"])
    dump_json(to_dashboard(rep, pack_id=pack_id, t=t, plan=plan_report(rep, cfg), id_map=idm), out)


# ------------------------------------------------------------------ thermal
@thermal_app.command("simulate")
def thermal_simulate(scenario: Annotated[str, typer.Option(help="see `celldx thermal scenarios`")] = "hot_pocket",
                     strategy: Annotated[str, typer.Option(help="fixed, safety_bypass, naive, celldx, or an ablation")] = "celldx",
                     seed: Annotated[int, typer.Option()] = 0,
                     out: Annotated[Path | None, typer.Option(help="metrics JSON")] = None,
                     frames: Annotated[Path | None, typer.Option(help="write REON live frames (JSON lines) for replay")] = None,
                     profile: Profile = "prototype", config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Run the thermal pack simulator under one strategy and print its metrics."""
    from ..thermal import sim
    cfg = load_config(config, _overrides(set_), profile)
    fh = frames.open("w", encoding="utf-8") if frames else None
    try:
        res = sim.run(scenario, strategy, seed, cfg=cfg)
        if fh:
            _write_frames(scenario, seed, cfg, fh)
    finally:
        if fh:
            fh.close()
    dump_json(res, out)


def _write_frames(scenario: str, seed: int, cfg: Any, fh: Any) -> None:
    """Fixed-string telemetry frames of a scenario (what a REON pack would stream), for replay."""
    from ..thermal import sim
    sc = sim.scenarios()[scenario]
    plant = sim.ThermalPack(sc, seed)
    req, _ = sim.request_profile(sc, seed)
    rng = np.random.default_rng(seed + 31_337)
    on = np.ones(plant.n, bool)
    i_last = 0.0
    for k, p in enumerate(req):
        cells, amb = plant.measure(rng, on, i_last)
        fh.write(json.dumps(sim.frame_from_plant(plant, cells, k * 60.0, amb, float(p))) + "\n")
        o = plant.advance(60.0, float(p), on, True)
        i_last = o["power_w"] / o["v"] if o["v"] > 0 and abs(o["power_w"]) > 0 else 0.0


@thermal_app.command("scenarios")
def thermal_scenarios() -> None:
    """List thermal simulator scenarios and strategies."""
    from ..thermal import sim
    dump_json({"scenarios": {k: v.description for k, v in sim.scenarios().items()}, "strategies": sim.STRATEGIES}, None)


@thermal_app.command("step")
def thermal_step(remaining_wh: Annotated[float | None, typer.Option(help="scheduled discharge energy left today (Wh)")] = None,
                 profile: Profile = "prototype", config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Live controller: read REON frames (JSON lines) on stdin, write one decision per frame on
    stdout. Frames: {"t", "ambient_C", "request_W", "remaining_Wh"?, "cells": [{"id","V","I","T","soc","soh","R_mOhm"}]}.
    Malformed lines produce a JSON error on stderr and are skipped."""
    from ..thermal import ThermalController, readings_from_frame
    cfg = load_config(config, _overrides(set_), profile)
    ctrl: ThermalController | None = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            fr = json.loads(line)
            t, cells, amb, req = readings_from_frame(fr)
            if ctrl is None:
                ctrl = ThermalController(cfg, [c.id for c in cells])
            rem = fr.get("remaining_Wh", remaining_wh)
            d = ctrl.step(t, cells, req, ambient_c=amb, remaining_energy_wh=rem,
                          remaining_charge_wh=fr.get("remaining_charge_Wh"))
            sys.stdout.write(json.dumps(_clean(_decision_out(d)), default=_json_default) + "\n")
            sys.stdout.flush()
        except Exception as exc:  # noqa: BLE001 - one bad frame never stops the controller
            sys.stderr.write(json.dumps({"error": type(exc).__name__, "detail": str(exc)[:300]}) + "\n")


def _decision_out(d: Any) -> dict[str, Any]:
    return {"t": d.t_s, "request_W": d.request_w, "granted_W": d.granted_w, "n_in": d.n_in,
            "string_V": d.string_v, "binding": d.binding, "rotation_active": d.rotation_active,
            "commands": [{"id": c.id, "in_string": c.in_string, "action": c.action, "reason": c.reason,
                          "T": c.temp_c, "T_pred": c.temp_pred_c, "floor_C": c.floor_c, "soc_cap": c.soc_cap}
                         for c in d.cells],
            "checks": d.checks}


@thermal_app.command("replay")
def thermal_replay(frames: Annotated[Path, typer.Argument(help="JSON-lines frames")],
                   out: Annotated[Path | None, typer.Option(help="decisions JSON lines")] = None,
                   remaining_wh: Annotated[float | None, typer.Option()] = None,
                   profile: Profile = "prototype", config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Run the controller over recorded frames (open loop: the recording is not affected by the
    decisions) and summarise what it would have done."""
    from ..thermal import ThermalController, readings_from_frame
    cfg = load_config(config, _overrides(set_), profile)
    ctrl: ThermalController | None = None
    counts: dict[str, int] = {}
    n = 0
    fh = out.open("w", encoding="utf-8") if out else None
    try:
        for line in frames.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fr = json.loads(line)
            t, cells, amb, req = readings_from_frame(fr)
            ctrl = ctrl or ThermalController(cfg, [c.id for c in cells])
            d = ctrl.step(t, cells, req, ambient_c=amb, remaining_energy_wh=fr.get("remaining_Wh", remaining_wh))
            n += 1
            for c in d.cells:
                counts[c.action] = counts.get(c.action, 0) + 1
            if fh:
                fh.write(json.dumps(_clean(_decision_out(d)), default=_json_default) + "\n")
    finally:
        if fh:
            fh.close()
    floors = ctrl.floors() if ctrl else {}
    dump_json({"frames": n, "cell_steps_by_action": counts, "learned_floors": floors}, None)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
