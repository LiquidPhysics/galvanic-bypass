"""Calibration: turn scores into probabilities (and confidence into P(verdict correct)).

Input is a labelled table, one row per (cell, window), with at least
``group`` (cell identity for grouped CV), ``label`` (true cause), ``fault_probability`` (or
``fault_probability_uncalibrated``), ``p_<cause>`` columns, ``confidence`` and
``class_correct``. Labels can come from teardown, warranty returns, maintenance outcomes or
injected-fault simulation (the only labels available in this build: see ``source``).

Methods
-------
* fault probability: isotonic regression (default) or Platt scaling (logistic on logit p);
* multi-class: temperature scaling, one scalar T minimising the negative log-likelihood of
  ``softmax(log p / T)``;
* confidence: isotonic regression of ``class_correct`` on confidence.

All metrics are cross-validated with GroupKFold over ``group`` so no cell appears in both
train and test. Until a calibration is fitted and loaded, reports carry
``probabilities: uncalibrated_scores``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import CAUSES, NON_FAULT_CAUSES

EPS = 1e-6


# ----------------------------------------------------------------------------- metrics
def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def log_loss_binary(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= a) & ((p < b) if b < 1 else (p <= b))
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def reliability_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= a) & ((p < b) if b < 1 else (p <= b))
        if m.any():
            out.append({"bin_lo": float(a), "bin_hi": float(b), "n": int(m.sum()), "mean_predicted": float(p[m].mean()),
                        "observed": float(y[m].mean())})
    return out


def multiclass_log_loss(P: np.ndarray, y_idx: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y_idx)), y_idx], EPS, 1.0))))


# ----------------------------------------------------------------------------- calibrators
@dataclass
class IsotonicMap:
    x: list[float]
    y: list[float]

    def __call__(self, p: Any) -> np.ndarray:
        return np.interp(np.asarray(p, float), self.x, self.y)

    @classmethod
    def fit(cls, p: np.ndarray, y: np.ndarray) -> IsotonicMap:
        from sklearn.isotonic import IsotonicRegression

        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(np.asarray(p, float), np.asarray(y, float))
        return cls(x=[float(v) for v in ir.X_thresholds_], y=[float(v) for v in ir.y_thresholds_])


@dataclass
class PlattMap:
    a: float
    b: float

    def __call__(self, p: Any) -> np.ndarray:
        q = np.clip(np.asarray(p, float), EPS, 1 - EPS)
        z = self.a * np.log(q / (1 - q)) + self.b
        return 1.0 / (1.0 + np.exp(-z))

    @classmethod
    def fit(cls, p: np.ndarray, y: np.ndarray) -> PlattMap:
        from sklearn.linear_model import LogisticRegression

        q = np.clip(np.asarray(p, float), EPS, 1 - EPS)
        X = np.log(q / (1 - q)).reshape(-1, 1)
        if len(np.unique(y)) < 2:
            return cls(1.0, 0.0)
        lr = LogisticRegression(C=1e6).fit(X, np.asarray(y, int))
        return cls(float(lr.coef_[0, 0]), float(lr.intercept_[0]))


def fit_temperature(P: np.ndarray, y_idx: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar

    L = np.log(np.clip(P, 1e-12, 1.0))

    def nll(t: float) -> float:
        Z = L / t
        Z = Z - Z.max(axis=1, keepdims=True)
        Q = np.exp(Z)
        Q /= Q.sum(axis=1, keepdims=True)
        return multiclass_log_loss(Q, y_idx)

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def apply_temperature(P: np.ndarray, t: float) -> np.ndarray:
    Z = np.log(np.clip(P, 1e-12, 1.0)) / t
    Z = Z - Z.max(axis=1, keepdims=True)
    Q = np.exp(Z)
    return Q / Q.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------- fitting + report
def _arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fp = df["fault_probability_uncalibrated"].to_numpy(float) if "fault_probability_uncalibrated" in df else df["fault_probability"].to_numpy(float)
    y_fault = (~df["label"].isin(NON_FAULT_CAUSES)).to_numpy(float)
    P = df[[f"p_{k}" for k in CAUSES]].to_numpy(float)
    y_idx = df["label"].map({k: i for i, k in enumerate(CAUSES)}).to_numpy(int)
    conf = df["confidence"].to_numpy(float)
    correct = df["class_correct"].to_numpy(float)
    return fp, y_fault, P, y_idx, conf, correct


def per_class(y_idx: np.ndarray, pred_idx: np.ndarray) -> dict[str, dict[str, float]]:
    out = {}
    for i, k in enumerate(CAUSES):
        tp = int(((pred_idx == i) & (y_idx == i)).sum())
        fp = int(((pred_idx == i) & (y_idx != i)).sum())
        fn = int(((pred_idx != i) & (y_idx == i)).sum())
        if tp + fp + fn == 0:
            continue
        out[k] = {"precision": tp / (tp + fp) if tp + fp else float("nan"), "recall": tp / (tp + fn) if tp + fn else float("nan"),
                  "support": int((y_idx == i).sum())}
    return out


def confusion(y_idx: np.ndarray, pred_idx: np.ndarray) -> dict[str, Any]:
    m = np.zeros((len(CAUSES), len(CAUSES)), int)
    for a, b in zip(y_idx, pred_idx, strict=True):
        m[a, b] += 1
    rows = [i for i in range(len(CAUSES)) if m[i].sum() > 0]
    return {"labels": [CAUSES[i] for i in rows], "verdicts": list(CAUSES), "matrix": m[rows].tolist()}


def metrics(fp: np.ndarray, y_fault: np.ndarray, P: np.ndarray, y_idx: np.ndarray, conf: np.ndarray,
            correct: np.ndarray) -> dict[str, Any]:
    pred = P.argmax(axis=1)
    return {
        "n": int(len(fp)),
        "fault_brier": brier(fp, y_fault), "fault_log_loss": log_loss_binary(fp, y_fault), "fault_ece": ece(fp, y_fault),
        "multiclass_log_loss": multiclass_log_loss(P, y_idx),
        "confidence_ece": ece(conf, correct), "confidence_brier": brier(conf, correct),
        "mean_confidence": float(conf.mean()), "accuracy_class": float(correct.mean()),
        "per_class": per_class(y_idx, pred), "confusion": confusion(y_idx, pred),
        "fault_reliability": reliability_table(fp, y_fault), "confidence_reliability": reliability_table(conf, correct),
    }


@dataclass
class FittedCalibration:
    fault: IsotonicMap | PlattMap
    temperature: float
    confidence: IsotonicMap
    method: str
    source: str
    n: int

    def to_engine(self) -> Any:
        from ..diagnosis.engine import Calibration

        return Calibration(fault=self.fault, multiclass_t=self.temperature, confidence=self.confidence, source=self.source)

    def save(self, path: str | Path) -> None:
        f = {"kind": "isotonic", "x": self.fault.x, "y": self.fault.y} if isinstance(self.fault, IsotonicMap) else \
            {"kind": "platt", "a": self.fault.a, "b": self.fault.b}
        Path(path).write_text(json.dumps({"schema": "celldx-calibration/1", "method": self.method, "source": self.source,
                                          "n": self.n, "fault": f, "temperature": self.temperature,
                                          "confidence": {"x": self.confidence.x, "y": self.confidence.y}}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> FittedCalibration:
        d = json.loads(Path(path).read_text())
        f = d["fault"]
        fault: IsotonicMap | PlattMap = IsotonicMap(f["x"], f["y"]) if f["kind"] == "isotonic" else PlattMap(f["a"], f["b"])
        return cls(fault, float(d["temperature"]), IsotonicMap(d["confidence"]["x"], d["confidence"]["y"]), d["method"],
                   d["source"], int(d["n"]))


def fit(df: pd.DataFrame, method: str = "isotonic", source: str = "labelled table") -> FittedCalibration:
    fp, y_fault, P, y_idx, conf, correct = _arrays(df)
    fault = IsotonicMap.fit(fp, y_fault) if method == "isotonic" else PlattMap.fit(fp, y_fault)
    t = fit_temperature(P, y_idx)
    cm = IsotonicMap.fit(conf, correct)
    return FittedCalibration(fault, t, cm, method, source, int(len(df)))


def cross_validate(df: pd.DataFrame, method: str = "isotonic", n_splits: int = 5, group_col: str = "group") -> dict[str, Any]:
    """Grouped K-fold: fit on train cells, measure on held-out cells. Returns before/after metrics."""
    from sklearn.model_selection import GroupKFold

    fp, y_fault, P, y_idx, conf, correct = _arrays(df)
    groups = df[group_col].to_numpy()
    fp_c = np.zeros_like(fp)
    P_c = np.zeros_like(P)
    conf_c = np.zeros_like(conf)
    temps = []
    for tr, te in GroupKFold(n_splits=n_splits).split(fp, y_fault, groups):
        f = IsotonicMap.fit(fp[tr], y_fault[tr]) if method == "isotonic" else PlattMap.fit(fp[tr], y_fault[tr])
        fp_c[te] = f(fp[te])
        t = fit_temperature(P[tr], y_idx[tr])
        temps.append(t)
        P_c[te] = apply_temperature(P[te], t)
        conf_c[te] = IsotonicMap.fit(conf[tr], correct[tr])(conf[te])
    overlap = len(set(groups[tr]) & set(groups[te]))
    return {"method": method, "n_splits": n_splits, "group_overlap_last_fold": overlap,
            "before": metrics(fp, y_fault, P, y_idx, conf, correct),
            "after": metrics(fp_c, y_fault, P_c, y_idx, conf_c, correct),
            "temperatures": temps,
            "_arrays": {"fp": fp, "fp_cal": fp_c, "y_fault": y_fault, "conf": conf, "conf_cal": conf_c, "correct": correct}}


def reliability_png(p: np.ndarray, y: np.ndarray, p_cal: np.ndarray | None, path: str | Path, title: str,
                    xlabel: str = "predicted probability", ylabel: str = "observed frequency") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=150)
    ax.plot([0, 1], [0, 1], color="#9aa0a6", lw=1, ls="--", label="perfect")
    for arr, lab, col in ((p, "uncalibrated", "#d93025"), (p_cal, "calibrated (grouped CV)", "#1a73e8")):
        if arr is None:
            continue
        t = reliability_table(arr, y)
        ax.plot([r["mean_predicted"] for r in t], [r["observed"] for r in t], marker="o", color=col, lw=1.6,
                label=f"{lab}, ECE {ece(arr, y):.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


# ----------------------------------------------------------------------------- threshold tuning
DEFAULT_GRID: dict[str, list[float]] = {
    "scoring.fault.c": [0.10, 0.125, 0.15, 0.175],
    "scoring.step_rel.c": [0.08, 0.10, 0.12],
    "scoring.r2_sei.c": [0.6, 0.7, 0.8],
    "scoring.weak.c": [0.05, 0.08, 0.11],
}


def rescore(df: pd.DataFrame, cfg: Any) -> tuple[np.ndarray, np.ndarray]:
    """Re-score stored feature columns (``f_*``) with ``cfg``; returns (verdict index, fault prob)."""
    from ..diagnosis.scoring import cause_scores, probabilities, sub_scores

    feats = {c[2:]: df[c].astype(float).to_numpy() for c in df.columns if c.startswith("f_")}
    has_ocv = df["has_ocv"].to_numpy(bool) if "has_ocv" in df else np.ones(len(df), bool)
    out_p = np.zeros((len(df), len(CAUSES)))
    for flag in (True, False):
        m = has_ocv == flag
        if not m.any():
            continue
        sub = sub_scores({k: v[m] for k, v in feats.items()}, cfg, flag)
        pr = probabilities(cause_scores(sub, cfg, flag))
        out_p[m] = np.column_stack([pr[k] for k in CAUSES])
    return out_p.argmax(axis=1), 1 - out_p[:, 0] - out_p[:, 1]


def _class_acc(df: pd.DataFrame, pred_idx: np.ndarray) -> float:
    pred = np.array(CAUSES)[pred_idx]
    nonf = df["label"].isin(NON_FAULT_CAUSES).to_numpy()
    ok = np.where(nonf, np.isin(pred, NON_FAULT_CAUSES), pred == df["label"].to_numpy())
    return float(ok.mean())


def tune_thresholds(df: pd.DataFrame, cfg: Any, grid: dict[str, list[float]] | None = None, n_splits: int = 5,
                    group_col: str = "group") -> dict[str, Any]:
    """Grid search of scoring centres maximising fault-class accuracy, with GroupKFold:
    the best grid point is chosen on the training cells of each fold and scored on the held-out
    cells. Returns CV accuracy of the default and tuned settings and the per-fold choices."""
    import itertools

    from sklearn.model_selection import GroupKFold

    grid = grid or DEFAULT_GRID
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))
    preds = {}
    for combo in combos:
        c = cfg.with_overrides(dict(zip(keys, combo, strict=True)))
        preds[combo], _ = rescore(df, c)
    base_pred, _ = rescore(df, cfg)
    groups = df[group_col].to_numpy()
    tuned_ok = np.zeros(len(df), bool)
    chosen = []
    for tr, te in GroupKFold(n_splits=n_splits).split(df, groups=groups):
        best = max(combos, key=lambda cb: _class_acc(df.iloc[tr], preds[cb][tr]))
        chosen.append(dict(zip(keys, best, strict=True)))
        sub = df.iloc[te]
        p = np.array(CAUSES)[preds[best][te]]
        nonf = sub["label"].isin(NON_FAULT_CAUSES).to_numpy()
        tuned_ok[te] = np.where(nonf, np.isin(p, NON_FAULT_CAUSES), p == sub["label"].to_numpy())
    return {"grid": grid, "cv_accuracy_default": _class_acc(df, base_pred), "cv_accuracy_tuned": float(tuned_ok.mean()),
            "chosen_per_fold": chosen, "n": int(len(df)), "n_groups": int(len(np.unique(groups)))}
