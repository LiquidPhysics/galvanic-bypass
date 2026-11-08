"""Hot-cell thermal controller: minute-level bypass scheduling that keeps a pack at its rated
output while minimising each cell's time spent hot and full.

This is a separate algorithm from the fault diagnosis. It shares only the configuration file
(section ``thermal``) and the bypass-switch abstraction.

Algorithm (every ``control_period_s``, default 60 s)
-------------------------------------------------
1. **Two reference points per cell.** The band edge ``T_band`` (30 C, between the cell's 25 C
   rated condition and its 35 C storage ceiling) and the cell's own *floor*: the temperature
   of the air around it, learned from how that cell cools when idle (Newtonian cooling
   ``dT/dt = -(T - T_floor) / tau`` fitted on idle or bypassed samples; see
   :class:`FloorEstimator`).
2. **Predictive hot flag.** ``T_pred = T + trend * 600 s`` where the trend is the
   least-squares slope of the last 10 minutes. A cell is hot when ``T_pred >= T_band``, so the
   trigger fires before the cell gets there.
3. **Ranking.** Hot cells are ranked by aging rate
   ``arr(T, Ea) * exp(k_soc (SOC - SOC_ref)) / SOH^gamma`` (hotter, fuller and weaker cells
   first), divided by ``1 + off_time / rotation_quantum`` so that, when slots are scarce, cells
   take turns.
4. **Release.** A bypassed cell (it no longer makes its own heat and cools toward its floor)
   goes back in once ``T <= T_band - release_margin`` or once it has no more heat to shed
   (``T - T_floor <= floor_epsilon`` or it has stopped cooling).
5. **Three limits checked on every decision.** (a) the remaining cells deliver the requested
   power inside the inverter voltage window and the cell current limit; (b) they hold enough
   energy for the rest of the day's schedule (x reserve factor); (c) no healthy cell, which
   carries more current when others are off, is pushed above the band edge within the
   horizon. The largest set of hot cells that passes all three is bypassed; with slack every
   hot cell stays off as long as it needs, and when a limit binds the hot cells rotate.
6. **Charging rules.** Cells in hot air (floor >= 30 C) charge only to 80%, (>= 35 C) to 70%;
   no cell charges below 15 C or above its charge limit. In a series string a cell that must
   not charge is bypassed for the charge.
7. **Hard cutoff.** Any cell at or above 57 C is switched off regardless of anything else; if
   that breaks a limit, the requested power is derated instead.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..config import Config

KEL = 273.15


@dataclass
class CellReading:
    """One cell's telemetry at one control step (REON / Firebreak fields)."""

    id: str
    temp_c: float
    voltage_v: float
    current_a: float = 0.0       # + discharge, as seen by the cell
    soc: float = 0.5
    soh: float = 1.0
    capacity_ah: float = 100.0
    r_ohm: float = 1.0e-3
    v_nominal: float = 3.1
    charge_max_c: float | None = None


class FloorEstimator:
    """Learns a cell's floor (surrounding-air temperature) and cooling time constant.

    Newtonian cooling while idle or bypassed: ``dT/dt = a + b * x`` with ``x = T - T_amb``
    (or ``x = T`` when no ambient sensor), so ``tau = -1/b`` and the floor offset is ``-a/b``.
    Regression sums are accumulated with exponential forgetting per idle segment so the
    estimate tracks slow changes. ``floor(t) = T_amb(t) + offset`` when an ambient sensor
    exists, otherwise the absolute fitted floor.
    """

    def __init__(self, cfg: Config) -> None:
        th = cfg.thermal
        self.min_points = int(th.floor_min_idle_points)
        self.min_drop = float(th.floor_min_drop_k)
        self.forget = float(th.floor_forgetting)
        self.tau_lo, self.tau_hi = (float(x) for x in th.tau_bounds_s)
        self.S = np.zeros(5)  # n, sx, sy, sxx, sxy
        self.prev: tuple[float, float, float] | None = None
        self.seg_points = 0
        self.seg_tmax = -math.inf
        self.seg_tmin = math.inf
        self.offset: float | None = None
        self.floor_abs: float | None = None
        self.tau: float | None = None
        self.n_fits = 0
        self.relative = True

    def update(self, t_s: float, temp_c: float, idle: bool, ambient_c: float | None) -> None:
        if not idle or not math.isfinite(temp_c):
            self._close_segment()
            self.prev = None
            return
        rel = ambient_c is not None and math.isfinite(ambient_c)
        self.relative = rel
        x = temp_c - (ambient_c if rel else 0.0)  # type: ignore[operator]
        if self.prev is not None:
            t0, T0, x0 = self.prev
            dt = t_s - t0
            if dt > 0:
                y = (temp_c - T0) / dt
                xm = 0.5 * (x + x0)
                self.S += np.array([1.0, xm, y, xm * xm, xm * y])
                self.seg_points += 1
        self.seg_tmax = max(self.seg_tmax, temp_c)
        self.seg_tmin = min(self.seg_tmin, temp_c)
        self.prev = (t_s, temp_c, x)
        self._fit()

    def _close_segment(self) -> None:
        if self.seg_points:
            self.S *= self.forget
        self.seg_points = 0
        self.seg_tmax, self.seg_tmin = -math.inf, math.inf

    def _fit(self) -> None:
        n, sx, sy, sxx, sxy = self.S
        if n < self.min_points:
            return
        den = n * sxx - sx * sx
        if den <= 1e-12:
            return
        b = (n * sxy - sx * sy) / den
        a = (sy - b * sx) / n
        if b >= 0:
            return
        tau = -1.0 / b
        if not (self.tau_lo <= tau <= self.tau_hi):
            return
        if self.n_fits == 0 and (self.seg_tmax - self.seg_tmin) < self.min_drop:
            return
        self.tau = tau
        val = -a / b
        if self.relative:
            self.offset = val
        else:
            self.floor_abs = val
        self.n_fits += 1

    def floor(self, ambient_c: float | None, fallback_c: float) -> tuple[float, str]:
        if self.offset is not None and ambient_c is not None and math.isfinite(ambient_c):
            return ambient_c + self.offset, "learned (ambient + offset)"
        if self.floor_abs is not None:
            return self.floor_abs, "learned"
        if ambient_c is not None and math.isfinite(ambient_c):
            return ambient_c, "ambient sensor (not yet learned)"
        return fallback_c, "fallback (not yet learned)"


@dataclass
class CellState:
    floor: FloorEstimator
    history: deque[tuple[float, float]]
    bypassed: bool = False
    reason: str = "in"
    off_since: float | None = None
    on_since: float | None = None
    last_off_reason: str = ""
    off_accum_s: float = 0.0           # decaying off-time used for turn-taking
    off_total_s: float = 0.0           # undecayed, for reporting
    last_t: float | None = None
    rearm_above_c: float = -math.inf   # after a "no heat left" release: must warm past this to be flagged again
    switches: int = 0
    heat_trend_k_per_s: float = 0.0     # heating trend when it was switched off hot (for the release test)
    # rule latches (hysteresis so noisy SOC/temperature readings cannot toggle a switch every minute)
    latch_cut: bool = False
    latch_empty: bool = False
    latch_full: bool = False
    latch_cold: bool = False
    latch_warm: bool = False
    cap_rules: set[float] = field(default_factory=set)   # hot-air cap rules (floor thresholds) latched for this cell


@dataclass
class CellDecision:
    id: str
    in_string: bool
    action: str          # IN | BYPASS_HOT | BYPASS_CUTOFF | BYPASS_CHARGE_RULE | BYPASS_SOC_LIMIT | BYPASS_HOLD | HOT_WAITING
    reason: str
    temp_c: float
    temp_pred_c: float
    trend_k_per_min: float
    floor_c: float
    floor_source: str
    tau_s: float | None
    aging_rate: float
    priority: float
    soc: float
    soc_cap: float
    off_accum_s: float


@dataclass
class Decision:
    t_s: float
    request_w: float
    granted_w: float
    n_in: int
    string_v: float
    string_current_a: float
    binding: list[str]
    rotation_active: bool
    hot_cells: list[str]
    bypassed: list[str]
    checks: dict[str, Any]
    cells: list[CellDecision] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def _json_safe(x: Any) -> Any:
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


def series_runtime_ah(usable_ah: np.ndarray, m: int) -> float:
    """Longest charge throughput (Ah per in-string cell) a series string can deliver when at
    least ``m`` cells must stay in and cells may be bypassed once empty: the largest ``t`` with
    ``sum_k min(a_k, t) >= m t`` (water-filling over the cells' usable Ah ``a_k``)."""
    a = np.sort(np.clip(np.asarray(usable_ah, dtype=float), 0.0, None))
    N = a.size
    if m <= 0 or N < m:
        return 0.0
    S = 0.0
    for j in range(N + 1):          # j = number of the smallest cells already exhausted
        if N - j < m:
            t = S / (m - (N - j))
            if j == N or t <= a[j]:
                return float(t)
        if j < N:
            S += a[j]
    return float(S / m)


MANDATORY_ACTIONS = ("BYPASS_CUTOFF", "BYPASS_CHARGE_RULE", "BYPASS_SOC_LIMIT")


class ThermalController:
    """Stateful controller; call :meth:`step` once per control period.

    ``request_w`` is the site's power request for the next period (W, positive = discharge,
    negative = charge). ``remaining_energy_wh`` is the energy the pack must still supply from
    its present charge before the schedule refills it (the maximum of the cumulative net
    scheduled discharge from now to the end of the day); ``None`` disables the energy check.
    ``ambient_c`` is the enclosure/room sensor, if any; floors are then learned as offsets
    from it so they follow the daily cycle.
    """

    def __init__(self, cfg: Config, cell_ids: Sequence[str]) -> None:
        self.cfg = cfg
        self.th = cfg.thermal
        self.ids = list(cell_ids)
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("cell ids must be unique")
        maxlen = 4 * max(3, int(float(self.th.trend_window_s) / float(self.th.control_period_s)) + 2)
        self.state = {c: CellState(FloorEstimator(cfg), deque(maxlen=maxlen)) for c in self.ids}
        self.requests: deque[tuple[float, float]] = deque()
        self.last_ambient_c: float | None = None
        self.cap_lifted: set[str] = set()   # hot-air charge caps lifted for energy (kept with hysteresis)

    # ------------------------------------------------------------------ helpers
    def aging_rate(self, temp_c: float, soc: float, soh: float) -> float:
        """``exp(Ea/Rg (1/T25 - 1/T)) * exp(k_soc (SOC - SOC_ref)) / SOH^gamma`` (relative to 25 C, SOC_ref, new)."""
        th = self.th
        arr = math.exp(float(th.aging_ea_j_per_mol) / 8.314 * (1.0 / (25.0 + KEL) - 1.0 / (temp_c + KEL)))
        socf = math.exp(float(th.aging_k_soc) * ((soc if math.isfinite(soc) else 0.5) - float(th.aging_soc_ref)))
        sohf = max(soh if math.isfinite(soh) else 1.0, 0.05) ** float(th.aging_soh_gamma)
        return arr * socf / sohf

    def soc_cap(self, floor_c: float, active: set[float] | None = None) -> float:
        """Charge ceiling for a cell whose surrounding air (floor) is ``floor_c``.

        With ``active`` (the cell's latched rules, updated in place) a rule that applies stays
        applied until the floor falls ``temp_hysteresis_k`` below its threshold, so a floor
        hovering at 30.0 C cannot flip the cap (and the cell's switch) every minute."""
        cap = float(self.th.charge_soc_max)
        hy = float(self.th.temp_hysteresis_k)
        for rule in sorted(self.th.charge_soc_caps, key=lambda r: r["floor_c"]):
            fc = float(rule["floor_c"])
            on = floor_c >= fc or (active is not None and fc in active and floor_c > fc - hy)
            if active is not None:
                if on:
                    active.add(fc)
                else:
                    active.discard(fc)
            if on:
                cap = min(cap, float(rule["soc_max"]))
        return cap

    def _trend(self, st: CellState) -> float:
        """Least-squares slope (K/s) over the last ``trend_window_s``."""
        if not st.history:
            return 0.0
        t_end = st.history[-1][0]
        pts = [(t, T) for t, T in st.history if t >= t_end - float(self.th.trend_window_s) - 1e-9]
        if len(pts) < int(self.th.trend_min_points):
            return 0.0
        t = np.array([p[0] for p in pts])
        T = np.array([p[1] for p in pts])
        tc = t - t.mean()
        den = float((tc * tc).sum())
        return float((tc * (T - T.mean())).sum() / den) if den > 0 else 0.0

    @staticmethod
    def string_operating_point(p_w: float, e_v: float, r_ohm: float) -> tuple[float, float]:
        """Current and terminal voltage of a string with open-circuit voltage ``e_v`` and
        resistance ``r_ohm`` exchanging ``p_w`` (W, + discharge): ``P = I (E - I R)``, smaller
        root. ``nan`` current when the power cannot be delivered at any current."""
        if e_v <= 0:
            return float("nan"), 0.0
        if abs(p_w) < 1e-9:
            return 0.0, e_v
        disc = e_v * e_v - 4.0 * r_ohm * p_w
        if disc < 0:
            return float("nan"), float("nan")
        i = (e_v - math.sqrt(disc)) / (2.0 * r_ohm) if r_ohm > 0 else p_w / e_v
        return float(i), float(e_v - i * r_ohm)

    def floors(self, ambient_c: float | None = None) -> dict[str, dict[str, Any]]:
        """Current floor estimate of every cell (for dashboards); uses the last ambient reading
        when ``ambient_c`` is not given."""
        ambient_c = self.last_ambient_c if ambient_c is None else ambient_c
        out = {}
        for cid, st in self.state.items():
            last = st.history[-1][1] if st.history else float("nan")
            f, src = st.floor.floor(ambient_c, fallback_c=last)
            out[cid] = {"floor_c": f, "source": src, "tau_s": st.floor.tau, "n_fits": st.floor.n_fits}
        return out

    # ------------------------------------------------------------------ main step
    def step(self, t_s: float, cells: Sequence[CellReading], request_w: float, *, ambient_c: float | None = None,
             remaining_energy_wh: float | None = None, remaining_charge_wh: float | None = None) -> Decision:
        th = self.th
        by_id = {c.id: c for c in cells}
        if set(by_id) != set(self.ids) or len(cells) != len(self.ids):
            raise ValueError("every step must list the same cells the controller was created with")
        ids = self.ids
        n = len(ids)
        idle_c = float(th.idle_current_c_rate)
        memory = float(th.rotation_memory_s)
        self.last_ambient_c = ambient_c
        # --- 1. update estimators, trends and turn-taking memory
        for cid in ids:
            c, st = by_id[cid], self.state[cid]
            idle = st.bypassed or abs(c.current_a) <= idle_c * c.capacity_ah
            st.floor.update(t_s, c.temp_c, idle, ambient_c)
            st.history.append((t_s, c.temp_c))
            if st.last_t is not None:
                dt = max(0.0, t_s - st.last_t)
                st.off_accum_s *= math.exp(-dt / memory) if memory > 0 else 1.0
                if st.bypassed:
                    st.off_accum_s += dt
                    st.off_total_s += dt
            st.last_t = t_s
        # request envelope: a new hot cell is only bypassed if the limits also hold at the largest
        # charge and discharge requests of the last few minutes (so a fluctuating request cannot
        # make it come straight back)
        self.requests.append((t_s, float(request_w)))
        while self.requests and self.requests[0][0] < t_s - float(th.request_envelope_s) - 1e-9:
            self.requests.popleft()
        env = [p for _, p in self.requests]
        env_dis, env_chg = max(max(env), 0.0), min(min(env), 0.0)
        T = np.array([by_id[c].temp_c for c in ids], dtype=float)
        V = np.array([by_id[c].voltage_v for c in ids], dtype=float)
        I_now = np.array([0.0 if self.state[c].bypassed else by_id[c].current_a for c in ids], dtype=float)
        R = np.array([by_id[c].r_ohm for c in ids], dtype=float)
        soc = np.array([by_id[c].soc for c in ids], dtype=float)
        soh = np.array([by_id[c].soh for c in ids], dtype=float)
        capah = np.array([by_id[c].capacity_ah for c in ids], dtype=float)
        vnom = np.array([by_id[c].v_nominal for c in ids], dtype=float)
        E_cell = V + I_now * R                       # open-circuit estimate of each cell
        trend = np.array([self._trend(self.state[c]) for c in ids])
        h = float(th.trend_horizon_s)
        T_pred = T + np.maximum(trend, 0.0) * h      # a cooling trend never moves the flag earlier
        floors, fsrc, taus = [], [], []
        med = float(np.median(T))
        for k, cid in enumerate(ids):
            f, s = self.state[cid].floor.floor(ambient_c, fallback_c=float(min(T[k], med)))
            floors.append(f)
            fsrc.append(s)
            taus.append(self.state[cid].floor.tau)
        floor = np.array(floors)
        band = float(th.band_edge_c)
        caps_rule = np.array([self.soc_cap(f, self.state[cid].cap_rules) for f, cid in zip(floor, ids, strict=True)])
        aging = np.array([self.aging_rate(T[k], soc[k], soh[k]) for k in range(n)])
        charging = request_w < 0
        discharging = request_w > 0
        e_req = None if remaining_energy_wh is None else float(th.energy_reserve_factor) * float(remaining_energy_wh)
        v_end = float(th.energy_v_end)
        soc_lo = float(th.energy_soc_min)
        # discharge current used for the energy estimate: the present request, else a typical C-rate
        n_cells = max(n, 1)
        p_eval = max(env_dis, float(request_w), 0.0)
        i_eval = (p_eval / (n_cells * float(np.mean(vnom)))) if p_eval > 0 else \
            float(th.energy_eval_c_rate) * float(np.median(capah))
        v_end_loaded = max(v_end - i_eval * float(np.median(R)), 0.5)
        m_end = max(1, math.ceil(float(th.inverter_v_min) / v_end_loaded))
        # cell voltage vs SOC is close to linear for sloping Na-ion OCV: anchor a line through
        # (soc_lo, v_end) and the cell's present (SOC, open-circuit estimate)
        span = soc - soc_lo
        slope = np.where(span > 0.05, (np.maximum(E_cell, v_end) - v_end) / np.maximum(span, 1e-9),
                         (vnom - v_end) / 0.45)

        def deliverable_wh(mask: np.ndarray, soc_target: np.ndarray | None = None) -> float:
            """Energy the cells in ``mask`` can deliver as a series string that must keep at least
            ``m_end`` cells in at the end of discharge (bypass lets cells drop out as they empty).
            With fewer than ``m_end`` cells the string reaches the inverter minimum first: each cell
            then stops at ``v_min / n_in`` (loaded), a higher SOC, and none can drop out."""
            s_ = soc if soc_target is None else soc_target
            n_in = int(mask.sum())
            if n_in == 0:
                return 0.0
            lo = np.full(n, soc_lo)
            m_req = m_end
            if n_in < m_end:
                dv = float(th.inverter_v_min) / n_in - v_end_loaded
                lo = soc_lo + dv / np.maximum(slope, 1e-6)
                m_req = n_in
            a = np.clip(s_ - lo, 0.0, None)[mask] * capah[mask]
            t_star = series_runtime_ah(a, m_req)
            used = np.minimum(a, t_star)                       # Ah each cell gives
            frac = used / np.maximum(capah[mask], 1e-9)          # SOC span each cell discharges
            top = np.clip(s_[mask] - soc_lo, 0.0, None)
            v_mean = v_end + slope[mask] * (top - 0.5 * frac) - i_eval * R[mask]   # loaded mean voltage
            return float((used * v_mean).sum())

        def soc_after_charge(cap_vec: np.ndarray, mask: np.ndarray) -> np.ndarray:
            """SOC each cell reaches if charging continues while the string stays inside the voltage
            window: cells drop out at their cap, and at least ``m_chg`` must remain in."""
            v_full = np.maximum(v_end + slope * (cap_vec - soc_lo), 1e-3)
            m_chg = max(1, math.ceil(float(th.inverter_v_min) / float(np.median(v_full[mask])))) if mask.any() else 1
            head = np.clip(cap_vec - soc, 0.0, None)[mask] * capah[mask]
            t_c = series_runtime_ah(head, m_chg)
            if remaining_charge_wh is not None:
                # the charge window ends: no more than the scheduled charge energy goes in
                v_mid = v_end + slope[mask] * (np.clip(soc[mask] - soc_lo, 0, None)
                                                + 0.5 * head / np.maximum(capah[mask], 1e-9))

                def e_in(t: float) -> float:
                    return float((np.minimum(head, t) * v_mid).sum())

                if e_in(t_c) > float(remaining_charge_wh):
                    lo_, hi_ = 0.0, t_c
                    for _ in range(50):
                        mid_ = 0.5 * (lo_ + hi_)
                        lo_, hi_ = (mid_, hi_) if e_in(mid_) <= float(remaining_charge_wh) else (lo_, mid_)
                    t_c = lo_
            out_ = soc.copy()
            out_[mask] = soc[mask] + np.minimum(head, t_c) / np.maximum(capah[mask], 1e-9)
            return out_

        # energy-aware charge caps: the reduced level never leaves the pack unable to cover the rest of
        # the day's schedule; if it would, caps are lifted for the cells in the coolest air first
        # A lifted cap stays lifted (hysteresis) until the projection clears the need by
        # ``cap_relax_release_margin`` without it, hottest air first; otherwise a projection hovering at the
        # threshold would flip a cap, and with it the cell's switch, every minute.
        caps = caps_rule.copy()
        soc_full = float(th.charge_soc_max)
        projected_wh: float | None = None
        relax = e_req is not None and bool(th.cap_relax_for_energy)
        self.cap_lifted &= {ids[k] for k in range(n) if caps_rule[k] < soc_full} if relax else set()
        if relax and (caps_rule < soc_full).any():
            assert e_req is not None
            chg_ok = (T >= float(th.charge_min_temp_c)) & (T <= float(th.charge_max_temp_c)) & (T < float(th.hard_cutoff_c))
            allm = np.ones(n, bool)

            def projection(c: np.ndarray) -> float:
                return deliverable_wh(allm, soc_after_charge(c, chg_ok))

            for k in range(n):
                if ids[k] in self.cap_lifted:
                    caps[k] = soc_full
            projected_wh = projection(caps)
            if projected_wh < e_req:
                for k in sorted(np.where(caps < soc_full)[0], key=lambda k: floor[k]):   # coolest air first
                    caps[k] = soc_full
                    self.cap_lifted.add(ids[k])
                    projected_wh = projection(caps)
                    if projected_wh >= e_req:
                        break
            else:
                need = e_req * (1.0 + float(th.cap_relax_release_margin))
                for k in sorted([k for k in range(n) if ids[k] in self.cap_lifted], key=lambda k: -floor[k]):
                    trial_caps = caps.copy()
                    trial_caps[k] = caps_rule[k]
                    p_trial = projection(trial_caps)
                    if p_trial < need:
                        break
                    caps, projected_wh = trial_caps, p_trial
                    self.cap_lifted.discard(ids[k])
        cap_relaxed = [ids[k] for k in range(n) if ids[k] in self.cap_lifted]
        # --- 2. mandatory off: hard cutoff (latched), SOC limits, charge rules (all with hysteresis)
        mandatory: dict[int, tuple[str, str]] = {}
        cut, cut_rel = float(th.hard_cutoff_c), float(th.hard_cutoff_release_c)
        sh, thy = float(th.soc_hysteresis), float(th.temp_hysteresis_k)
        dmin = float(th.discharge_soc_min)
        cmin = float(th.charge_min_temp_c)
        for k, cid in enumerate(ids):
            st = self.state[cid]
            cm = by_id[cid].charge_max_c
            cmax = float(cm) if cm is not None else float(th.charge_max_temp_c)
            st.latch_cut = T[k] >= cut or (st.latch_cut and T[k] > cut_rel)
            st.latch_empty = soc[k] <= dmin + 1e-12 or (st.latch_empty and soc[k] < dmin + sh)
            st.latch_full = soc[k] >= caps[k] - 1e-9 or (st.latch_full and soc[k] > caps[k] - sh)
            st.latch_cold = T[k] < cmin or (st.latch_cold and T[k] < cmin + thy)
            st.latch_warm = T[k] > cmax or (st.latch_warm and T[k] > cmax - thy)
            if st.latch_cut:
                mandatory[k] = ("BYPASS_CUTOFF", f"{T[k]:.1f} C >= hard cutoff {cut:g} C" if T[k] >= cut else
                                f"latched after a {cut:g} C cutoff until it cools to {cut_rel:g} C (now {T[k]:.1f} C)")
            elif discharging and st.latch_empty:
                mandatory[k] = ("BYPASS_SOC_LIMIT", f"SOC {soc[k]:.0%} at its {dmin:.0%} discharge floor")
            elif charging and st.latch_cold:
                mandatory[k] = ("BYPASS_CHARGE_RULE", f"{T[k]:.1f} C below the {cmin:g} C charge minimum")
            elif charging and st.latch_warm:
                mandatory[k] = ("BYPASS_CHARGE_RULE", f"{T[k]:.1f} C above its {cmax:g} C charge limit")
            elif charging and st.latch_full:
                why = (f"floor {floor[k]:.1f} C is hot air: charge capped at {caps[k]:.0%}"
                       if caps[k] < float(th.charge_soc_max) else "full")
                mandatory[k] = ("BYPASS_CHARGE_RULE", f"SOC {soc[k]:.0%} reached its cap ({why})")
        # --- 3. release of bypassed hot cells, then hot flags
        release = np.zeros(n, bool)
        release_kind = [""] * n
        margin = float(th.release_margin_k)
        use_trend = bool(th.release_uses_heating_trend)
        for k, cid in enumerate(ids):
            st = self.state[cid]
            if T[k] <= band - margin:
                st.rearm_above_c = -math.inf
            if st.bypassed and st.reason == "BYPASS_HOT":
                off_for = t_s - (st.off_since if st.off_since is not None else t_s)
                t_rel = T[k] + (st.heat_trend_k_per_s * h if use_trend else 0.0)
                if t_rel <= band - margin:
                    release[k], release_kind[k] = True, "cooled"
                elif (T[k] - floor[k]) <= float(th.floor_epsilon_k):
                    release[k], release_kind[k] = True, "at_floor"
                elif off_for >= float(th.min_off_s) and (-trend[k] * 60.0) < float(th.floor_rate_epsilon_k_per_min):
                    release[k], release_kind[k] = True, "stalled"
        rearm = float(th.rearm_margin_k)
        min_on = float(th.min_on_s)

        def rested(st: CellState) -> bool:
            return not (st.last_off_reason == "BYPASS_HOT" and st.on_since is not None and t_s - st.on_since < min_on)

        hot = [k for k in range(n) if k not in mandatory and not release[k] and T_pred[k] >= band
               and (T_pred[k] - floor[k]) >= rearm and T[k] >= self.state[ids[k]].rearm_above_c
               and (self.state[ids[k]].bypassed or rested(self.state[ids[k]]))]
        still_off = [k for k, cid in enumerate(ids) if k not in mandatory and self.state[cid].bypassed
                     and self.state[cid].reason == "BYPASS_HOT" and not release[k]]
        cand = sorted(set(hot) | set(still_off))
        quantum = float(th.rotation_quantum_s)
        prio = np.array([aging[k] / (1.0 + self.state[ids[k]].off_accum_s / quantum) for k in range(n)])
        def dwell_s(st: CellState) -> float:
            return t_s - st.off_since if st.off_since is not None else math.inf

        locked = [k for k in cand if self.state[ids[k]].bypassed and dwell_s(self.state[ids[k]]) < float(th.min_off_s)]
        cand_set = set(cand)
        # a cell that went off for a rule (cutoff/SOC/charge) that no longer applies stays off for the
        # minimum dwell if the limits allow, so a request that flips sign every minute cannot toggle it.
        # A SOC or charge rule whose latch is still set also holds while the request keeps returning to the
        # direction that triggered it (inside the request envelope), e.g. a full cell under a regulation signal.
        def rule_hold(st: CellState) -> bool:
            if st.reason == "BYPASS_CHARGE_RULE":
                return env_chg < 0 and (st.latch_full or st.latch_cold or st.latch_warm)
            if st.reason == "BYPASS_SOC_LIMIT":
                return env_dis > 0 and st.latch_empty
            return False

        hold = [k for k, cid in enumerate(ids) if k not in mandatory and k not in cand_set and self.state[cid].bypassed
                and self.state[cid].off_since is not None
                and (dwell_s(self.state[cid]) < float(th.min_off_s) or rule_hold(self.state[cid]))]
        order = (sorted(locked, key=lambda k: -prio[k]) + hold
                 + sorted([k for k in cand if k not in locked], key=lambda k: -prio[k]))
        Cth = float(th.heat_capacity_j_per_k)

        # --- 4. the three limits
        healthy = np.array([k not in cand_set for k in range(n)])
        tau_arr = np.array([t if t else h for t in taus], dtype=float)
        att = tau_arr / h * (1.0 - np.exp(-h / tau_arr)) if h > 0 else np.ones(n)
        heat_gain = R * h / Cth * att            # K per A^2 of extra current over the horizon
        # the controller's string model (open-circuit estimate + ohmic R) does not see polarisation or the
        # switch resistance, so decisions keep a margin inside the inverter window
        v_lo = float(th.inverter_v_min) + float(th.inverter_v_margin_v)
        v_hi = float(th.inverter_v_max) - float(th.inverter_v_margin_v)
        check_energy = bool(th.check_energy)
        check_band = bool(th.check_band_protection)

        admit_margin = float(th.energy_admit_margin)

        def check(off: set[int], p_w: float, i_base: float | None = None, energy: bool = True,
                  admit: bool = False) -> dict[str, Any]:
            """Power/voltage window always; with ``i_base`` (the current with only the safety set
            off) also the energy reserve and band protection, which are limits on *extra* cells off.
            ``admit``: the decision switches a new cell off, which needs ``energy_admit_margin`` more
            energy than keeping one off (hysteresis at the energy threshold)."""
            on = np.ones(n, bool)
            if off:
                on[list(off)] = False
            n_in = int(on.sum())
            out: dict[str, Any] = {"n_in": n_in, "i": float("nan"), "v": 0.0, "energy_in_wh": None,
                                   "energy_required_wh": e_req, "healthy_max_pred_c": None}
            if n_in == 0:
                out.update(ok=False, violations=["power"])
                return out
            i, v = self.string_operating_point(p_w, float(E_cell[on].sum()), float(R[on].sum()))
            out.update(i=i, v=v)
            viol: list[str] = []
            if not math.isfinite(i) or abs(i) > float(th.cell_current_max_a) + 1e-9:
                viol.append("power")
            if not math.isfinite(v) or v < v_lo - 1e-9 or v > v_hi + 1e-9:
                viol.append("voltage_window")
            if i_base is not None and math.isfinite(i):
                if energy and check_energy and e_req is not None and p_w > 0:
                    out["energy_in_wh"] = deliverable_wh(on)
                    if out["energy_in_wh"] < e_req * ((1.0 + admit_margin) if admit else 1.0):
                        viol.append("energy")
                m = on & healthy
                if m.any():
                    tp_trial = T_pred[m] + np.maximum((i * i - I_now[m] ** 2) * heat_gain[m], 0.0)
                    tp_base = T_pred[m] + np.maximum((i_base * i_base - I_now[m] ** 2) * heat_gain[m], 0.0)
                    out["healthy_max_pred_c"] = float(tp_trial.max())
                    # "pushed above the band": a healthy cell that would stay at or below the band with only
                    # the safety set off must not cross it (a cell already above it from its own air is not
                    # pushed across, and does not veto every other decision)
                    if check_band and bool(((tp_trial > band) & (tp_base <= band) & (tp_trial > tp_base + 1e-6)).any()):
                        viol.append("band_protection")
            out.update(ok=not viol, violations=viol)
            return out

        chosen = set(mandatory)
        binding: list[str] = []
        # Rule bypasses (cutoff, SOC, charge rules) are checked against the power/voltage limit and the
        # power is derated if they break it. Band protection applies to the hot-cell decisions; it is
        # applied to the charge rules too only if configured (it slows charging, which costs energy).
        not_cut = np.array([not (k in mandatory and mandatory[k][0] == "BYPASS_CUTOFF") for k in range(n)])
        band_on_rules = charging and check_band and bool(th.band_protection_on_charge_rules)

        def base_check(p_w: float) -> dict[str, Any]:
            if not band_on_rules:
                return check(chosen, p_w)
            i_ref, _ = self.string_operating_point(p_w, float(E_cell[not_cut].sum()), float(R[not_cut].sum()))
            return check(chosen, p_w, i_ref if math.isfinite(i_ref) else None, energy=False)

        res = base_check(request_w)
        granted = float(request_w)
        derated = False
        if not res["ok"]:
            # the rule set alone breaks a limit: keep the rule set and derate the power
            derated = True
            binding.extend(f"derate:{v}" for v in res["violations"])
            lo, hi = 0.0, abs(float(request_w))
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if base_check(math.copysign(mid, request_w))["ok"]:
                    lo = mid
                else:
                    hi = mid
            granted = math.copysign(lo, request_w) if request_w else 0.0
        i_base = check(chosen, granted)["i"]
        res = check(chosen, granted, i_base if math.isfinite(i_base) else None)
        locked_set = set(locked)

        def env_ok(trial: set[int]) -> bool:
            for p_env in {env_dis, env_chg} - {0.0, float(granted)}:
                ib = check(chosen_rules, p_env)["i"]
                if not check(trial, p_env, ib if math.isfinite(ib) else None)["ok"]:
                    return False
            return True

        chosen_rules = set(chosen)
        # marginal aging benefit: heat the cell stops making, weighted by how fast aging rises with its
        # temperature, must exceed the extra heat the higher string current puts into the cells that stay
        # in, weighted the same way (bypass at constant power raises the current through everyone else)
        dadT = aging * float(th.aging_ea_j_per_mol) / (8.314 * (T + KEL) ** 2)
        r_sw = float(th.switch_r_ohm)
        need_benefit = bool(th.require_net_aging_benefit)

        def net_benefit(trial: set[int], k: int, i_cur: float, i_new: float) -> float:
            if not (math.isfinite(i_cur) and math.isfinite(i_new)):
                return -math.inf
            on_new = np.ones(n, bool)
            on_new[list(trial)] = False
            saved = (i_cur * i_cur * R[k] - i_new * i_new * r_sw) * dadT[k]
            added = float(((i_new * i_new - i_cur * i_cur) * (R[on_new] + r_sw) * dadT[on_new]).sum())
            return float(saved - added)

        benefit: dict[str, float] = {}
        hold_set = set(hold)
        for k in order:
            trial = chosen | {k}
            r = check(trial, granted, i_base if math.isfinite(i_base) else None, admit=not self.state[ids[k]].bypassed)
            if r["ok"] and need_benefit and k not in locked_set and k not in hold_set:
                nb = net_benefit(trial, k, float(res["i"]), float(r["i"]))
                benefit[ids[k]] = nb
                if nb <= 0:
                    if "no_net_benefit" not in binding:
                        binding.append("no_net_benefit")
                    continue
            if r["ok"] and (k in locked_set or k in hold_set or env_ok(trial)):
                chosen, res = trial, r
            elif r["ok"]:
                if "request_envelope" not in binding:
                    binding.append("request_envelope")
            else:
                binding.extend(v for v in r["violations"] if v not in binding)
        waiting = [k for k in cand if k not in chosen]
        # --- 5. apply
        decisions: list[CellDecision] = []
        for k, cid in enumerate(ids):
            st = self.state[cid]
            if k in mandatory:
                action, why = mandatory[k]
            elif k in chosen and k in hold:
                action = "BYPASS_HOLD"
                why = (f"held off for the {float(th.min_off_s):g} s minimum dwell after {st.reason.lower()}"
                       if st.off_since is not None and t_s - st.off_since < float(th.min_off_s) else
                       f"held off: {st.reason.lower()} still latched and the request keeps returning to it")
            elif k in chosen:
                action = "BYPASS_HOT"
                why = (f"predicted {T_pred[k]:.1f} C (now {T[k]:.1f} C, {trend[k] * 60:+.2f} K/min) >= band {band:g} C;"
                       f" aging rank {prio[k]:.2f}" if k in hot else f"cooling toward its floor {floor[k]:.1f} C")
            elif k in cand_set:
                action = "HOT_WAITING"
                why = "hot, but a limit binds (" + ", ".join(sorted({b.split(':')[-1] for b in binding})) + "): taking turns"
            else:
                action = "IN"
                why = "in string"
                if release[k]:
                    why = {"cooled": f"released: cooled to {T[k]:.1f} C, {margin:g} K below the band"
                                         + (" (with its heating trend)" if use_trend and st.heat_trend_k_per_s > 0 else ""),
                           "at_floor": f"released: at its floor {floor[k]:.1f} C, no heat left to shed",
                           "stalled": "released: stopped cooling, no heat left to shed"}[release_kind[k]]
                    if release_kind[k] in ("at_floor", "stalled"):
                        st.rearm_above_c = float(T[k]) + rearm
            off = action.startswith("BYPASS")
            if off != st.bypassed:
                st.switches += 1
            if off and not st.bypassed:
                st.off_since = t_s
                st.heat_trend_k_per_s = max(float(trend[k]), 0.0) if action == "BYPASS_HOT" else 0.0
            if not off and st.bypassed:
                st.on_since = t_s
                st.last_off_reason = st.reason
            if not off:
                st.off_since = None
            st.bypassed = off
            if action != "BYPASS_HOLD":
                st.reason = action if off else "in"
            decisions.append(CellDecision(cid, not off, action, why, float(T[k]), float(T_pred[k]), float(trend[k] * 60.0),
                                          float(floor[k]), fsrc[k], taus[k], float(aging[k]), float(prio[k]), float(soc[k]),
                                          float(caps[k]), float(st.off_accum_s)))
        return Decision(t_s=float(t_s), request_w=float(request_w), granted_w=float(granted), n_in=int(res["n_in"]),
                        string_v=float(res["v"]), string_current_a=float(res["i"]),
                        binding=sorted(set(binding)), rotation_active=bool(waiting),
                        hot_cells=[ids[k] for k in cand], bypassed=[d.id for d in decisions if not d.in_string],
                        checks={"ok": bool(res["ok"]), "violations_after_decision": list(res["violations"]),
                                "derated": derated, "mandatory": {ids[k]: v[0] for k, v in mandatory.items()},
                                "locked": [ids[k] for k in locked], "held": [ids[k] for k in hold if k in chosen],
                                "waiting": [ids[k] for k in waiting],
                                "energy_in_wh": res["energy_in_wh"], "energy_required_wh": res["energy_required_wh"],
                                "cap_relaxed": cap_relaxed, "projected_after_charge_wh": projected_wh,
                                "net_benefit": benefit,
                                "healthy_max_pred_c": res["healthy_max_pred_c"]},
                        cells=decisions)


def readings_from_frame(frame: Mapping[str, Any]) -> tuple[float, list[CellReading], float | None, float]:
    """Parse one Firebreak/REON live JSON frame
    ``{"t", "ambient_C", "request_W", "cells": [{"id", "V", "I", "T", "soc", "soh", "R_mOhm"}]}``."""
    cells = []
    for c in frame["cells"]:
        cells.append(CellReading(id=str(c["id"]), temp_c=float(c["T"]), voltage_v=float(c["V"]), current_a=float(c.get("I", 0.0)),
                                 soc=float(c.get("soc", 0.5)), soh=float(c.get("soh", 1.0)),
                                 capacity_ah=float(c.get("capacity_Ah", 100.0)),
                                 r_ohm=float(c.get("R_mOhm", 1.0)) / 1000.0, v_nominal=float(c.get("V_nominal", 3.1))))
    amb = frame.get("ambient_C")
    return float(frame["t"]), cells, (float(amb) if amb is not None else None), float(frame.get("request_W", frame.get("pack_power_W", 0.0)))
