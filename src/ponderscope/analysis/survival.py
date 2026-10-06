"""Censor-aware time-to-closure analysis.

Termination is treated as a time-to-event problem:

- **event**: a natural native think-end / reasoning closure was observed;
- **time**: generated token count to closure (or to the observation horizon);
- **censoring**: the max-token observation horizon was reached before closure.

``max_tokens`` is an *observation horizon*, never a natural stopping threshold.
Kaplan-Meier and restricted-mean estimates are implemented directly (no SciPy or
lifelines dependency) so every number is auditable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

Observation = tuple[float, bool]  # (time, event_observed)


def _as_observations(times: Iterable[float], events: Iterable[bool]) -> list[Observation]:
    obs = [(float(t), bool(e)) for t, e in zip(times, events, strict=True)]
    return obs


def kaplan_meier(times: Iterable[float], events: Iterable[bool]) -> dict[str, Any]:
    """Kaplan-Meier product-limit estimator of the closure-survival function.

    Returns step-function coordinates plus counts. ``survival[i]`` is S after the
    event time ``times[i]``. Tied times are grouped. If the largest time is an
    event with no remaining at-risk subjects, S drops to 0 there.
    """
    obs = _as_observations(times, events)
    n = len(obs)
    if n == 0:
        return {
            "n": 0,
            "n_events": 0,
            "n_censored": 0,
            "times": [],
            "survival": [],
            "at_risk": [],
            "events": [],
            "censored": [],
            "median": None,
            "max_time": None,
            "note": "no observations",
        }

    ordered = sorted(obs, key=lambda o: o[0])
    unique_times = sorted({t for t, _ in ordered})
    surv = 1.0
    out_times: list[float] = []
    out_surv: list[float] = []
    out_at_risk: list[int] = []
    out_events: list[int] = []
    out_censored: list[int] = []

    for t in unique_times:
        at_risk = sum(1 for ot, _ in ordered if ot >= t)
        d = sum(1 for ot, e in ordered if ot == t and e)
        c = sum(1 for ot, e in ordered if ot == t and not e)
        if d > 0 and at_risk > 0:
            surv *= 1.0 - d / at_risk
        out_times.append(t)
        out_surv.append(surv)
        out_at_risk.append(at_risk)
        out_events.append(d)
        out_censored.append(c)

    median: float | None = None
    for t, s in zip(out_times, out_surv, strict=True):
        if s <= 0.5:
            median = t
            break

    return {
        "n": n,
        "n_events": sum(1 for _, e in ordered if e),
        "n_censored": sum(1 for _, e in ordered if not e),
        "times": out_times,
        "survival": out_surv,
        "at_risk": out_at_risk,
        "events": out_events,
        "censored": out_censored,
        "median": median,
        "max_time": max(t for t, _ in ordered),
        "note": (
            "Product-limit closure survival; time is generated tokens to native "
            "think-end closure; censored at the observation horizon."
        ),
    }


def rmst(times: Iterable[float], events: Iterable[bool], tau: float) -> float | None:
    """Restricted mean survival time (area under KM) up to an explicit horizon tau.

    This is the restricted mean *unclosed-reasoning* time: expected generated
    tokens until closure, restricted to ``tau``. ``None`` when ``tau <= 0`` or
    there are no observations.
    """
    if tau <= 0:
        return None
    km = kaplan_meier(times, events)
    if km["n"] == 0:
        return None
    step_times = [0.0]
    step_surv = [1.0]
    for t, s in zip(km["times"], km["survival"], strict=True):
        if t <= tau:
            step_times.append(float(t))
            step_surv.append(float(s))
    area = 0.0
    for i, start in enumerate(step_times):
        end = step_times[i + 1] if i + 1 < len(step_times) else float(tau)
        if end > start:
            area += step_surv[i] * (end - start)
    return area


def survival_summary(
    times: Iterable[float],
    events: Iterable[bool],
    *,
    tau: float,
    family: str | None = None,
) -> dict[str, Any]:
    times = list(times)
    events = list(events)
    km = kaplan_meier(times, events)
    rmst_value = rmst(times, events, tau)
    return {
        "family": family,
        "tau": tau,
        "kaplan_meier": km,
        "rmst": rmst_value,
        "median_tokens_to_closure": km["median"],
        "n_at_risk_initial": km["n"],
        "n_events": km["n_events"],
        "n_censored": km["n_censored"],
        "note": (
            "If only one event exists, uncertainty is enormous; median/RMST are "
            "descriptive and should not be over-interpreted."
        ),
    }


def observations_from_records(
    records: list[dict[str, Any]],
    *,
    time_key: Callable[[dict[str, Any]], float],
    event_key: Callable[[dict[str, Any]], bool],
) -> list[Observation]:
    return [(float(time_key(r)), bool(event_key(r))) for r in records]


def survival_by_family(
    records: list[dict[str, Any]],
    *,
    time_key: Callable[[dict[str, Any]], float],
    event_key: Callable[[dict[str, Any]], bool],
    tau: float,
    min_n: int = 2,
) -> dict[str, Any]:
    """Per-family survival summaries where sample size permits."""
    families: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        families.setdefault(str(r.get("family", "?")), []).append(r)
    out: dict[str, Any] = {}
    for fam, rs in sorted(families.items()):
        if len(rs) < min_n:
            out[fam] = {
                "family": fam,
                "n": len(rs),
                "insufficient_data": True,
                "note": f"n={len(rs)} < min_n={min_n}; no per-family estimate",
            }
            continue
        summary = survival_summary(
            [time_key(r) for r in rs], [event_key(r) for r in rs], tau=tau, family=fam
        )
        summary["insufficient_data"] = False
        summary["n"] = len(rs)
        out[fam] = summary
    return out


def rmst_delta_clustered(
    a_by_task: dict[str, list[Observation]],
    b_by_task: dict[str, list[Observation]],
    *,
    tau: float,
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Task-clustered bootstrap of the RMST difference A - B.

    Resamples matched task clusters with replacement, pools observations within
    each resampled task, and recomputes RMST for both arms. This is the
    censor-aware analog of the paired cluster bootstrap used elsewhere.
    """
    keys = sorted(set(a_by_task) & set(b_by_task))
    if not keys:
        return {
            "n_clusters": 0,
            "mean": None,
            "lo": None,
            "hi": None,
            "excludes_zero": None,
            "note": "no matched task clusters",
        }

    def _rmst_for(by_task: dict[str, list[Observation]], sel: list[str]) -> float | None:
        obs = [o for k in sel for o in by_task[k]]
        if not obs:
            return None
        return rmst([o[0] for o in obs], [o[1] for o in obs], tau)

    point = _rmst_for(a_by_task, keys)
    point_b = _rmst_for(b_by_task, keys)
    point_delta = point - point_b if point is not None and point_b is not None else None

    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(keys), size=len(keys))
        sel = [keys[i] for i in idx]
        ra = _rmst_for(a_by_task, sel)
        rb = _rmst_for(b_by_task, sel)
        if ra is not None and rb is not None:
            deltas.append(ra - rb)
    if not deltas:
        return {
            "n_clusters": len(keys),
            "mean": None,
            "lo": None,
            "hi": None,
            "excludes_zero": None,
            "note": "no usable bootstrap resamples",
        }
    arr = np.asarray(deltas, dtype=float)
    lo = float(np.percentile(arr, 100 * alpha / 2))
    hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return {
        "n_clusters": len(keys),
        "n_resamples_used": int(arr.size),
        "point": point_delta,
        "mean": float(arr.mean()),
        "lo": lo,
        "hi": hi,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "tau": tau,
        "note": "task-clustered bootstrap of RMST(A) - RMST(B); positive favors A.",
    }
