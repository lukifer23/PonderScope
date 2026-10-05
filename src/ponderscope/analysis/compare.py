"""Compare two deployment configurations using matched evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

from ..evidence.run import RunStore
from .stats import classify_effect, paired_bootstrap_delta

_METRICS: dict[str, Callable[[dict[str, Any]], float]] = {
    "accuracy": lambda r: 1.0 if r["correct"] else 0.0,
    "reasoning_tokens": lambda r: float(r["reasoning_tokens"]),
    "total_tokens": lambda r: float(r["total_tokens"]),
    "wall_ms": lambda r: float(r["trace"]["wall_ms"]),
    "repeated_ngram_fraction_4": lambda r: float(r["metrics"]["repeated_ngram_fraction_4"]),
    "unique_token_ratio": lambda r: float(r["metrics"]["unique_token_ratio"]),
    "terminated_by_eos": lambda r: 1.0 if r["termination"]["terminated_by_eos"] else 0.0,
    "missing_answer": lambda r: 1.0 if r["termination"]["missing_answer"] else 0.0,
}


def _key(r: dict[str, Any]) -> tuple:
    c = r["condition"]
    return (r["task_id"], c["mode"], c.get("seed"), c.get("repeat"))


def _index(traces: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    out: dict[tuple, dict[str, Any]] = {}
    for r in traces:
        out[_key(r)] = r
    return out


def _noise_scale(run: RunStore, metric: str, mode: str) -> float | None:
    """Average within-condition std for a metric, across repeated observations."""
    traces = run.read_traces()
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in traces:
        c = r["condition"]
        if c["mode"] != mode:
            continue
        gkey = (r["task_id"], c.get("seed"))
        groups[gkey].append(_METRICS[metric](r))
    stds = [float(np.std(v)) for v in groups.values() if len(v) > 1]
    if not stds:
        return None
    return float(np.mean(stds))


def compare_configs(
    run_a: RunStore,
    run_b: RunStore,
    *,
    mode: str = "greedy",
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    traces_a = [r for r in run_a.read_traces() if r["condition"]["mode"] == mode]
    traces_b = [r for r in run_b.read_traces() if r["condition"]["mode"] == mode]
    idx_a = _index(traces_a)
    idx_b = _index(traces_b)
    keys = sorted(set(idx_a) & set(idx_b))

    result: dict[str, Any] = {
        "mode": mode,
        "config_a": run_a.config_id,
        "config_b": run_b.config_id,
        "description_a": run_a.manifest.get("deployment_description"),
        "description_b": run_b.manifest.get("deployment_description"),
        "n_matched": len(keys),
        "metrics": {},
    }

    for metric, extract in _METRICS.items():
        a_vals = [extract(idx_a[k]) for k in keys]
        b_vals = [extract(idx_b[k]) for k in keys]
        delta = paired_bootstrap_delta(a_vals, b_vals, n_resamples=n_resamples, seed=seed)
        noise = _noise_scale(run_a, metric, mode)
        result["metrics"][metric] = {
            "delta": delta,
            "noise_scale": noise,
            "classification": classify_effect(delta, noise if noise is not None else 0.0),
            "summary_a": _brief(a_vals),
            "summary_b": _brief(b_vals),
        }

    # answer-transition comparison from probes, if present
    probes_a = run_a.read_probes()
    probes_b = run_b.read_probes()
    if probes_a and probes_b:
        from collections import Counter

        ca = Counter(_trajectory_state(run_a, p) for p in probes_a)
        cb = Counter(_trajectory_state(run_b, p) for p in probes_b)
        result["trajectory_states"] = {"a": dict(ca), "b": dict(cb)}

    return result


def _brief(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "mean": None}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def _trajectory_state(run: RunStore, probe: dict[str, Any]) -> str:
    # Recompute per-task state from prefix correctness for a compact comparison.
    ps = sorted(
        [p for p in run.read_probes() if p["task_id"] == probe["task_id"]],
        key=lambda p: p["prefix_len"],
    )
    from ..reasoning.transitions import classify_transitions

    traces = run.read_traces()
    final_correct = next(
        (
            r["correct"]
            for r in traces
            if r["task_id"] == probe["task_id"] and r["condition"]["mode"] == "greedy"
        ),
        False,
    )
    return classify_transitions(final_correct, [bool(p["correct"]) for p in ps]).primary
