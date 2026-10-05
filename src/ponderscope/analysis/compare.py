"""Compare two deployment configurations using matched, task-clustered evidence.

Before any metric is compared, a canonical configuration-difference report is
produced and the requested contrast is checked for confounds. Comparisons whose
provenance cannot establish the intended contrast are refused unless explicitly
requested as exploratory.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

from ..evidence.run import RunStore
from .stats import (
    classify_effect,
    combine_noise_scales,
    group_values_by_task,
    paired_cluster_bootstrap_delta,
)

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

_DEPLOYMENT_FIELDS = (
    "repo_id",
    "revision",
    "precision",
    "quantization",
    "quantization_bits",
    "quantization_group_size",
    "weight_files",
    "tokenizer_files",
    "chat_template_sha256",
)
_RUNTIME_FIELDS = (
    "runtime",
    "runtime_version",
    "backend",
    "hardware",
    "os",
    "python_version",
    "device",
)

_MISSING = object()


def _key(r: dict[str, Any]) -> tuple:
    c = r["condition"]
    return (r["task_id"], c["mode"], c.get("seed"), c.get("repeat"))


def _index(traces: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    out: dict[tuple, dict[str, Any]] = {}
    for r in traces:
        out[_key(r)] = r
    return out


def _model(run: RunStore) -> dict[str, Any]:
    return run.manifest.get("artifact", run.manifest.get("deployment", {}).get("model", {}))


def _runtime(run: RunStore) -> dict[str, Any]:
    return run.manifest.get("runtime", run.manifest.get("deployment", {}).get("runtime", {}))


def config_diff(run_a: RunStore, run_b: RunStore) -> dict[str, Any]:
    a_model, b_model = _model(run_a), _model(run_b)
    a_rt, b_rt = _runtime(run_a), _runtime(run_b)
    identical: dict[str, Any] = {}
    changed: dict[str, Any] = {}
    missing: list[str] = []

    def check(section: str, fields: tuple[str, ...], a: dict, b: dict) -> None:
        for f in fields:
            av, bv = a.get(f, _MISSING), b.get(f, _MISSING)
            key = f"{section}.{f}"
            if av is _MISSING or bv is _MISSING:
                missing.append(key)
            elif av == bv:
                identical[key] = av
            else:
                changed[key] = {"a": av, "b": bv}

    check("model", _DEPLOYMENT_FIELDS, a_model, b_model)
    check("runtime", _RUNTIME_FIELDS, a_rt, b_rt)

    conds_a = sorted(
        (c["decoding"] for c in run_a.manifest.get("conditions", [])),
        key=lambda d: json.dumps(d, sort_keys=True),
    )
    conds_b = sorted(
        (c["decoding"] for c in run_b.manifest.get("conditions", [])),
        key=lambda d: json.dumps(d, sort_keys=True),
    )
    if conds_a == conds_b:
        identical["decoding.conditions"] = conds_a
    else:
        changed["decoding.conditions"] = {"a": conds_a, "b": conds_b}

    return {"identical": identical, "changed": changed, "missing": missing}


def _deployment_change_count(diff: dict[str, Any]) -> int:
    """Count distinct deployment dimensions changed (decoding counted separately)."""
    counts = {
        "model": sum(1 for k in diff["changed"] if k.startswith("model.")),
        "runtime": sum(1 for k in diff["changed"] if k.startswith("runtime.")),
        "decoding": sum(1 for k in diff["changed"] if k.startswith("decoding.")),
    }
    return counts["model"] + counts["runtime"] + counts["decoding"]


def assess_comparison(
    diff: dict[str, Any],
    *,
    task_ids_a: set[str],
    task_ids_b: set[str],
    mode: str,
    decoding_intentional: bool = False,
) -> dict[str, Any]:
    reasons: list[str] = []
    if diff["missing"]:
        reasons.append(f"provenance fields missing/unknown: {sorted(diff['missing'])}")
    if task_ids_a != task_ids_b:
        reasons.append(
            f"task populations differ (A={len(task_ids_a)}, B={len(task_ids_b)}, "
            f"onlyA={len(task_ids_a - task_ids_b)}, onlyB={len(task_ids_b - task_ids_a)})"
        )
    model_changed = [k for k in diff["changed"] if k.startswith("model.")]
    runtime_changed = [k for k in diff["changed"] if k.startswith("runtime.")]
    decoding_changed = [k for k in diff["changed"] if k.startswith("decoding.")]
    if model_changed and runtime_changed:
        reasons.append("both model-artifact and runtime/hardware changed simultaneously")
    if len(runtime_changed) > 1:
        reasons.append(f"multiple runtime variables changed: {sorted(runtime_changed)}")
    if decoding_changed and not decoding_intentional:
        reasons.append(
            f"decoding policy differs but is not declared part of the hypothesis: {decoding_changed}"
        )
    return {
        "clean": not reasons,
        "reasons": reasons,
        "model_changed": model_changed,
        "runtime_changed": runtime_changed,
        "decoding_changed": decoding_changed,
        "n_dimensions_changed": _deployment_change_count(diff),
    }


def _noise_scale(run: RunStore, metric: str, mode: str) -> float | None:
    """Mean within-deployment, within-(task,seed) std for a metric."""
    traces = run.read_traces()
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in traces:
        c = r["condition"]
        if c["mode"] != mode:
            continue
        groups[(r["task_id"], c.get("seed"))].append(_METRICS[metric](r))
    stds = [float(np.std(v, ddof=1)) for v in groups.values() if len(v) > 1]
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
    decoding_intentional: bool = False,
    allow_confounded: bool = False,
) -> dict[str, Any]:
    traces_a = [r for r in run_a.read_traces() if r["condition"]["mode"] == mode]
    traces_b = [r for r in run_b.read_traces() if r["condition"]["mode"] == mode]
    idx_a = _index(traces_a)
    idx_b = _index(traces_b)
    keys = sorted(set(idx_a) & set(idx_b))

    diff = config_diff(run_a, run_b)
    task_ids_a = {r["task_id"] for r in traces_a}
    task_ids_b = {r["task_id"] for r in traces_b}
    validity = assess_comparison(
        diff,
        task_ids_a=task_ids_a,
        task_ids_b=task_ids_b,
        mode=mode,
        decoding_intentional=decoding_intentional,
    )

    result: dict[str, Any] = {
        "mode": mode,
        "artifact_a": run_a.artifact_id,
        "artifact_b": run_b.artifact_id,
        "deployment_a": run_a.deployment_id,
        "deployment_b": run_b.deployment_id,
        "description_a": run_a.manifest.get("deployment_description"),
        "description_b": run_b.manifest.get("deployment_description"),
        "n_matched_trials": len(keys),
        "n_matched_tasks": len({k[0] for k in keys}),
        "configuration_diff": diff,
        "validity": validity,
        "exploratory": bool(not validity["clean"] and allow_confounded),
        "metrics": {},
    }
    if not validity["clean"] and not allow_confounded:
        result["refused"] = True
        return result
    result["refused"] = False

    for metric, extract in _METRICS.items():
        pairs = [(idx_a[k], idx_b[k]) for k in keys]
        a_vals = group_values_by_task([pa for pa, _ in pairs], "task_id", extract)
        b_vals = group_values_by_task([pb for _, pb in pairs], "task_id", extract)
        delta = paired_cluster_bootstrap_delta(a_vals, b_vals, n_resamples=n_resamples, seed=seed)
        noise_a = _noise_scale(run_a, metric, mode)
        noise_b = _noise_scale(run_b, metric, mode)
        noise = combine_noise_scales(noise_a, noise_b)
        result["metrics"][metric] = {
            "delta": delta,
            "noise_scale_a": noise_a,
            "noise_scale_b": noise_b,
            "noise_scale": noise,
            "classification": classify_effect(delta, noise),
            "summary_a": _brief([extract(pa) for pa, _ in pairs]),
            "summary_b": _brief([extract(pb) for _, pb in pairs]),
        }

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
    ps = sorted(
        [p for p in run.read_probes() if p["task_id"] == probe["task_id"]],
        key=lambda p: p.get("reasoning_prefix_tokens", p.get("prefix_len", 0)),
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
    lengths = [p.get("reasoning_prefix_tokens", p.get("prefix_len", 0)) for p in ps]
    return classify_transitions(
        final_correct, [bool(p["correct"]) for p in ps], prefix_token_lengths=lengths
    ).primary
