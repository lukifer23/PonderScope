"""Compare two deployment configurations using matched, task-clustered evidence.

Before any metric is compared, a canonical configuration-difference report is
produced and the requested contrast is checked for confounds. Comparisons whose
provenance cannot establish the intended contrast are refused unless explicitly
requested as exploratory.

The default deployment-comparison contract requires identical structural task
population, identical presentation/stimulus IDs, identical prompt policy,
identical decoding condition, identical task-pack/generator version, a
compatible observation horizon, and a compatible capture lane. Any deliberate
change requires its explicit ``*_intentional`` flag; otherwise the contrast is
refused so that e.g. pp-v1 vs pp-v2 cannot masquerade as a clean deployment
contrast.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

from ..evidence.run import RunStore
from .analyze import natural_final_status_of
from .stats import (
    classify_effect,
    combine_noise_scales,
    group_values_by_task,
    paired_cluster_bootstrap_delta,
)


def _conditional_accuracy(r: dict[str, Any]) -> float | None:
    """Accuracy among completed, observed-answer trials; ``None`` if not estimable."""
    if not r["termination"]["terminated_by_eos"]:
        return None
    if r.get("answer_normalized") is None:
        return None
    return 1.0 if r["correct"] else 0.0


_METRICS: dict[str, Callable[[dict[str, Any]], float | None]] = {
    "success_at_budget": lambda r: 1.0 if r["correct"] else 0.0,
    "reasoning_closure_rate": lambda r: 1.0 if r["termination"]["think_end_reached"] else 0.0,
    "completion_rate": lambda r: 1.0 if r["termination"]["terminated_by_eos"] else 0.0,
    "censored_rate": lambda r: 1.0 if natural_final_status_of(r) == "censored" else 0.0,
    "error_rate": lambda r: 1.0 if natural_final_status_of(r) == "error" else 0.0,
    "answer_observation_rate": lambda r: 1.0 if r.get("answer_normalized") is not None else 0.0,
    "conditional_accuracy_given_completed": _conditional_accuracy,
    "reasoning_tokens": lambda r: float(r["reasoning_tokens"]),
    "total_tokens": lambda r: float(r["total_tokens"]),
    "wall_ms": lambda r: float(r["trace"]["wall_ms"]),
    "repeated_ngram_fraction_4": lambda r: float(r["metrics"]["repeated_ngram_fraction_4"]),
    "unique_token_ratio": lambda r: float(r["metrics"]["unique_token_ratio"]),
    "longest_run_length": lambda r: (
        float(r["loop"]["longest_run_length"]) if r.get("loop") else None
    ),
    "terminated_by_eos": lambda r: 1.0 if r["termination"]["terminated_by_eos"] else 0.0,
    "missing_answer": lambda r: 1.0 if r["termination"]["missing_answer"] else 0.0,
}

_DEPLOYMENT_FIELDS = (
    "repo_id",
    "revision",
    "representation",
    "precision",
    "quantization",
    "quantization_bits",
    "quantization_group_size",
    "weight_files",
    "tokenizer_files",
    "chat_template_sha256",
    "variant_weight_files",
    "conversion",
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
_SPEC_FIELDS = (
    "prompt_policy",
    "max_tokens",
    "capture_level",
    "task_pack",
    "split",
    "task_seed",
    "n_per_family",
    "model_policy",
)
_PACK_FIELDS = (
    "pack_version",
    "generator_version",
    "task_ids_sha256",
    "presentation_ids_sha256",
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
    """Flatten source-artifact and weight-variant provenance for field checks."""
    variant = run.manifest.get(
        "weight_variant",
        run.manifest.get("artifact", run.manifest.get("deployment", {}).get("model", {})),
    )
    source = run.manifest.get("source_artifact") or variant.get("source", {})
    return {**source, **variant}


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
    check("spec", _SPEC_FIELDS, run_a.manifest.get("spec", {}), run_b.manifest.get("spec", {}))
    check(
        "task_pack",
        _PACK_FIELDS,
        run_a.manifest.get("task_pack", {}),
        run_b.manifest.get("task_pack", {}),
    )

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
    structural_task_ids_a: set[str],
    structural_task_ids_b: set[str],
    presentation_ids_a: set[str],
    presentation_ids_b: set[str],
    mode: str,
    decoding_intentional: bool = False,
    prompt_policy_intentional: bool = False,
    horizon_intentional: bool = False,
    capture_intentional: bool = False,
) -> dict[str, Any]:
    reasons: list[str] = []
    if diff["missing"]:
        reasons.append(f"provenance fields missing/unknown: {sorted(diff['missing'])}")
    if structural_task_ids_a != structural_task_ids_b:
        reasons.append(
            f"structural task populations differ (A={len(structural_task_ids_a)}, "
            f"B={len(structural_task_ids_b)}, onlyA={len(structural_task_ids_a - structural_task_ids_b)}, "
            f"onlyB={len(structural_task_ids_b - structural_task_ids_a)})"
        )

    model_changed = [k for k in diff["changed"] if k.startswith("model.")]
    runtime_changed = [k for k in diff["changed"] if k.startswith("runtime.")]
    decoding_changed = [k for k in diff["changed"] if k.startswith("decoding.")]
    spec_changed = {k.split(".", 1)[1] for k in diff["changed"] if k.startswith("spec.")}
    pack_changed = {k.split(".", 1)[1] for k in diff["changed"] if k.startswith("task_pack.")}

    if model_changed and runtime_changed:
        reasons.append("both model-artifact and runtime/hardware changed simultaneously")
    if len(runtime_changed) > 1:
        reasons.append(f"multiple runtime variables changed: {sorted(runtime_changed)}")
    if decoding_changed and not decoding_intentional:
        reasons.append(
            f"decoding policy differs but is not declared part of the hypothesis: {decoding_changed}"
        )
    if {"pack_version", "generator_version"} & pack_changed:
        reasons.append(
            f"task-pack/generator version differs (not comparable): {sorted(pack_changed)}"
        )
    if presentation_ids_a != presentation_ids_b and not prompt_policy_intentional:
        reasons.append(
            "presentation/stimulus populations differ (A="
            f"{len(presentation_ids_a)}, B={len(presentation_ids_b)}); a prompt-policy "
            "change is not a clean deployment contrast"
        )
    if "prompt_policy" in spec_changed and not prompt_policy_intentional:
        reasons.append("prompt policy differs but is not declared intentional")
    if "max_tokens" in spec_changed and not horizon_intentional:
        reasons.append(
            f"observation horizon differs ({diff['changed']['spec.max_tokens']}) but is not "
            "declared intentional"
        )
    if "capture_level" in spec_changed and not capture_intentional:
        reasons.append("capture lane differs but is not declared intentional")

    return {
        "clean": not reasons,
        "reasons": reasons,
        "model_changed": model_changed,
        "runtime_changed": runtime_changed,
        "decoding_changed": decoding_changed,
        "spec_changed": sorted(spec_changed),
        "task_pack_changed": sorted(pack_changed),
        "presentation_mismatch": presentation_ids_a != presentation_ids_b,
        "n_dimensions_changed": _deployment_change_count(diff),
        "overrides": {
            "decoding_intentional": decoding_intentional,
            "prompt_policy_intentional": prompt_policy_intentional,
            "horizon_intentional": horizon_intentional,
            "capture_intentional": capture_intentional,
        },
    }


def _noise_scale(run: RunStore, metric: str, mode: str) -> float | None:
    """Mean within-deployment, within-(task,seed) std for a metric."""
    traces = run.read_traces()
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in traces:
        c = r["condition"]
        if c["mode"] != mode:
            continue
        value = _METRICS[metric](r)
        if value is None:
            continue
        groups[(r["task_id"], c.get("seed"))].append(value)
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
    prompt_policy_intentional: bool = False,
    horizon_intentional: bool = False,
    capture_intentional: bool = False,
    allow_confounded: bool = False,
) -> dict[str, Any]:
    traces_a = [r for r in run_a.read_traces() if r["condition"]["mode"] == mode]
    traces_b = [r for r in run_b.read_traces() if r["condition"]["mode"] == mode]
    idx_a = _index(traces_a)
    idx_b = _index(traces_b)
    keys = sorted(set(idx_a) & set(idx_b))

    diff = config_diff(run_a, run_b)
    structural_a = {r["task_id"] for r in traces_a}
    structural_b = {r["task_id"] for r in traces_b}
    presentation_a = {r.get("presentation_id") or r["task_id"] for r in traces_a}
    presentation_b = {r.get("presentation_id") or r["task_id"] for r in traces_b}
    validity = assess_comparison(
        diff,
        structural_task_ids_a=structural_a,
        structural_task_ids_b=structural_b,
        presentation_ids_a=presentation_a,
        presentation_ids_b=presentation_b,
        mode=mode,
        decoding_intentional=decoding_intentional,
        prompt_policy_intentional=prompt_policy_intentional,
        horizon_intentional=horizon_intentional,
        capture_intentional=capture_intentional,
    )

    result: dict[str, Any] = {
        "mode": mode,
        "artifact_a": run_a.artifact_id,
        "artifact_b": run_b.artifact_id,
        "deployment_a": run_a.deployment_id,
        "deployment_b": run_b.deployment_id,
        "prompt_policy_a": run_a.manifest.get("prompt_policy"),
        "prompt_policy_b": run_b.manifest.get("prompt_policy"),
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
            "summary_a": _brief([x for values in a_vals.values() for x in values]),
            "summary_b": _brief([x for values in b_vals.values() for x in values]),
            "estimable": bool(a_vals and b_vals),
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
    greedy_record = next(
        (
            r
            for r in traces
            if r["task_id"] == probe["task_id"] and r["condition"]["mode"] == "greedy"
        ),
        None,
    )
    natural_status = natural_final_status_of(greedy_record) if greedy_record else "censored"
    lengths = [p.get("reasoning_prefix_tokens", p.get("prefix_len", 0)) for p in ps]
    return classify_transitions(
        [bool(p["correct"]) for p in ps], natural_status, prefix_token_lengths=lengths
    ).prefix_state
