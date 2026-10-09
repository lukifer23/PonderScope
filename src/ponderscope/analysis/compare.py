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

from ..evidence.run import RunStore, verify_run
from .analyze import (
    _is_generation_termination,
    _is_reasoning_closure,
    _time_to_closure,
    _time_to_termination,
    natural_final_status_of,
)
from .stats import (
    classify_effect,
    combine_noise_scales,
    group_values_by_task,
    paired_cluster_bootstrap_delta,
)
from .survival import rmst_delta_clustered


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

# Model fields whose change is intrinsic to a same-source precision/quantization
# contrast: the executable representation changes while the upstream source
# artifact (repo, revision, source weight/tokenizer/template hashes) does not.
_WEIGHT_REPRESENTATION_FIELDS = frozenset(
    {
        "representation",
        "precision",
        "quantization",
        "quantization_bits",
        "quantization_group_size",
        "variant_weight_files",
        "conversion",
    }
)
# Fields that identify the upstream source artifact itself.
_SOURCE_IDENTITY_FIELDS = frozenset(
    {
        "repo_id",
        "revision",
        "weight_files",
        "tokenizer_files",
        "chat_template_sha256",
    }
)


def classify_contrast(
    diff: dict[str, Any],
    *,
    model_changed: list[str],
    runtime_changed: list[str],
    decoding_changed: list[str],
) -> tuple[str, list[str]]:
    """Classify the requested contrast by hypothesis, not by raw field count.

    Returns ``(label, changed_fields)``. A same-source BF16-to-Q4 contrast is
    ``weight_representation`` even though several model metadata fields change
    together (precision, quantization, bits, group size, variant weight hashes,
    conversion), because those all describe one controlled manipulation. A
    changed source revision/tokenizer/template is a *different source model*, not
    a weight-representation contrast, and is labelled accordingly.
    """
    model_fields = {k.split(".", 1)[1] for k in model_changed}
    source_changed = sorted(model_fields & _SOURCE_IDENTITY_FIELDS)
    if source_changed:
        return "different_source_model", source_changed
    if model_fields and not runtime_changed and model_fields <= _WEIGHT_REPRESENTATION_FIELDS:
        return "weight_representation", sorted(model_fields)
    if model_fields and runtime_changed:
        return "confounded_model_runtime", sorted(model_fields)
    if model_fields:
        return "model_other", sorted(model_fields)
    if runtime_changed:
        return "runtime_hardware", sorted(k.split(".", 1)[1] for k in runtime_changed)
    if decoding_changed:
        return "decoding", sorted(k.split(".", 1)[1] for k in decoding_changed)
    return "same_deployment", []


def _key(r: dict[str, Any]) -> tuple:
    """Canonical matched-trial key.

    Includes the rendered presentation (so two prompt policies over one
    structural task cannot be paired as if identical), the structural task, the
    **decoding condition id** (so two sampled policies cannot collide), the
    decoding mode, the sampling seed, and the technical repetition. Deployment
    identity is deliberately excluded: pairing a BF16 trial with its Q4
    counterpart requires that they share everything except the weight variant.
    """
    c = r["condition"]
    return (
        str(r.get("presentation_id") or r["task_id"]),
        str(r["task_id"]),
        str(r["condition_id"]),
        str(c["mode"]),
        c.get("seed"),
        c.get("repeat"),
    )


def _trial_projection(record: dict[str, Any]) -> tuple:
    """Trial identity without the structural task id (task is 1:1 with pres)."""
    c = record["condition"]
    return (
        str(record.get("presentation_id") or record["task_id"]),
        str(record["condition_id"]),
        str(c["mode"]),
        c.get("seed"),
        c.get("repeat"),
    )


def _expected_trial_keys(run: RunStore, mode: str) -> tuple[set[tuple] | None, dict[str, Any]]:
    """Declared trial-identity set for a run/mode, from the frozen declaration.

    Derived from the frozen task presentations, the declared decoding
    conditions, the declared seed list, and the repetition counts. Returns
    ``(keys, meta)``; ``keys`` is ``None`` when the declaration is incomplete, so
    callers fail closed rather than assuming completeness.
    """
    pack = run.manifest.get("task_pack", {})
    presentations = pack.get("presentation_ids") or []
    spec = run.manifest.get("spec", {})
    conditions = [
        c for c in run.manifest.get("conditions", []) if c.get("decoding", {}).get("mode") == mode
    ]
    if not presentations:
        return None, {"reason": "no declared presentation population"}
    if not conditions:
        return None, {"reason": f"no declared {mode!r} condition"}
    if mode == "sampled":
        seeds = spec.get("sampled_seeds")
        repeats = spec.get("sampled_repeats_per_seed")
    elif mode == "greedy":
        seeds = [None]
        repeats = spec.get("greedy_repeats")
    else:
        return None, {"reason": f"unknown mode {mode!r}"}
    if seeds is None or repeats is None:
        return None, {"reason": "declared seeds/repeats missing from spec"}
    keys: set[tuple] = set()
    for condition in conditions:
        cid = str(condition["condition_id"])
        for presentation in presentations:
            for seed in seeds:
                for repeat in range(int(repeats)):
                    keys.add((str(presentation), cid, mode, seed, repeat))
    return keys, {
        "n_conditions": len(conditions),
        "n_presentations": len(presentations),
        "n_seeds": len(seeds),
        "repeats_per_seed": int(repeats),
        "expected_total": len(keys),
    }


def _ambiguous_draw_count(traces: list[dict[str, Any]]) -> int:
    """Same-seed technical-repeat groups whose executions are not identical."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in traces:
        c = r["condition"]
        groups[
            (str(r.get("presentation_id") or r["task_id"]), str(r["condition_id"]), c.get("seed"))
        ].append(r)
    ambiguous = 0
    for executions in groups.values():
        if len(executions) < 2:
            continue
        base = executions[0]["trace"]["token_ids"]
        if not all(e["trace"]["token_ids"] == base for e in executions[1:]):
            ambiguous += 1
    return ambiguous


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

    contrast, contrast_fields = classify_contrast(
        diff,
        model_changed=model_changed,
        runtime_changed=runtime_changed,
        decoding_changed=decoding_changed,
    )

    # Same repo+revision but different source weight/tokenizer/template hashes is
    # inconsistent provenance: the source artifact must be immutable, so this is
    # never a clean same-source contrast.
    same_repo = "model.repo_id" not in diff["changed"]
    same_revision = "model.revision" not in diff["changed"]
    inconsistent_source = sorted(set(contrast_fields) & _SOURCE_IDENTITY_FIELDS)
    if same_repo and same_revision and inconsistent_source:
        reasons.append(
            "source-artifact contents differ at the same repo/revision "
            f"(inconsistent provenance): {inconsistent_source}"
        )

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
        "contrast": contrast,
        "contrast_changed_fields": contrast_fields,
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


def _draw_key(r: dict[str, Any]) -> tuple:
    """Canonical stochastic-draw key: presentation + task + condition + seed.

    Technical repetition is deliberately excluded (it is an execution
    dimension), and deployment identity is excluded (so BF16 and Q4 pair).
    """
    c = r["condition"]
    return (
        str(r.get("presentation_id") or r["task_id"]),
        str(r["task_id"]),
        str(r["condition_id"]),
        c.get("seed"),
    )


def _collapsed_draws(
    run: RunStore,
    matched_keys: set[tuple],
    *,
    traces: list[dict[str, Any]] | None = None,
) -> dict[tuple, dict[str, Any]]:
    """Collapse matched executions to one entry per stochastic draw.

    Grouping is by the canonical draw key (including ``condition_id``), so two
    different decoding conditions with the same presentation and seed never
    collapse together. Same-seed technical repeats of one condition collapse to
    one draw iff every execution is token-identical; divergent (ambiguous) draws
    are flagged with ``record=None``.
    """
    source = run.read_traces() if traces is None else traces
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in source:
        if _key(r) in matched_keys:
            groups[_draw_key(r)].append(r)
    draws: dict[tuple, dict[str, Any]] = {}
    for key, executions in groups.items():
        executions = sorted(executions, key=lambda r: r["condition"].get("repeat", 0))
        base = executions[0]["trace"]["token_ids"]
        identical = all(e["trace"]["token_ids"] == base for e in executions[1:])
        draws[key] = {
            "task_id": str(key[1]),
            "record": executions[0] if identical else None,
            "ambiguous": not identical,
            "n_executions": len(executions),
        }
    return draws


def _matched_draw_records(
    run: RunStore,
    matched_keys: set[tuple],
    *,
    traces: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Draw-collapsed records grouped by task (ambiguous draws excluded)."""
    draws = _collapsed_draws(run, matched_keys, traces=traces)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    n_ambiguous = 0
    n_executions = 0
    for entry in draws.values():
        n_executions += entry["n_executions"]
        if entry["ambiguous"]:
            n_ambiguous += 1
            continue
        by_task[entry["task_id"]].append(entry["record"])
    return dict(by_task), {
        "n_draws": len(draws),
        "n_executions": n_executions,
        "n_ambiguous_draws": n_ambiguous,
    }


def _paired_draw_population(
    run_a: RunStore,
    run_b: RunStore,
    matched_keys: set[tuple],
    *,
    traces_a: list[dict[str, Any]] | None = None,
    traces_b: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Finalize the paired stochastic-draw population for a comparison.

    Collapses each arm independently, then restricts **both** arms to the
    intersection of their usable (non-ambiguous) draw keys, so a draw that is
    ambiguous in only one arm is excluded from both. Every paired metric uses
    this same finalized population.
    """
    a = _collapsed_draws(run_a, matched_keys, traces=traces_a)
    b = _collapsed_draws(run_b, matched_keys, traces=traces_b)
    usable_a = {k for k, v in a.items() if not v["ambiguous"]}
    usable_b = {k for k, v in b.items() if not v["ambiguous"]}
    paired = usable_a & usable_b
    a_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    b_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key in paired:
        a_by_task[a[key]["task_id"]].append(a[key]["record"])
        b_by_task[b[key]["task_id"]].append(b[key]["record"])
    population = {
        "unit": "stochastic draw (presentation x task x condition x seed)",
        "n_draws_a": len(a),
        "n_draws_b": len(b),
        "n_usable_draws_a": len(usable_a),
        "n_usable_draws_b": len(usable_b),
        "n_paired_draws": len(paired),
        "n_ambiguous_a": sum(1 for v in a.values() if v["ambiguous"]),
        "n_ambiguous_b": sum(1 for v in b.values() if v["ambiguous"]),
        "n_unpaired_a": len(usable_a - paired),
        "n_unpaired_b": len(usable_b - paired),
        "unpaired_examples_a": [list(k) for k in sorted(usable_a - paired)[:3]],
        "unpaired_examples_b": [list(k) for k in sorted(usable_b - paired)[:3]],
        "note": (
            "Every paired metric uses only the intersection of usable draws from "
            "both arms; ambiguous or unmatched draws are excluded from both and "
            "remain visible in the diagnostics."
        ),
    }
    return dict(a_by_task), dict(b_by_task), population


def _rmst_section(
    run_a: RunStore,
    run_b: RunStore,
    keys: list[tuple],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Task-clustered bootstrap of the censor-aware RMST difference A - B.

    Reported separately for the primary endpoint (native reasoning closure,
    timed at the think-end token) and the secondary endpoint (EOS generation
    termination, timed at the terminal token). Both arms are restricted to the
    matched stochastic draws, so an unbalanced seed/repeat population cannot
    silently enter the estimate. Positive favors A.
    """
    matched = set(keys)
    tau = float(run_a.manifest.get("spec", {}).get("max_tokens") or 0)
    out: dict[str, Any] = {"tau": tau or None, "endpoints": {}}
    if not tau or not matched:
        return out

    endpoint_specs = (
        ("reasoning_closure", _time_to_closure, _is_reasoning_closure),
        ("generation_termination", _time_to_termination, _is_generation_termination),
    )
    a_by_task, b_by_task, population = _paired_draw_population(run_a, run_b, matched)
    out["paired_population"] = population
    for label, time_key, event_key in endpoint_specs:
        a_obs = {
            task: [(float(time_key(r)), bool(event_key(r))) for r in records]
            for task, records in a_by_task.items()
        }
        b_obs = {
            task: [(float(time_key(r)), bool(event_key(r))) for r in records]
            for task, records in b_by_task.items()
        }
        delta = rmst_delta_clustered(a_obs, b_obs, tau=tau, n_resamples=n_resamples, seed=seed)
        out["endpoints"][label] = delta
    return out


def _trial_population(
    run_a: RunStore,
    run_b: RunStore,
    traces_a: list[dict[str, Any]],
    traces_b: list[dict[str, Any]],
    keys: list[tuple],
    mode: str,
) -> dict[str, Any]:
    """Account for the matched-trial population against the **declared** design.

    Completeness requires that each arm's observed trial-identity set equals the
    set declared by its own frozen spec (presentations x conditions x seeds x
    repetitions), that neither arm has duplicate logical identities or ambiguous
    same-seed repeats, that both arms declare and observe the *same* population,
    and that the matched intersection equals it. A total count that happens to
    match is not sufficient: a missing seed replaced by an unexpected seed is
    detected because the declared identity set is compared exactly.
    """
    expected_a, meta_a = _expected_trial_keys(run_a, mode)
    expected_b, meta_b = _expected_trial_keys(run_b, mode)
    idx_a = _index(traces_a)
    idx_b = _index(traces_b)
    proj_a = {_trial_projection(r) for r in traces_a}
    proj_b = {_trial_projection(r) for r in traces_b}
    dup_a = len(traces_a) - len(idx_a)
    dup_b = len(traces_b) - len(idx_b)
    amb_a = _ambiguous_draw_count(traces_a)
    amb_b = _ambiguous_draw_count(traces_b)
    matched = len(keys)

    declared_a = {str(p) for p in run_a.manifest.get("task_pack", {}).get("presentation_ids", [])}
    declared_b = {str(p) for p in run_b.manifest.get("task_pack", {}).get("presentation_ids", [])}
    observed_pres_a = {str(r.get("presentation_id") or r["task_id"]) for r in traces_a}
    observed_pres_b = {str(r.get("presentation_id") or r["task_id"]) for r in traces_b}
    presentation_match_a = bool(declared_a) and observed_pres_a == declared_a
    presentation_match_b = bool(declared_b) and observed_pres_b == declared_b

    missing_a = sorted((expected_a or set()) - proj_a)
    unexpected_a = sorted(proj_a - (expected_a or set()))
    missing_b = sorted((expected_b or set()) - proj_b)
    unexpected_b = sorted(proj_b - (expected_b or set()))

    same_expected = expected_a is not None and expected_a == expected_b
    arm_a_ok = bool(
        expected_a is not None
        and proj_a == expected_a
        and dup_a == 0
        and amb_a == 0
        and presentation_match_a
    )
    arm_b_ok = bool(
        expected_b is not None
        and proj_b == expected_b
        and dup_b == 0
        and amb_b == 0
        and presentation_match_b
    )
    complete = bool(
        same_expected
        and arm_a_ok
        and arm_b_ok
        and expected_a is not None
        and matched == len(expected_a)
    )

    reasons: list[str] = []
    if expected_a is None or expected_b is None:
        reasons.append(
            f"declared population unavailable (A: {meta_a.get('reason')}, B: {meta_b.get('reason')})"
        )
    if expected_a is not None and expected_b is not None and not same_expected:
        reasons.append("the two arms declare different trial populations")
    for side, missing, unexpected, dup, amb, pres_ok, ok in (
        ("A", missing_a, unexpected_a, dup_a, amb_a, presentation_match_a, arm_a_ok),
        ("B", missing_b, unexpected_b, dup_b, amb_b, presentation_match_b, arm_b_ok),
    ):
        if not ok:
            reasons.append(
                f"arm {side} incomplete "
                f"(missing={len(missing)} unexpected={len(unexpected)} "
                f"duplicates={dup} ambiguous_draws={amb} presentations_match={pres_ok})"
            )
    if expected_a is not None and matched != len(expected_a):
        reasons.append(f"matched intersection {matched} != declared population {len(expected_a)}")

    return {
        "mode": mode,
        "expected_known": expected_a is not None and expected_b is not None,
        "expected_trials_a": len(expected_a) if expected_a is not None else None,
        "expected_trials_b": len(expected_b) if expected_b is not None else None,
        "expected_meta_a": meta_a,
        "expected_meta_b": meta_b,
        "observed_executions_a": len(traces_a),
        "observed_executions_b": len(traces_b),
        "unique_trial_keys_a": len(idx_a),
        "unique_trial_keys_b": len(idx_b),
        "matched_pairs": matched,
        "unmatched_a": len(idx_a) - matched,
        "unmatched_b": len(idx_b) - matched,
        "missing_trials_a": len(missing_a),
        "missing_trials_b": len(missing_b),
        "unexpected_trials_a": len(unexpected_a),
        "unexpected_trials_b": len(unexpected_b),
        "missing_examples_a": [list(k) for k in missing_a[:3]],
        "missing_examples_b": [list(k) for k in missing_b[:3]],
        "unexpected_examples_a": [list(k) for k in unexpected_a[:3]],
        "unexpected_examples_b": [list(k) for k in unexpected_b[:3]],
        "duplicate_keys_a": dup_a,
        "duplicate_keys_b": dup_b,
        "ambiguous_draws_a": amb_a,
        "ambiguous_draws_b": amb_b,
        "presentation_population_match_a": presentation_match_a,
        "presentation_population_match_b": presentation_match_b,
        "complete": complete,
        "incomplete_reasons": reasons,
        "note": (
            "complete requires each arm's observed trial identities to equal its "
            "declared population (presentation x condition_id x seed x repeat), no "
            "duplicate identities, no ambiguous same-seed repeats, matching "
            "presentation populations, and a matched intersection equal to the "
            "declared population."
        ),
    }


def cross_study_summary(
    comparisons: list[dict[str, Any]],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Summarize several comparisons side by side **without pooling**.

    Distinguishes the original calibration contrast from an independent
    replication. Pooled estimates are deliberately withheld: the two populations
    are distinct and pooling would require an explicit, justified method.
    """
    studies: list[dict[str, Any]] = []
    for i, c in enumerate(comparisons):
        rmst = c.get("rmst", {}).get("endpoints", {}).get("reasoning_closure", {})
        closure = c.get("metrics", {}).get("reasoning_closure_rate", {}).get("delta", {})
        studies.append(
            {
                "label": labels[i] if labels and i < len(labels) else f"study-{i}",
                "deployment_a": c.get("deployment_a"),
                "deployment_b": c.get("deployment_b"),
                "contrast": c.get("contrast"),
                "n_matched_pairs": c.get("n_matched_trials"),
                "population_complete": c.get("trial_population", {}).get("complete"),
                "publication_grade": c.get("publication_grade"),
                "closure_rate_delta": closure.get("mean"),
                "closure_rate_ci": [closure.get("lo"), closure.get("hi")],
                "rmst_reasoning_delta": rmst.get("mean"),
                "rmst_reasoning_ci": [rmst.get("lo"), rmst.get("hi")],
            }
        )
    return {
        "studies": studies,
        "pooled": None,
        "pooling_note": (
            "Calibration and replication populations are distinct; pooled estimates "
            "are not produced automatically and would require an explicit, "
            "justified method."
        ),
    }


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
    require_complete: bool = False,
    verify: bool = True,
    allow_unverified: bool = False,
) -> dict[str, Any]:
    traces_a = [r for r in run_a.read_traces() if r["condition"]["mode"] == mode]
    traces_b = [r for r in run_b.read_traces() if r["condition"]["mode"] == mode]
    idx_a = _index(traces_a)
    idx_b = _index(traces_b)
    keys = sorted(set(idx_a) & set(idx_b))

    verification: dict[str, Any] = {
        "checked": bool(verify),
        "a": verify_run(run_a.path) if verify else None,
        "b": verify_run(run_b.path) if verify else None,
    }

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
    population = _trial_population(run_a, run_b, traces_a, traces_b, keys, mode)

    verification_failed = bool(
        verify
        and not (
            verification["a"]
            and verification["a"].get("pass")
            and verification["b"]
            and verification["b"].get("pass")
        )
    )
    refusal_reasons: list[str] = list(validity["reasons"])
    overrides_applied: list[str] = []
    if not verify:
        # Skipping verification is an explicit override and can never be
        # publication-grade, even if everything else looks clean.
        overrides_applied.append("verification_skipped")
    elif verification_failed:
        if allow_unverified:
            overrides_applied.append("unverified_evidence")
        else:
            refusal_reasons.append(
                "evidence seal does not verify; pass allow_unverified=True only for "
                "labelled exploratory/forensic analysis"
            )
    if require_complete and not population["complete"]:
        if allow_confounded:
            overrides_applied.append("incomplete_population")
        else:
            refusal_reasons.append(
                "trial population is not complete "
                f"(matched={population['matched_pairs']}, "
                f"unmatched A={population['unmatched_a']} B={population['unmatched_b']}, "
                f"duplicates A={population['duplicate_keys_a']} B={population['duplicate_keys_b']})"
            )
    if not validity["clean"] and allow_confounded:
        overrides_applied.append("confounded")

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
        "n_matched_tasks": len({k[1] for k in keys}),
        "trial_population": population,
        "verification": verification,
        "configuration_diff": diff,
        "validity": validity,
        "contrast": validity["contrast"],
        "refusal_reasons": refusal_reasons,
        "overrides_applied": overrides_applied,
        "exploratory": bool(overrides_applied),
        "metrics": {},
    }
    verified_all = bool(
        verify
        and verification["a"]
        and verification["a"].get("pass")
        and verification["b"]
        and verification["b"].get("pass")
    )
    result["publication_grade"] = bool(
        verified_all
        and not refusal_reasons
        and validity["clean"]
        and population["complete"]
        and not overrides_applied
    )
    if refusal_reasons and not allow_confounded:
        result["refused"] = True
        return result
    result["refused"] = False

    # All paired metrics use the same finalized paired-draw population: each arm
    # is collapsed to one primary record per stochastic draw (condition-aware),
    # then restricted to the intersection of usable draws from both arms.
    a_by_task, b_by_task, analysis_population = _paired_draw_population(run_a, run_b, set(keys))
    a_flat = [r for records in a_by_task.values() for r in records]
    b_flat = [r for records in b_by_task.values() for r in records]
    analysis_population["n_draws_used_a"] = len(a_flat)
    analysis_population["n_draws_used_b"] = len(b_flat)
    result["analysis_population"] = analysis_population

    for metric, extract in _METRICS.items():
        a_vals = group_values_by_task(a_flat, "task_id", extract)
        b_vals = group_values_by_task(b_flat, "task_id", extract)
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
            "noise_scale_note": (
                None
                if noise is not None
                else "no within-(task,seed) technical repeats; within-deployment noise "
                "is not estimable — uncertainty is the task-clustered bootstrap only"
            ),
            "summary_a": _brief([x for values in a_vals.values() for x in values]),
            "summary_b": _brief([x for values in b_vals.values() for x in values]),
            "estimable": bool(a_vals and b_vals),
            "analysis_population": analysis_population,
        }

    result["rmst"] = _rmst_section(run_a, run_b, keys, n_resamples=n_resamples, seed=seed)

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
