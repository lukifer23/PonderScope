"""Analyze immutable saved evidence. Never touches the model.

Noise floor is measured as three distinct quantities:
1. greedy replay variation (same deterministic condition, repeated);
2. same-seed sampled replay variation (same sampler configuration and seed);
3. across-seed stochastic variation.

These are conceptually different and are never collapsed into one standard
deviation. Statistical unit is the task.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from ..evidence.run import RunStore
from ..reasoning.transitions import classify_transitions
from .stats import (
    cluster_bootstrap_ci,
    group_values_by_task,
    summarize,
    within_cluster_std,
)


def _accuracy_macro(records: list[dict[str, Any]]) -> float | None:
    by_task = group_values_by_task(records, "task_id", lambda r: 1.0 if r["correct"] else 0.0)
    if not by_task:
        return None
    return float(np.mean([np.mean(v) for v in by_task.values()]))


def _termination_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for r in records:
        term = r["termination"]
        if term["terminated_by_eos"]:
            counts["natural_eos"] += 1
        elif term["capped"]:
            counts["capped_length"] += 1
        else:
            counts["error_or_other"] += 1
        if term.get("eos_observed"):
            counts["eos_observed"] += 1
        if term["think_end_reached"]:
            counts["think_end_reached"] += 1
        if term["missing_answer"]:
            counts["missing_answer"] += 1
    return dict(counts)


def _config_summary(condition_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        families[r["family"]].append(r)
    by_task = group_values_by_task(records, "task_id", lambda r: 1.0 if r["correct"] else 0.0)
    acc_ci = cluster_bootstrap_ci(by_task)
    return {
        "condition_id": condition_id,
        "mode": records[0]["condition"]["mode"],
        "n": len(records),
        "n_tasks": len({r["task_id"] for r in records}),
        "deployment_description": _describe(records[0]["deployment"]),
        "accuracy": acc_ci["mean"],
        "accuracy_ci": (acc_ci["lo"], acc_ci["hi"]) if acc_ci["mean"] is not None else None,
        "accuracy_ci_clusters": acc_ci,
        "reasoning_tokens": summarize([r["reasoning_tokens"] for r in records]),
        "answer_tokens": summarize([r["answer_tokens"] for r in records]),
        "total_tokens": summarize([r["total_tokens"] for r in records]),
        "wall_ms": summarize([r["trace"]["wall_ms"] for r in records]),
        "tokens_per_sec": summarize([r["trace"]["tokens_per_sec"] for r in records]),
        "ttft_ms": summarize(
            [r["trace"]["ttft_ms"] for r in records if r["trace"]["ttft_ms"] is not None]
        ),
        "unique_token_ratio": summarize([r["metrics"]["unique_token_ratio"] for r in records]),
        "repeated_ngram_fraction_4": summarize(
            [r["metrics"]["repeated_ngram_fraction_4"] for r in records]
        ),
        "text_repeat_ratio": summarize([r["metrics"]["text_repeat_ratio"] for r in records]),
        "termination": _termination_counts(records),
        "per_family": {
            fam: {
                "n": len(rs),
                "accuracy": _accuracy_macro(rs),
                "reasoning_tokens_mean": summarize([r["reasoning_tokens"] for r in rs])["mean"],
            }
            for fam, rs in sorted(families.items())
        },
    }


def _describe(deployment: dict[str, Any]) -> str:
    model = deployment["model"]
    runtime = deployment["runtime"]
    decoding = deployment["decoding"]
    quant = model.get("quantization") or "none"
    return (
        f"{model['repo_id']}@{model['revision'][:12]} "
        f"precision={model['precision']} quant={quant} "
        f"| {runtime['runtime']} {runtime['runtime_version']} on {runtime['hardware']} "
        f"| {decoding['mode']}"
    )


def _replay_stats(rs: list[dict[str, Any]]) -> dict[str, Any]:
    base = rs[0]["trace"]["token_ids"]
    identical = all(r["trace"]["token_ids"] == base for r in rs[1:])
    first_div = None
    if not identical:
        for r in rs[1:]:
            seq = r["trace"]["token_ids"]
            for i in range(max(len(seq), len(base))):
                a = base[i] if i < len(base) else None
                b = seq[i] if i < len(seq) else None
                if a != b:
                    first_div = i
                    break
            if first_div is not None:
                break
    base_digests = [s.get("logprob_digest") for s in rs[0]["trace"]["steps"]]
    digest_div = None
    if any(d is not None for d in base_digests) and len(rs) > 1:
        other = rs[1]["trace"]["steps"]
        for i in range(max(len(base_digests), len(other))):
            bd = base_digests[i] if i < len(base_digests) else None
            od = other[i].get("logprob_digest") if i < len(other) else None
            if bd != od:
                digest_div = i
                break
    answers = [r["answer_normalized"] for r in rs]
    return {
        "n_repeats": len(rs),
        "identical_tokens": identical,
        "first_token_divergence": first_div,
        "first_logprob_digest_divergence": digest_div,
        "answer_agreement": len(set(answers)) == 1,
        "reasoning_token_std": float(np.std([r["reasoning_tokens"] for r in rs]))
        if len(rs) > 1
        else 0.0,
        "wall_ms_std": float(np.std([r["trace"]["wall_ms"] for r in rs])) if len(rs) > 1 else 0.0,
    }


def greedy_replay_variation(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r["condition"]["mode"] == "greedy":
            groups[(r["condition_id"], r["task_id"])].append(r)
    per_task = []
    for (condition_id, task_id), rs in sorted(groups.items()):
        if len(rs) < 2:
            continue
        rs = sorted(rs, key=lambda r: r["condition"]["repeat"])
        per_task.append({"condition_id": condition_id, "task_id": task_id, **_replay_stats(rs)})
    n = len(per_task)
    identical_n = sum(1 for p in per_task if p["identical_tokens"])
    return {
        "available": bool(per_task),
        "n_task_conditions": n,
        "token_identical_count": identical_n,
        "token_identical_rate": (identical_n / n) if n else None,
        "answer_agreement_rate": (sum(1 for p in per_task if p["answer_agreement"]) / n)
        if n
        else None,
        "mean_reasoning_token_std": float(np.mean([p["reasoning_token_std"] for p in per_task]))
        if n
        else None,
        "mean_wall_ms_std": float(np.mean([p["wall_ms_std"] for p in per_task])) if n else None,
        "per_task": per_task,
        "note": "Same-process, same-condition greedy replay on this machine.",
    }


def same_seed_replay_variation(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Sampled generations sharing one condition and one explicit seed."""
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r["condition"]["mode"] == "sampled" and r["condition"].get("seed") is not None:
            groups[(r["condition_id"], r["task_id"], int(r["condition"]["seed"]))].append(r)
    per_task = []
    for (condition_id, task_id, seed), rs in sorted(groups.items()):
        if len(rs) < 2:
            continue
        rs = sorted(rs, key=lambda r: r["condition"]["repeat"])
        per_task.append(
            {"condition_id": condition_id, "task_id": task_id, "seed": seed, **_replay_stats(rs)}
        )
    n = len(per_task)
    return {
        "available": bool(per_task),
        "n_task_condition_seeds": n,
        "token_identical_count": sum(1 for p in per_task if p["identical_tokens"]),
        "token_identical_rate": (sum(1 for p in per_task if p["identical_tokens"]) / n)
        if n
        else None,
        "answer_agreement_rate": (sum(1 for p in per_task if p["answer_agreement"]) / n)
        if n
        else None,
        "mean_reasoning_token_std": float(np.mean([p["reasoning_token_std"] for p in per_task]))
        if n
        else None,
        "per_task": per_task,
        "note": "Same seed, same sampler configuration, re-executed.",
    }


def across_seed_variation(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Different seeds under the same sampler policy (task-clustered)."""
    by_condition_task: dict[tuple[str, str], dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in records:
        if r["condition"]["mode"] == "sampled" and r["condition"].get("seed") is not None:
            by_condition_task[(r["condition_id"], r["task_id"])][
                int(r["condition"]["seed"])
            ].append(r)
    per_task = []
    for (condition_id, task_id), by_seed in sorted(by_condition_task.items()):
        if len(by_seed) < 2:
            continue
        seed_answer = {seed: rs[0].get("answer_normalized") for seed, rs in sorted(by_seed.items())}
        seed_correct = {
            seed: float(np.mean([1.0 if r["correct"] else 0.0 for r in rs]))
            for seed, rs in by_seed.items()
        }
        seed_reasoning = {
            seed: float(np.mean([r["reasoning_tokens"] for r in rs]))
            for seed, rs in by_seed.items()
        }
        answers = list(seed_answer.values())
        per_task.append(
            {
                "condition_id": condition_id,
                "task_id": task_id,
                "n_seeds": len(by_seed),
                "distinct_answers": len(set(answers)),
                "accuracy_std_across_seeds": float(np.std(list(seed_correct.values()))),
                "reasoning_tokens_std_across_seeds": float(np.std(list(seed_reasoning.values()))),
            }
        )
    n = len(per_task)
    return {
        "available": bool(per_task),
        "n_task_conditions": n,
        "mean_distinct_answers": float(np.mean([p["distinct_answers"] for p in per_task]))
        if n
        else None,
        "mean_accuracy_std_across_seeds": float(
            np.mean([p["accuracy_std_across_seeds"] for p in per_task])
        )
        if n
        else None,
        "mean_reasoning_tokens_std_across_seeds": float(
            np.mean([p["reasoning_tokens_std_across_seeds"] for p in per_task])
        )
        if n
        else None,
        "per_task": per_task,
        "note": "Different seeds under a fixed sampler policy; conceptually distinct from replay failure.",
    }


def noise_floor(traces: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "greedy_replay": greedy_replay_variation(traces),
        "same_seed_replay": same_seed_replay_variation(traces),
        "across_seed": across_seed_variation(traces),
    }


def _noise_scale_for(records: list[dict[str, Any]], mode: str) -> float | None:
    """Mean within-task std for repeated observations under a mode."""
    groups: dict[tuple[str, Any], list[float]] = defaultdict(list)
    for r in records:
        c = r["condition"]
        if c["mode"] != mode:
            continue
        groups[(r["task_id"], c.get("seed"))].append(float(r["correct"]))
    return within_cluster_std({f"{k[0]}|{k[1]}": v for k, v in groups.items()})


def analyze_run(store: RunStore) -> dict[str, Any]:
    traces = store.read_traces()
    probes = store.read_probes()
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in traces:
        by_condition[r["condition_id"]].append(r)

    configs = {cid: _config_summary(cid, rs) for cid, rs in sorted(by_condition.items())}

    trajectory_states: Counter[str] = Counter()
    probe_per_task: list[dict[str, Any]] = []
    probes_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in probes:
        probes_by_task[p["task_id"]].append(p)
    n_stable_sufficient = 0
    for task_id, ps in sorted(probes_by_task.items()):
        ps = sorted(ps, key=lambda p: p.get("reasoning_prefix_tokens", p.get("prefix_len", 0)))
        greedy_correct = next(
            (
                r["correct"]
                for r in traces
                if r["task_id"] == task_id and r["condition"]["mode"] == "greedy"
            ),
            False,
        )
        lengths = [p.get("reasoning_prefix_tokens", p.get("prefix_len", 0)) for p in ps]
        state = classify_transitions(
            greedy_correct, [bool(p["correct"]) for p in ps], prefix_token_lengths=lengths
        )
        trajectory_states[state.primary] += 1
        if state.stable_sufficient_prefix_tokens is not None:
            n_stable_sufficient += 1
        probe_per_task.append(
            {
                "task_id": task_id,
                "family": ps[0]["family"],
                "primary_state": state.primary,
                "flips": state.flips,
                "first_correct_probe_index": state.first_correct_probe_index,
                "first_correct_prefix_tokens": state.first_correct_prefix_tokens,
                "stable_sufficient_prefix_tokens": state.stable_sufficient_prefix_tokens,
                "final_correct": state.final_correct,
                "prefix_correct": [bool(p["correct"]) for p in ps],
                "prefix_token_lengths": lengths,
            }
        )

    analysis = {
        "run_id": store.run_id,
        "artifact_id": store.artifact_id,
        "deployment_id": store.deployment_id,
        "condition_ids": store.condition_ids,
        "deployment_description": store.manifest.get("deployment_description"),
        "task_pack": store.manifest.get("task_pack"),
        "n_generations": len(traces),
        "n_probes": len(probes),
        "n_tasks": len({r["task_id"] for r in traces}),
        "configs": configs,
        "noise_floor": noise_floor(traces),
        "probes": {
            "n": len(probes),
            "supported": store.manifest.get("probe_supported"),
            "n_stable_sufficient": n_stable_sufficient,
            "trajectory_state_counts": dict(trajectory_states),
            "per_task": probe_per_task,
        },
        "status": store.manifest.get("status", {}),
    }
    store.write_analysis(analysis)
    return analysis
