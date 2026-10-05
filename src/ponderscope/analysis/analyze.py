"""Analyze immutable saved evidence. Never touches the model."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from ..evidence.run import RunStore
from ..reasoning.transitions import classify_transitions
from .stats import bootstrap_ci, summarize


def _accuracy(records: list[dict[str, Any]]) -> float | None:
    if not records:
        return None
    return float(np.mean([1.0 if r["correct"] else 0.0 for r in records]))


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
        if term["think_end_reached"]:
            counts["think_end_reached"] += 1
        if term["missing_answer"]:
            counts["missing_answer"] += 1
    return dict(counts)


def _config_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        families[r["family"]].append(r)
    return {
        "mode": records[0]["condition"]["mode"],
        "n": len(records),
        "n_tasks": len({r["task_id"] for r in records}),
        "deployment_description": _describe(records[0]["deployment"]),
        "accuracy": _accuracy(records),
        "accuracy_ci": bootstrap_ci([1.0 if r["correct"] else 0.0 for r in records]),
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
                "accuracy": _accuracy(rs),
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


def greedy_determinism(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r["condition"]["mode"] == "greedy":
            groups[(r["config_id"], r["task_id"])].append(r)
    per_task: list[dict[str, Any]] = []
    for (config_id, task_id), rs in sorted(groups.items()):
        if len(rs) < 2:
            continue
        rs = sorted(rs, key=lambda r: r["condition"]["repeat"])
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
        # per-token logprob digest divergence (if captured)
        digest_div = None
        base_steps = rs[0]["trace"]["steps"]
        base_digests = [s.get("logprob_digest") for s in base_steps]
        if any(d is not None for d in base_digests) and len(rs) > 1:
            other_steps = rs[1]["trace"]["steps"]
            for i in range(max(len(base_digests), len(other_steps))):
                base_d = base_digests[i] if i < len(base_digests) else None
                other_d = other_steps[i].get("logprob_digest") if i < len(other_steps) else None
                if base_d != other_d:
                    digest_div = i
                    break
        per_task.append(
            {
                "config_id": config_id,
                "task_id": task_id,
                "repeats": len(rs),
                "identical_tokens": identical,
                "first_token_divergence": first_div,
                "first_logprob_digest_divergence": digest_div,
            }
        )
    n = len(per_task)
    identical_n = sum(1 for p in per_task if p["identical_tokens"])
    return {
        "n_task_configs": n,
        "token_identical_count": identical_n,
        "token_identical_rate": (identical_n / n) if n else None,
        "per_task": per_task,
        "note": "Determinism here is same-process, same-condition replay on this machine.",
    }


def sampled_noise(records: list[dict[str, Any]]) -> dict[str, Any]:
    sampled = [r for r in records if r["condition"]["mode"] == "sampled"]
    if not sampled:
        return {"available": False}
    # across-seed variation per task
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in sampled:
        by_task[r["task_id"]].append(r)
    acc_spreads: list[float] = []
    tok_spreads: list[float] = []
    flips = 0
    for rs in by_task.values():
        answers = [r["answer_normalized"] for r in rs]
        flips += sum(1 for a, b in zip(answers, answers[1:], strict=False) if a != b)
        accs = [1.0 if r["correct"] else 0.0 for r in rs]
        toks = [r["reasoning_tokens"] for r in rs]
        if len(rs) > 1:
            acc_spreads.append(float(np.std(accs)))
            tok_spreads.append(float(np.std(toks)))
    return {
        "available": True,
        "n": len(sampled),
        "accuracy_std_across_runs_mean": float(np.mean(acc_spreads)) if acc_spreads else None,
        "reasoning_tokens_std_across_runs_mean": float(np.mean(tok_spreads))
        if tok_spreads
        else None,
        "consecutive_answer_flip_count": flips,
        "consecutive_answer_flip_rate": flips / max(1, len(sampled) - 1),
    }


def analyze_run(store: RunStore) -> dict[str, Any]:
    traces = store.read_traces()
    probes = store.read_probes()
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in traces:
        by_config[r["config_id"]].append(r)

    configs = {cid: _config_summary(rs) for cid, rs in sorted(by_config.items())}

    trajectory_states: Counter[str] = Counter()
    probe_per_task: list[dict[str, Any]] = []
    probes_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in probes:
        probes_by_task[p["task_id"]].append(p)
    for task_id, ps in sorted(probes_by_task.items()):
        ps = sorted(ps, key=lambda p: p["prefix_len"])
        greedy_correct = next(
            (
                r["correct"]
                for r in traces
                if r["task_id"] == task_id and r["condition"]["mode"] == "greedy"
            ),
            False,
        )
        state = classify_transitions(greedy_correct, [bool(p["correct"]) for p in ps])
        trajectory_states[state.primary] += 1
        probe_per_task.append(
            {
                "task_id": task_id,
                "family": ps[0]["family"],
                "primary_state": state.primary,
                "flips": state.flips,
                "first_correct_prefix": state.first_correct_prefix,
                "final_correct": state.final_correct,
                "prefix_correct": [bool(p["correct"]) for p in ps],
            }
        )

    analysis = {
        "run_id": store.run_id,
        "config_id": store.config_id,
        "deployment_description": store.manifest.get("deployment_description"),
        "n_generations": len(traces),
        "n_probes": len(probes),
        "n_tasks": len({r["task_id"] for r in traces}),
        "configs": configs,
        "noise_floor": {
            "greedy": greedy_determinism(traces),
            "sampled": sampled_noise(traces),
        },
        "probes": {
            "n": len(probes),
            "trajectory_state_counts": dict(trajectory_states),
            "per_task": probe_per_task,
        },
        "status": store.manifest.get("status", {}),
    }
    store.write_analysis(analysis)
    return analysis
