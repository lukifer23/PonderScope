"""Analyze immutable saved evidence. Never touches the model.

Three distinct measurement questions are kept separate:

1. **Budget outcomes** — success_at_budget, completion_rate, censored_rate,
   conditional accuracy among completed answer trials. A capped generation is a
   budget failure, NOT an observed wrong answer.
2. **Repeatability** — greedy replay, same-seed sampled replay (with ambiguity
   detection), and token-level across-seed variation that stays valid while
   generations are censored.
3. **Transitions** — prefix states over observed forced probes only, with a
   separate natural final status.

Statistical unit is the task.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from ..evidence.run import RunStore
from ..reasoning.transitions import (
    classify_transitions,
    natural_final_status_for_record,
)
from .draws import (
    collapse_to_stochastic_draws,
    draw_summary,
    draws_by_condition,
)
from .stats import (
    cluster_bootstrap_ci,
    group_values_by_task,
    summarize,
    within_cluster_std,
)
from .survival import survival_by_family, survival_summary

NULLABLE_FLOAT = float | None


def prefix_invariance(traces_by_cap: dict[int, list[int]]) -> dict[str, Any]:
    """Check that shorter greedy generations are exact prefixes of longer ones.

    If this holds, closure can be discovered with a single generous run and the
    censoring a smaller cap would have caused derived without re-running it.

    A single cap yields **zero** comparisons, which is not evidence of anything:
    that case is reported as ``status="insufficient_data"`` with
    ``exact_prefix=None`` rather than a vacuous ``True``.
    """
    caps = sorted(traces_by_cap)
    checks = []
    for short, long in zip(caps, caps[1:], strict=False):
        a = traces_by_cap[short]
        b = traces_by_cap[long]
        is_prefix = len(a) <= len(b) and list(b[: len(a)]) == list(a)
        checks.append(
            {
                "short_cap": short,
                "long_cap": long,
                "short_len": len(a),
                "long_len": len(b),
                "exact_prefix": is_prefix,
            }
        )
    n_comparisons = len(checks)
    if n_comparisons == 0:
        return {
            "caps": caps,
            "n_comparisons": 0,
            "exact_prefix": None,
            "status": "insufficient_data",
            "checks": checks,
        }
    ok = all(c["exact_prefix"] for c in checks)
    return {
        "caps": caps,
        "n_comparisons": n_comparisons,
        "exact_prefix": ok,
        "status": "ok" if ok else "divergent",
        "checks": checks,
    }


def natural_final_status_of(record: dict[str, Any]) -> str:
    """Return the saved status, deriving it for pre-Phase-1.2 evidence."""
    status = record.get("natural_final_status")
    if status:
        return str(status)
    return natural_final_status_for_record(record)


def _accuracy_macro(records: list[dict[str, Any]]) -> float | None:
    by_task = group_values_by_task(records, "task_id", lambda r: 1.0 if r["correct"] else 0.0)
    if not by_task:
        return None
    return float(np.mean([np.mean(v) for v in by_task.values()]))


def _termination_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for r in records:
        term = r["termination"]
        counts[f"status:{natural_final_status_of(r)}"] += 1
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


def _budget_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate end-to-end success at budget from observed-answer accuracy."""
    n = len(records)
    if n == 0:
        return {
            "n_attempted": 0,
            "n_completed": 0,
            "n_answer_observed": 0,
            "n_censored": 0,
            "n_error": 0,
            "n_unparseable": 0,
            "success_at_budget": None,
            "completion_rate": None,
            "conditional_accuracy_given_completed": None,
            "answer_observed_accuracy": None,
            "censored_rate": None,
            "error_rate": None,
            "unparseable_rate": None,
            "answer_observation_rate": None,
        }
    statuses = [natural_final_status_of(r) for r in records]
    completed = [r for r in records if r["termination"]["terminated_by_eos"]]
    completed_observed = [r for r in completed if r["answer_normalized"] is not None]
    observed = [r for r in records if r["answer_normalized"] is not None]
    correct = [r for r in records if r["correct"]]
    n_censored = sum(1 for s in statuses if s == "censored")
    n_error = sum(1 for s in statuses if s == "error")
    n_unparseable = sum(1 for s in statuses if s == "unparseable")
    return {
        "n_attempted": n,
        "n_completed": len(completed),
        "n_answer_observed": len(observed),
        "n_censored": n_censored,
        "n_error": n_error,
        "n_unparseable": n_unparseable,
        "success_at_budget": len(correct) / n,
        "completion_rate": len(completed) / n,
        "conditional_accuracy_given_completed": (
            sum(1 for r in completed_observed if r["correct"]) / len(completed_observed)
            if completed_observed
            else None
        ),
        "answer_observed_accuracy": (
            sum(1 for r in observed if r["correct"]) / len(observed) if observed else None
        ),
        "censored_rate": n_censored / n,
        "error_rate": n_error / n,
        "unparseable_rate": n_unparseable / n,
        "answer_observation_rate": len(observed) / n,
    }


def _config_summary(condition_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        families[r["family"]].append(r)
    by_task = group_values_by_task(records, "task_id", lambda r: 1.0 if r["correct"] else 0.0)
    acc_ci = cluster_bootstrap_ci(by_task)
    ds = draw_summary(collapse_to_stochastic_draws(records))
    return {
        "condition_id": condition_id,
        "mode": records[0]["condition"]["mode"],
        "n": len(records),
        "n_executions": ds["n_executions"],
        "n_unique_draws": ds["n_unique_draws"],
        "n_ambiguous_draws": ds["n_ambiguous_draws"],
        "n_tasks": len({r["task_id"] for r in records}),
        "deployment_description": _describe(records[0]["deployment"]),
        "success_at_budget": acc_ci["mean"],
        "success_at_budget_ci": (acc_ci["lo"], acc_ci["hi"])
        if acc_ci["mean"] is not None
        else None,
        "success_at_budget_ci_clusters": acc_ci,
        "outcomes": _budget_metrics(records),
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
                "success_at_budget": _accuracy_macro(rs),
                "outcomes": _budget_metrics(rs),
                "reasoning_tokens_mean": summarize([r["reasoning_tokens"] for r in rs])["mean"],
            }
            for fam, rs in sorted(families.items())
        },
    }


def _describe(deployment: dict[str, Any]) -> str:
    model = deployment.get("weight_variant") or deployment.get("model", {})
    source = model.get("source", model)
    runtime = deployment.get("runtime", {})
    decoding = deployment.get("decoding", {})
    quant = model.get("quantization") or "none"
    repo_id = source.get("repo_id", "?")
    revision = str(source.get("revision", ""))[:12]
    return (
        f"{repo_id}@{revision} "
        f"precision={model.get('precision')} quant={quant} "
        f"| {runtime.get('runtime')} {runtime.get('runtime_version')} "
        f"on {runtime.get('hardware')} "
        f"| {decoding.get('mode')}"
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
    observed_answers = [a for a in answers if a is not None]
    if not observed_answers:
        answer_agreement: bool | None = None
    elif any(a is None for a in answers):
        answer_agreement = False  # some repeats never produced an answer
    else:
        answer_agreement = len(set(answers)) == 1
    return {
        "n_repeats": len(rs),
        "identical_tokens": identical,
        "seed_ambiguous": not identical,
        "first_token_divergence": first_div,
        "first_logprob_digest_divergence": digest_div,
        "answer_agreement": answer_agreement,
        "answer_observed": bool(observed_answers),
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
    observed = [p for p in per_task if p["answer_observed"]]
    return {
        "available": bool(per_task),
        "n_task_conditions": n,
        "token_identical_count": identical_n,
        "token_identical_rate": (identical_n / n) if n else None,
        "answer_agreement_rate": (sum(1 for p in observed if p["answer_agreement"]) / len(observed))
        if observed
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
    ambiguous = sum(1 for p in per_task if p["seed_ambiguous"])
    observed = [p for p in per_task if p["answer_observed"]]
    return {
        "available": bool(per_task),
        "n_task_condition_seeds": n,
        "token_identical_count": sum(1 for p in per_task if p["identical_tokens"]),
        "token_identical_rate": (sum(1 for p in per_task if p["identical_tokens"]) / n)
        if n
        else None,
        "ambiguous_seed_count": ambiguous,
        "answer_agreement_rate": (sum(1 for p in observed if p["answer_agreement"]) / len(observed))
        if observed
        else None,
        "mean_reasoning_token_std": float(np.mean([p["reasoning_token_std"] for p in per_task]))
        if n
        else None,
        "per_task": per_task,
        "note": "Same seed, same sampler configuration, re-executed.",
    }


def cross_seed_token_variation(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Token-level across-seed variation, valid even when answers are censored."""
    groups: dict[tuple[str, str], dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in records:
        if r["condition"]["mode"] == "sampled" and r["condition"].get("seed") is not None:
            groups[(r["condition_id"], r["task_id"])][int(r["condition"]["seed"])].append(r)
    per_task = []
    for (condition_id, task_id), by_seed in sorted(groups.items()):
        if len(by_seed) < 2:
            continue
        base_seq = None
        first_div = None
        for seed in sorted(by_seed):
            seq = sorted(by_seed[seed], key=lambda r: r["condition"]["repeat"])[0]["trace"][
                "token_ids"
            ]
            if base_seq is None:
                base_seq = seq
                continue
            for i in range(max(len(base_seq), len(seq))):
                a = base_seq[i] if i < len(base_seq) else None
                b = seq[i] if i < len(seq) else None
                if a != b:
                    if first_div is None or i < first_div:
                        first_div = i
                    break
        lengths = [float(np.mean([r["reasoning_tokens"] for r in rs])) for rs in by_seed.values()]
        per_task.append(
            {
                "condition_id": condition_id,
                "task_id": task_id,
                "n_seeds": len(by_seed),
                "first_token_divergence": first_div,
                "reasoning_tokens_std_across_seeds": float(np.std(lengths))
                if len(lengths) > 1
                else 0.0,
            }
        )
    n = len(per_task)
    return {
        "available": bool(per_task),
        "n_task_conditions": n,
        "mean_first_token_divergence": float(
            np.mean(
                [
                    p["first_token_divergence"]
                    for p in per_task
                    if p["first_token_divergence"] is not None
                ]
            )
        )
        if any(p["first_token_divergence"] is not None for p in per_task)
        else None,
        "mean_reasoning_tokens_std_across_seeds": float(
            np.mean([p["reasoning_tokens_std_across_seeds"] for p in per_task])
        )
        if n
        else None,
        "per_task": per_task,
        "note": "Seed variation is observable at the token level even when all final answers are censored.",
    }


def across_seed_variation(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Different seeds under the same sampler policy (task-clustered).

    Final-answer diversity/accuracy is reported only over seeds with an observed
    answer. A seed whose repeats diverged is flagged ambiguous and its answer is
    never silently taken from the first repeat.
    """
    by_condition_task: dict[tuple[str, str], dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in records:
        if r["condition"]["mode"] == "sampled" and r["condition"].get("seed") is not None:
            by_condition_task[(r["condition_id"], r["task_id"])][
                int(r["condition"]["seed"])
            ].append(r)
    per_task: list[dict[str, Any]] = []
    for (condition_id, task_id), by_seed in sorted(by_condition_task.items()):
        if len(by_seed) < 2:
            continue
        seed_answer: dict[int, str | None] = {}
        seed_correct: dict[int, float] = {}
        ambiguous = 0
        for seed, rs in by_seed.items():
            rs = sorted(rs, key=lambda r: r["condition"]["repeat"])
            identical = all(r["trace"]["token_ids"] == rs[0]["trace"]["token_ids"] for r in rs[1:])
            if not identical:
                ambiguous += 1
                seed_answer[seed] = None  # never use rs[0] for an ambiguous seed
                continue
            seed_answer[seed] = rs[0].get("answer_normalized")
            seed_correct[seed] = float(np.mean([1.0 if r["correct"] else 0.0 for r in rs]))
        observed_answers = [a for a in seed_answer.values() if a is not None]
        observed_correct = [seed_correct[s] for s, a in seed_answer.items() if a is not None]
        per_task.append(
            {
                "condition_id": condition_id,
                "task_id": task_id,
                "n_seeds": len(by_seed),
                "n_seeds_with_observed_answer": len(observed_answers),
                "n_ambiguous_seeds": ambiguous,
                "distinct_answers": len(set(observed_answers)) if observed_answers else None,
                "answer_observation_rate": len(observed_answers) / len(by_seed),
                "accuracy_std_across_seeds": (
                    float(np.std(observed_correct)) if len(observed_correct) > 1 else None
                ),
            }
        )
    n = len(per_task)
    n_total = len(by_condition_task)
    n_estimable = sum(1 for p in per_task if p["distinct_answers"] is not None)
    return {
        "available": bool(per_task),
        "n_task_conditions": n,
        "n_task_conditions_total": n_total,
        "n_task_conditions_with_estimable_answer_diversity": n_estimable,
        "proportion_estimable": (n_estimable / n_total) if n_total else None,
        "mean_distinct_answers": _mean_or_none(
            [p["distinct_answers"] for p in per_task if p["distinct_answers"] is not None]
        ),
        "mean_accuracy_std_across_seeds": _mean_or_none(
            [
                p["accuracy_std_across_seeds"]
                for p in per_task
                if p["accuracy_std_across_seeds"] is not None
            ]
        ),
        "any_ambiguous_seeds": any(p["n_ambiguous_seeds"] > 0 for p in per_task),
        "per_task": per_task,
        "note": (
            "Final-answer diversity is only defined over observed answers. A mean is "
            "always reported alongside its support count "
            "(n_task_conditions_with_estimable_answer_diversity / "
            "n_task_conditions_total); None means unobserved, not zero."
        ),
    }


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def noise_floor(traces: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "greedy_replay": greedy_replay_variation(traces),
        "same_seed_replay": same_seed_replay_variation(traces),
        "across_seed": across_seed_variation(traces),
        "cross_seed_tokens": cross_seed_token_variation(traces),
    }


def _is_reasoning_closure(record: dict[str, Any]) -> bool:
    """PRIMARY endpoint: a native think-end reasoning closure was observed.

    EOS alone is deliberately NOT enough: a generation can stop without the
    model emitting its native reasoning-close token, and that is not reasoning
    closure.
    """
    return bool(record["termination"].get("think_end_reached"))


def _is_generation_termination(record: dict[str, Any]) -> bool:
    """SECONDARY endpoint: the generation terminated via EOS."""
    return bool(record["termination"].get("terminated_by_eos"))


def _time_to_closure(record: dict[str, Any]) -> float:
    """Generated tokens to closure (or to the observation horizon if censored)."""
    return float(record["total_tokens"])


def _survival_section(by_condition: dict[str, list[dict[str, Any]]], tau: float) -> dict[str, Any]:
    per_condition: dict[str, Any] = {}
    for cid, rs in by_condition.items():
        draws = collapse_to_stochastic_draws(rs)
        # Primary KM/RMST use unique stochastic draws, never technical repeats.
        primary_records = [d["record"] for d in draws if d["record"] is not None]
        reasoning = survival_summary(
            [_time_to_closure(r) for r in primary_records],
            [_is_reasoning_closure(r) for r in primary_records],
            tau=tau,
        )
        termination = survival_summary(
            [_time_to_closure(r) for r in primary_records],
            [_is_generation_termination(r) for r in primary_records],
            tau=tau,
        )
        summary = dict(reasoning)
        summary.update(
            {
                "n_executions": sum(d["n_executions"] for d in draws),
                "n_unique_draws": len(draws),
                "n_ambiguous_draws": sum(1 for d in draws if d["ambiguous"]),
                "n_draws_used": len(primary_records),
                "generation_termination": termination,
                "by_family": survival_by_family(
                    primary_records,
                    time_key=_time_to_closure,
                    event_key=_is_reasoning_closure,
                    tau=tau,
                ),
            }
        )
        per_condition[cid] = summary
    return {
        "tau": tau,
        "time_definition": "generated tokens to closure (observation horizon if censored)",
        "event_definition": "PRIMARY: native think-end reasoning closure observed",
        "secondary_event_definition": "SECONDARY: EOS observed (generation termination)",
        "censoring_definition": "max-token observation horizon reached before closure",
        "analysis_unit": "unique stochastic draws (same-seed technical repeats collapsed)",
        "per_condition": per_condition,
        "note": (
            "max_tokens is an observation horizon, not a natural stopping threshold. "
            "An EOS without a native think-end is not counted as reasoning closure."
        ),
    }


def _loop_section(by_condition: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    per_condition: dict[str, Any] = {}
    for cid, rs in by_condition.items():
        loops = [r["loop"] for r in rs if r.get("loop")]
        if not loops:
            per_condition[cid] = {"available": False}
            continue
        onsets = [
            loop["degeneration_onset_index"]
            for loop in loops
            if loop.get("degeneration_onset_index") is not None
        ]
        per_condition[cid] = {
            "available": True,
            "n": len(loops),
            "longest_run_length": summarize([loop["longest_run_length"] for loop in loops]),
            "degeneration_onset_index": summarize(onsets),
            "n_with_degeneration_onset": len(onsets),
            "special_longest_run_count": sum(
                1 for loop in loops if loop.get("longest_run_is_special")
            ),
            "top_motif_count": summarize([loop["top_motif_count"] for loop in loops]),
        }
    return {
        "per_condition": per_condition,
        "note": (
            "Descriptive loop structure with an explicitly documented onset rule; "
            "not a validated universal loop detector."
        ),
    }


def _noise_scale_for(records: list[dict[str, Any]], mode: str) -> NULLABLE_FLOAT:
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
    n_observed_stable = 0
    for task_id, ps in sorted(probes_by_task.items()):
        ps = sorted(ps, key=lambda p: p.get("reasoning_prefix_tokens", p.get("prefix_len", 0)))
        greedy_record = next(
            (r for r in traces if r["task_id"] == task_id and r["condition"]["mode"] == "greedy"),
            None,
        )
        natural_status = natural_final_status_of(greedy_record) if greedy_record else "censored"
        lengths = [p.get("reasoning_prefix_tokens", p.get("prefix_len", 0)) for p in ps]
        state = classify_transitions(
            [bool(p["correct"]) for p in ps], natural_status, prefix_token_lengths=lengths
        )
        trajectory_states[state.prefix_state] += 1
        if state.stable_sufficient_with_natural_final_tokens is not None:
            n_stable_sufficient += 1
        if state.observed_probe_stable_from_tokens is not None:
            n_observed_stable += 1
        probe_per_task.append(
            {
                "task_id": task_id,
                "family": ps[0]["family"],
                "prefix_state": state.prefix_state,
                "prefix_flips": state.prefix_flips,
                "natural_final_status": state.natural_final_status,
                "natural_final_correct": state.natural_final_correct,
                "first_correct_probe_index": state.first_correct_probe_index,
                "first_correct_prefix_tokens": state.first_correct_prefix_tokens,
                "observed_probe_stable_from_tokens": state.observed_probe_stable_from_tokens,
                "stable_sufficient_with_natural_final_tokens": (
                    state.stable_sufficient_with_natural_final_tokens
                ),
                "harmful_overthinking_observed": state.harmful_overthinking_observed,
                "prefix_correct": [bool(p["correct"]) for p in ps],
                "prefix_token_lengths": lengths,
            }
        )

    spec_max = store.manifest.get("spec", {}).get("max_tokens")
    tau = (
        float(spec_max) if spec_max else float(max((r["total_tokens"] for r in traces), default=0))
    )

    draws_per_condition = draws_by_condition(traces)
    all_draws = [d for ds in draws_per_condition.values() for d in ds]
    draws_section = {
        "n_executions": len(traces),
        "n_unique_draws": len(all_draws),
        "n_ambiguous_draws": sum(1 for d in all_draws if d["ambiguous"]),
        "per_condition": {cid: draw_summary(ds) for cid, ds in draws_per_condition.items()},
        "note": (
            "Stochastic draw = presentation + condition + seed (no repeat). "
            "Same-seed technical repeats collapse to one draw iff token-identical; "
            "divergent draws are flagged ambiguous and excluded from primary summaries."
        ),
    }

    analysis = {
        "interpretation": "phase1.3b",
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
        "draws": draws_section,
        "survival": _survival_section(by_condition, tau),
        "loop": _loop_section(by_condition),
        "probes": {
            "n": len(probes),
            "supported": store.manifest.get("probe_supported"),
            "n_stable_sufficient_with_natural_final": n_stable_sufficient,
            "n_observed_probe_stable": n_observed_stable,
            "prefix_state_counts": dict(trajectory_states),
            "per_task": probe_per_task,
        },
        "status": store.manifest.get("status", {}),
    }
    store.write_analysis(analysis)
    return analysis
