"""Termination-calibration helpers.

These are bounded qualification helpers that drive the real backend directly (no
sealed run) to answer instrument questions before a calibration study: chiefly,
whether shorter greedy generations are exact prefixes of longer ones. The
resulting JSON is a qualification artifact, not sealed research evidence.
"""

from __future__ import annotations

from typing import Any

from .analysis.analyze import prefix_invariance
from .backends.base import Backend, CaptureSpec
from .config.identity import DecodingPolicy


def run_same_seed_replay_subset(
    backend: Backend,
    prompts: list[dict[str, Any]],
    decoding: DecodingPolicy,
    *,
    repeats: int = 3,
) -> dict[str, Any]:
    """Replay one seeded decoding on predeclared prompts to test reproducibility.

    These are *technical replicates*: they prove the custom generated-history
    processor is deterministic and are deliberately kept separate from any
    primary stochastic-draw N.
    """
    repeats = max(2, int(repeats))
    per_task: list[dict[str, Any]] = []
    for prompt in prompts:
        traces = [
            backend.generate(prompt["token_ids"], decoding, CaptureSpec.minimal())
            for _ in range(repeats)
        ]
        base = traces[0].token_ids
        identical = all(t.token_ids == base for t in traces[1:])
        first_div: int | None = None
        if not identical:
            for trace in traces[1:]:
                seq = trace.token_ids
                for i in range(max(len(base), len(seq))):
                    a = base[i] if i < len(base) else None
                    b = seq[i] if i < len(seq) else None
                    if a != b:
                        first_div = i
                        break
                if first_div is not None:
                    break
        per_task.append(
            {
                "task_id": prompt["task_id"],
                "family": prompt["family"],
                "n_executions": len(traces),
                "token_identical": identical,
                "first_token_divergence": first_div,
                "generated_tokens": [len(t.token_ids) for t in traces],
            }
        )
    return {
        "condition_id": decoding.condition_id,
        "presence_scope": decoding.presence_scope,
        "seed": decoding.seed,
        "repeats_per_task": repeats,
        "n_tasks": len(per_task),
        "all_token_identical": all(p["token_identical"] for p in per_task),
        "per_task": per_task,
        "note": (
            "Same-seed technical replay to confirm the custom logits processor is "
            "reproducible. These executions are NOT part of any primary "
            "stochastic-draw N."
        ),
    }


def run_cap_prefix_invariance(
    backend: Backend,
    prompts: list[dict[str, Any]],
    caps: list[int],
) -> dict[str, Any]:
    """Generate each prompt greedily at every cap and check prefix nesting."""
    caps = sorted({int(c) for c in caps})
    per_task: list[dict[str, Any]] = []
    for prompt in prompts:
        sequences: dict[int, list[int]] = {}
        for cap in caps:
            trace = backend.generate(
                prompt["token_ids"],
                DecodingPolicy(mode="greedy", max_tokens=cap),
                CaptureSpec.minimal(),
            )
            sequences[cap] = trace.token_ids
        per_task.append(
            {
                "task_id": prompt["task_id"],
                "family": prompt["family"],
                "lengths": {str(c): len(sequences[c]) for c in caps},
                "invariance": prefix_invariance(sequences),
            }
        )
    n_comparisons = sum(t["invariance"]["n_comparisons"] for t in per_task)
    sufficient = len(caps) >= 2 and all(t["invariance"]["n_comparisons"] >= 1 for t in per_task)
    all_exact = bool(sufficient and all(t["invariance"]["exact_prefix"] for t in per_task))
    return {
        "caps": caps,
        "n_prompts": len(prompts),
        "n_comparisons": n_comparisons,
        "sufficient": sufficient,
        "status": "ok" if sufficient else "insufficient_data",
        "all_exact_prefix": all_exact if sufficient else None,
        "per_task": per_task,
        "note": (
            "A single cap yields zero comparisons and is NOT a validation; "
            "all_exact_prefix is only meaningful with at least two distinct caps. "
            "If True, closure and lower-cap censoring can be derived from one "
            "generous greedy run without re-running each cap."
        ),
    }
