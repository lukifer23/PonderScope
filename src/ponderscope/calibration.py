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


def run_cap_prefix_invariance(
    backend: Backend,
    prompts: list[dict[str, Any]],
    caps: list[int],
) -> dict[str, Any]:
    """Generate each prompt greedily at every cap and check prefix nesting."""
    caps = sorted(int(c) for c in caps)
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
    return {
        "caps": caps,
        "n_prompts": len(prompts),
        "all_exact_prefix": all(t["invariance"]["exact_prefix"] for t in per_task),
        "per_task": per_task,
        "note": (
            "If all_exact_prefix is True, closure and lower-cap censoring can be "
            "derived from one generous greedy run without re-running each cap."
        ),
    }
