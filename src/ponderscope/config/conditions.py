"""Decoding conditions derived from an experiment spec.

A run may contain several decoding conditions; each has its own
``condition_id``. Seeds define trial state, not new conditions.
"""

from __future__ import annotations

from typing import Any

from .identity import DecodingPolicy
from .schema import ExperimentSpec


def condition_specs(spec: ExperimentSpec) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    for r in range(spec.greedy_repeats):
        conditions.append({"mode": "greedy", "repeat": r})
    for seed in spec.sampled_seeds:
        for r in range(spec.sampled_repeats_per_seed):
            conditions.append({"mode": "sampled", "seed": seed, "repeat": r})
    return conditions


def decoding_for(spec: ExperimentSpec, condition: dict[str, Any]) -> DecodingPolicy:
    if condition["mode"] == "greedy":
        return DecodingPolicy(
            mode="greedy",
            max_tokens=spec.max_tokens,
            stop_on_eos=True,
            context={"enable_thinking": True},
        )
    return DecodingPolicy(
        mode="sampled",
        max_tokens=spec.max_tokens,
        temperature=spec.sampled_temperature,
        top_p=spec.sampled_top_p,
        top_k=spec.sampled_top_k,
        min_p=spec.sampled_min_p,
        presence_penalty=spec.sampled_presence_penalty,
        presence_context_size=spec.sampled_presence_context_size,
        repetition_penalty=spec.sampled_repetition_penalty,
        repetition_context_size=spec.sampled_repetition_context_size,
        frequency_penalty=spec.sampled_frequency_penalty,
        frequency_context_size=spec.sampled_frequency_context_size,
        seed=int(condition["seed"]),
        stop_on_eos=True,
        context={"enable_thinking": True},
    )


def condition_identities(spec: ExperimentSpec) -> list[DecodingPolicy]:
    """One representative DecodingPolicy per distinct condition."""
    seen: dict[str, DecodingPolicy] = {}
    for condition in condition_specs(spec):
        policy = decoding_for(spec, condition)
        seen.setdefault(policy.condition_id, policy)
    return list(seen.values())
