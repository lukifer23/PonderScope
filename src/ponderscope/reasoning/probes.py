"""Forced-finalization prefix probes.

This is a measurement primitive, not a claimed contribution: probing reasoning
prefixes already exists in the literature. PonderScope uses it to observe
answer-state transitions along a trajectory.
"""

from __future__ import annotations

from typing import Any

from ..backends.base import Backend, CaptureSpec
from ..config.identity import DecodingPolicy
from ..tasks.scorers import normalize
from .parse import parse_reasoning
from .transitions import prefix_lengths


def run_prefix_probes(
    backend: Backend,
    *,
    family: str,
    answer: str,
    prompt_token_ids: list[int],
    prefix_token_ids: list[int],
    n_probes: int,
    probe_decoding: DecodingPolicy,
    capture: CaptureSpec,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for length in prefix_lengths(len(prefix_token_ids), n_probes):
        prefix = prefix_token_ids[:length]
        trace = backend.probe(prompt_token_ids, prefix, probe_decoding, capture)
        parsed = parse_reasoning(trace.text, trace.think_end_reached)
        answer_raw = parsed.answer_raw
        answer_norm = normalize(family, answer_raw) if answer_raw is not None else None
        # `normalize` strips an answer cue if the probe text still contains one.
        if answer_norm is None and answer_raw is not None:
            answer_norm = normalize(family, answer_raw)
        results.append(
            {
                "prefix_len": length,
                "answer_raw": answer_raw,
                "answer_normalized": answer_norm,
                "correct": answer_norm == answer.strip(),
                "generated_tokens": trace.generated_tokens,
                "finish_reason": trace.finish_reason,
                "think_end_reached": trace.think_end_reached,
                "text": trace.text,
            }
        )
    return results
