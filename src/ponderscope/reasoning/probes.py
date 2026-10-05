"""Forced-finalization prefix probes.

This is a measurement primitive, not a claimed contribution: probing reasoning
prefixes already exists in the literature. PonderScope uses it to observe
answer-state transitions along a trajectory.

The probe appends the model's think-end token followed by a fixed answer cue to
the prompt, so the *generated* text is the answer channel itself. The first
non-empty line of that continuation is taken as the raw answer.
"""

from __future__ import annotations

import re
from typing import Any

from ..backends.base import Backend, CaptureSpec
from ..config.identity import DecodingPolicy
from ..tasks.scorers import normalize, score
from .transitions import prefix_lengths

_SPECIAL = re.compile(r"<\|(?:im_end|im_start|endoftext)\|>|<think>|</think>|<\|[^|]*\|>")


def _first_line(text: str) -> str | None:
    for line in text.splitlines():
        cleaned = _SPECIAL.sub("", line).strip().strip("`'\"")
        if cleaned:
            return cleaned
    return None


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
        answer_raw = _first_line(trace.text)
        answer_norm = normalize(family, answer_raw) if answer_raw is not None else None
        results.append(
            {
                "prefix_len": length,
                "answer_raw": answer_raw,
                "answer_normalized": answer_norm,
                "correct": bool(answer_raw is not None and score(family, answer_raw, answer)),
                "generated_tokens": trace.generated_tokens,
                "finish_reason": trace.finish_reason,
                "think_end_reached": trace.think_end_reached,
                "text": trace.text,
            }
        )
    return results
