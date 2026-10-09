"""Forced-finalization prefix probes.

This is a measurement primitive, not a claimed contribution: probing reasoning
prefixes already exists in the literature. PonderScope uses it to observe
answer-state transitions along a trajectory.

A probe replays a **reasoning-only** prefix (never including final-answer or EOS
tokens), appends the model's validated native closing sequence, and reads the
generated continuation. The continuation *is* the forced final-answer channel,
so it is parsed as a forced answer, never as a natural complete trajectory. No
hand-written answer cue is injected.
"""

from __future__ import annotations

from typing import Any

from ..backends.base import Backend, CaptureSpec
from ..config.identity import DecodingPolicy
from ..tasks.scorers import normalize, score
from .parse import extract_answer
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
    eos_token_ids: set[int] | None = None,
    checkpoints: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Run forced-finalization probes over reasoning prefixes.

    ``prefix_token_ids`` must contain reasoning tokens only; the caller is
    responsible for stripping the final channel. Raises
    ``PrefixProbeUnsupported`` (propagated from the backend) if the native
    closure cannot be validated. ``checkpoints`` selects explicit prefix token
    lengths (bounded by the actual reasoning length); otherwise a uniform grid is
    used.
    """
    eos_ids = eos_token_ids or set()
    results: list[dict[str, Any]] = []
    for length in prefix_lengths(len(prefix_token_ids), n_probes, checkpoints=checkpoints):
        prefix = prefix_token_ids[:length]
        trace = backend.probe(prompt_token_ids, prefix, probe_decoding, capture)
        answer_raw = extract_answer(trace.text)
        answer_norm = normalize(family, answer_raw) if answer_raw is not None else None
        results.append(
            {
                "prefix_len": length,
                "reasoning_prefix_tokens": length,
                "forced_close": bool(trace.extra.get("forced_close", False)),
                "forced_close_sequence": trace.extra.get("forced_close_sequence"),
                "final_token_ids": trace.token_ids,
                "final_text": trace.text,
                "answer_raw": answer_raw,
                "answer_normalized": answer_norm,
                "correct": bool(answer_raw is not None and score(family, answer_raw, answer)),
                "generated_tokens": trace.generated_tokens,
                "finish_reason": trace.finish_reason,
                "terminated_by_eos": trace.terminated_by_eos,
                "eos_observed": bool(eos_ids) and any(t in eos_ids for t in trace.token_ids),
                "think_end_reached": trace.think_end_reached,
                "error": trace.error,
            }
        )
    return results
