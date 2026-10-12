"""Forced-finalization prefix probes.

This is a measurement primitive, not a claimed contribution: probing reasoning
prefixes already exists in the literature. PonderScope uses it to observe
answer-state transitions along a trajectory.

A probe replays a **reasoning-only** prefix (never including final-answer or EOS
tokens), appends the model's validated native closing sequence, and reads the
generated continuation. The continuation *is* the forced final-answer channel,
so it is parsed as a forced answer, never as a natural complete trajectory. No
hand-written answer cue is injected.

Each probe records an explicit :mod:`~ponderscope.reasoning.outcomes` outcome so
that a probe which exhausts its own finalization budget is classified
``censored`` and is never mistaken for an observed wrong answer.
"""

from __future__ import annotations

from typing import Any

from ..backends.base import Backend, CaptureSpec
from ..config.identity import DecodingPolicy
from ..tasks.scorers import normalize
from .outcomes import classify_forced_probe
from .parse import extract_answer
from .transitions import prefix_lengths


def probe_result(
    trace: Any,
    *,
    family: str,
    answer: str,
    length: int,
    eos_token_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Classify one ``backend.probe`` trace into a rich probe record."""
    eos_ids = eos_token_ids or set()
    answer_raw = extract_answer(trace.text)
    answer_norm = normalize(family, answer_raw) if answer_raw is not None else None
    eos_observed = bool(eos_ids) and any(t in eos_ids for t in trace.token_ids)
    classified = classify_forced_probe(
        family,
        answer,
        trace.text,
        finish_reason=trace.finish_reason,
        capped=trace.capped,
        error=trace.error,
        answer_raw=answer_raw,
        answer_normalized=answer_norm,
        eos_observed=eos_observed,
        native_close_injected=bool(trace.extra.get("forced_close", False)),
        generated_answer_token_ids=trace.token_ids,
    )
    return {
        "prefix_len": length,
        "reasoning_prefix_tokens": length,
        "forced_close": bool(trace.extra.get("forced_close", False)),
        "forced_close_sequence": trace.extra.get("forced_close_sequence"),
        "final_token_ids": trace.token_ids,
        "final_text": trace.text,
        "generated_tokens": trace.generated_tokens,
        "finish_reason": trace.finish_reason,
        "terminated_by_eos": trace.terminated_by_eos,
        "think_end_reached": trace.think_end_reached,
        "error": trace.error,
        # explicit outcome taxonomy (Phase 1.8)
        "outcome": classified["outcome"],
        "pre_censor_outcome": classified["pre_censor_outcome"],
        "censored_with_extractable_answer": classified["censored_with_extractable_answer"],
        "answer_observed": classified["answer_observed"],
        "answer_scorable": classified["answer_scorable"],
        "native_close_injected": classified["native_close_injected"],
        "eos_observed": classified["eos_observed"],
        "generated_answer_token_ids": classified["generated_answer_token_ids"],
        "forced_answer_text": classified["forced_answer_text"],
        # backward-compatible keys
        "answer_raw": answer_raw,
        "answer_normalized": answer_norm,
        "correct": bool(classified["correct"]),
    }


def run_prefix_probes_at(
    backend: Backend,
    *,
    family: str,
    answer: str,
    prompt_token_ids: list[int],
    prefix_token_ids: list[int],
    lengths: list[int],
    probe_decoding: DecodingPolicy,
    capture: CaptureSpec,
    eos_token_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Run forced-finalization probes at *exact* prefix token lengths.

    Every requested length must satisfy ``0 <= length <= len(prefix_token_ids)``;
    a caller must never substitute a shorter prefix for a requested longer one.
    """
    results: list[dict[str, Any]] = []
    for length in lengths:
        if not 0 <= length <= len(prefix_token_ids):
            raise ValueError(
                f"requested prefix length {length} exceeds reasoning length {len(prefix_token_ids)}"
            )
        prefix = prefix_token_ids[:length]
        trace = backend.probe(prompt_token_ids, prefix, probe_decoding, capture)
        results.append(
            probe_result(
                trace, family=family, answer=answer, length=length, eos_token_ids=eos_token_ids
            )
        )
    return results


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
    """Run forced-finalization probes over a grid of reasoning prefixes.

    ``prefix_token_ids`` must contain reasoning tokens only; the caller is
    responsible for stripping the final channel. Raises
    ``PrefixProbeUnsupported`` (propagated from the backend) if the native
    closure cannot be validated. ``checkpoints`` selects explicit prefix token
    lengths (bounded by the actual reasoning length); otherwise a uniform grid is
    used.
    """
    lengths = prefix_lengths(len(prefix_token_ids), n_probes, checkpoints=checkpoints)
    return run_prefix_probes_at(
        backend,
        family=family,
        answer=answer,
        prompt_token_ids=prompt_token_ids,
        prefix_token_ids=prefix_token_ids,
        lengths=lengths,
        probe_decoding=probe_decoding,
        capture=capture,
        eos_token_ids=eos_token_ids,
    )
