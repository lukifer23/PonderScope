"""Forced-probe outcome taxonomy.

A forced-finalization probe is **not** a natural generation. The continuation it
produces is read as a forced final-answer channel, and a probe that runs out of
its own finalization budget without emitting EOS or a complete scorable answer
must **never** be recorded as an observed wrong answer. It is *censored*.

The taxonomy is deliberately small and exhaustive:

    correct            a scorable, normalized answer that exactly matches gold
    scorable_incorrect a scorable, normalized answer that does not match gold
    unparseable        a final channel was produced but no scorable answer could
                       be extracted from it (an unscorable string is never wrong)
    censored           the probe budget was reached with no scorable answer
    error              the backend returned an error

``answer_observed`` (raw final-channel text existed) and ``answer_scorable`` (a
normalized answer was extracted) are recorded separately, because an unscorable
string and an absent answer are different failure modes.
"""

from __future__ import annotations

from typing import Any

from ..tasks.scorers import normalize, score
from .parse import extract_answer

FORCED_OUTCOMES = (
    "correct",
    "scorable_incorrect",
    "unparseable",
    "censored",
    "error",
)

# Outcomes that constitute an *observed, scorable* answer. Transition logic may
# only draw a correctness conclusion from these; ``censored``/``unparseable``/
# ``error`` are indeterminate and must never be reinterpreted as "wrong".
OBSERVED_SCORABLE_OUTCOMES = ("correct", "scorable_incorrect")


def classify_forced_probe(
    family: str,
    gold_answer: str,
    forced_text: str | None,
    *,
    finish_reason: str | None,
    capped: bool,
    error: str | None = None,
    answer_raw: str | None = None,
    answer_normalized: str | None = None,
    eos_observed: bool = False,
    native_close_injected: bool = True,
    generated_answer_token_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Classify one forced-probe continuation into the outcome taxonomy.

    ``answer_raw``/``answer_normalized`` may be supplied by the caller if it has
    already parsed the continuation; otherwise they are extracted here. The
    outcome is derived solely from the continuation, never from the source
    trajectory.
    """
    if error:
        return _result(
            "error",
            family=family,
            gold_answer=gold_answer,
            forced_text=forced_text,
            answer_raw=answer_raw,
            answer_normalized=answer_normalized,
            finish_reason=finish_reason,
            capped=capped,
            error=error,
            eos_observed=eos_observed,
            native_close_injected=native_close_injected,
            generated_answer_token_ids=generated_answer_token_ids,
        )

    if answer_raw is None and forced_text:
        answer_raw = extract_answer(forced_text)
    if answer_normalized is None and answer_raw is not None:
        answer_normalized = normalize(family, answer_raw)

    if answer_normalized is not None:
        correct = score(family, answer_raw or "", gold_answer)
        outcome = "correct" if correct else "scorable_incorrect"
    elif answer_raw is not None:
        outcome = "unparseable"
    elif capped or finish_reason == "length":
        outcome = "censored"
    else:
        outcome = "unparseable"

    return _result(
        outcome,
        family=family,
        gold_answer=gold_answer,
        forced_text=forced_text,
        answer_raw=answer_raw,
        answer_normalized=answer_normalized,
        finish_reason=finish_reason,
        capped=capped,
        error=error,
        eos_observed=eos_observed,
        native_close_injected=native_close_injected,
        generated_answer_token_ids=generated_answer_token_ids,
    )


def _result(
    outcome: str,
    *,
    family: str,
    gold_answer: str,
    forced_text: str | None,
    answer_raw: str | None,
    answer_normalized: str | None,
    finish_reason: str | None,
    capped: bool,
    error: str | None,
    eos_observed: bool,
    native_close_injected: bool,
    generated_answer_token_ids: list[int] | None,
) -> dict[str, Any]:
    assert outcome in FORCED_OUTCOMES, outcome
    answer_observed = answer_raw is not None
    answer_scorable = answer_normalized is not None
    correct = bool(answer_scorable and score(family, answer_raw or "", gold_answer))
    return {
        "outcome": outcome,
        "answer_observed": answer_observed,
        "answer_scorable": answer_scorable,
        "answer_raw": answer_raw,
        "answer_normalized": answer_normalized,
        "correct": correct,
        "forced_answer_text": forced_text,
        "generated_answer_token_ids": list(generated_answer_token_ids or []),
        "finish_reason": finish_reason,
        "capped": bool(capped),
        "eos_observed": bool(eos_observed),
        "native_close_injected": bool(native_close_injected),
        "probe_error": error,
    }
