"""Answer-transition classification for reasoning trajectories.

The classifier separates three things that must never be conflated:

1. **prefix states** — correctness at forced-finalization probes, computed only
   over *observed* probes;
2. **natural final status** — what actually happened at the end of the natural
   trajectory: ``correct``, ``incorrect``, ``censored`` (no final-answer channel
   observed), ``unparseable`` (a final channel was observed but no answer could
   be extracted), or ``error``;
3. **sufficiency points** — the earliest measured reasoning prefix that stayed
   correct, reported both with and without the requirement that the natural
   final answer was actually observed.

A capped / censored trajectory has **no observed final answer**, so it can never
create a correctness transition and can never be evidence of harmful
overthinking. This is the central correctness fix of Phase 1.2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Natural final outcome categories. ``censored`` means the final answer channel
# was never observed because the observation horizon was reached; correctness is
# unknown, NOT false. ``terminated_no_closure`` means generation stopped (EOS)
# without ever emitting the native reasoning-close token: there was no final
# channel and the end was not the observation horizon. ``unparseable`` means a
# final-answer channel WAS observed (or the stop cause was neither horizon nor
# EOS) but no answer could be extracted from it.
NATURAL_FINAL_STATUSES = (
    "correct",
    "incorrect",
    "censored",
    "terminated_no_closure",
    "unparseable",
    "error",
)

# Prefix states are computed only over observed forced-probe results.
PREFIX_STATES = (
    "initially_correct",
    "wrong_to_correct",
    "correct_to_wrong",
    "multiple_flips",
    "stable_correct",
    "never_correct",
)

# Backwards-compatible alias for the set of state labels.
STATES = PREFIX_STATES


def natural_final_status_from_termination(
    *,
    correct: bool,
    answer_scorable: bool,
    think_end_reached: bool,
    capped: bool,
    finish_reason: str | None = None,
    error_type: str | None = None,
    raw_answer_present: bool = False,
) -> str:
    """Classify the natural final outcome.

    ``answer_scorable`` is True only when a **normalized/scorable** answer was
    observed. A raw final-channel string that fails normalization is *not* a
    scorable answer and must be classified ``unparseable``, never ``incorrect``
    (an unscorable string is not evidence of a wrong answer). ``raw_answer_present``
    records that some final-channel text existed even if it did not normalize.
    """
    if error_type or finish_reason == "error":
        return "error"
    if answer_scorable:
        return "correct" if correct else "incorrect"
    # No scorable answer was observed.
    if raw_answer_present:
        return "unparseable"
    if capped or finish_reason == "length":
        # Capped inside reasoning vs capped after the native close are distinct.
        return "unparseable" if think_end_reached else "censored"
    if finish_reason == "stop" and not think_end_reached:
        return "terminated_no_closure"
    return "unparseable"


def budget_exhausted_after_closure(record: dict[str, Any]) -> bool:
    """True when the native reasoning channel closed but EOS was never emitted."""
    term = record.get("termination", {})
    return bool(term.get("think_end_reached")) and bool(term.get("capped"))


def natural_final_status_for_record(record: dict[str, Any]) -> str:
    """Derive the natural final status from a saved generation record."""
    term = record.get("termination", {})
    trace = record.get("trace") or {}
    # Scorable presence is ``answer_normalized`` for records written from Phase
    # 1.7 onward; legacy records fall back to the older raw-answer flag.
    answer_scorable = record.get("answer_normalized") is not None
    raw_present = not bool(term.get("missing_answer", True))
    return natural_final_status_from_termination(
        correct=bool(record.get("correct")),
        answer_scorable=answer_scorable,
        think_end_reached=bool(term.get("think_end_reached", False)),
        capped=bool(term.get("capped", False)),
        finish_reason=term.get("finish_reason"),
        error_type=term.get("error_type") or trace.get("error"),
        raw_answer_present=raw_present,
    )


def natural_final_correctness(status: str) -> bool | None:
    """Map a natural final status to a tri-state correctness value."""
    if status == "correct":
        return True
    if status == "incorrect":
        return False
    return None


@dataclass
class TrajectoryState:
    """Censoring-aware trajectory classification.

    - ``prefix_state`` / ``prefix_flips`` describe observed forced probes only.
    - ``natural_final_status`` and ``natural_final_correct`` describe the natural
      trajectory (``natural_final_correct`` is ``None`` whenever unobserved).
    - ``observed_probe_stable_from_tokens`` is the earliest probe that is correct
      and stays correct through every later observed probe; it makes no claim
      about the natural final.
    - ``stable_sufficient_with_natural_final_tokens`` additionally requires the
      natural final to have been observed correct.
    - ``harmful_overthinking_observed`` can only be True when an *observed*
      natural final is wrong after an earlier correct probe.
    """

    prefix_state: str
    prefix_flips: int
    natural_final_status: str
    natural_final_correct: bool | None
    first_correct_probe_index: int | None
    first_correct_prefix_tokens: int | None
    observed_probe_stable_from_tokens: int | None
    stable_sufficient_with_natural_final_tokens: int | None
    harmful_overthinking_observed: bool
    flags: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _prefix_flips(prefix_correct: list[bool]) -> int:
    return sum(1 for a, b in zip(prefix_correct, prefix_correct[1:], strict=False) if a != b)


def _prefix_state(prefix_correct: list[bool]) -> str:
    if not prefix_correct:
        return "never_correct"
    flips = _prefix_flips(prefix_correct)
    if flips >= 2:
        return "multiple_flips"
    if all(prefix_correct):
        return "stable_correct"
    if not any(prefix_correct):
        return "never_correct"
    if flips == 1:
        return "wrong_to_correct" if prefix_correct[-1] else "correct_to_wrong"
    return "never_correct"


def _observed_stable_index(prefix_correct: list[bool]) -> int | None:
    """Earliest index correct from there through the last observed probe."""
    for i in range(len(prefix_correct)):
        if prefix_correct[i] and all(prefix_correct[i:]):
            return i
    return None


def classify_transitions(
    prefix_correct: list[bool],
    natural_final_status: str,
    prefix_token_lengths: list[int] | None = None,
) -> TrajectoryState:
    """Classify a trajectory from observed probes plus a natural final status.

    ``natural_final_status`` must be one of :data:`NATURAL_FINAL_STATUSES`. A
    censored/unparseable/error final has unknown correctness and therefore
    contributes no flip and no harmful-overthinking claim.
    """
    if natural_final_status not in NATURAL_FINAL_STATUSES:
        raise ValueError(f"unknown natural_final_status: {natural_final_status!r}")
    observed = [bool(c) for c in prefix_correct]
    prefix_flips = _prefix_flips(observed)
    prefix_state = _prefix_state(observed)
    final_correct = natural_final_correctness(natural_final_status)

    first_idx = next((i for i, c in enumerate(observed) if c), None)
    observed_stable_idx = _observed_stable_index(observed)
    natural_stable_idx = observed_stable_idx if final_correct is True else None
    harmful = final_correct is False and any(observed)

    def tokens(idx: int | None) -> int | None:
        if idx is None:
            return None
        if prefix_token_lengths is not None and idx < len(prefix_token_lengths):
            return prefix_token_lengths[idx]
        return idx

    flags = dict.fromkeys(PREFIX_STATES, False)
    flags[prefix_state] = True
    flags["never_correct"] = prefix_state == "never_correct"
    flags["stable_correct"] = prefix_state == "stable_correct"
    flags["natural_final_observed"] = final_correct is not None
    flags["harmful_overthinking_observed"] = harmful

    return TrajectoryState(
        prefix_state=prefix_state,
        prefix_flips=prefix_flips,
        natural_final_status=natural_final_status,
        natural_final_correct=final_correct,
        first_correct_probe_index=first_idx,
        first_correct_prefix_tokens=tokens(first_idx),
        observed_probe_stable_from_tokens=tokens(observed_stable_idx),
        stable_sufficient_with_natural_final_tokens=tokens(natural_stable_idx),
        harmful_overthinking_observed=harmful,
        flags=flags,
    )


def prefix_lengths(
    n_reasoning: int,
    n_probes: int,
    min_prefix: int = 16,
    checkpoints: list[int] | None = None,
) -> list[int]:
    """Prefix token counts for forced-finalization probing.

    Includes the empty prefix (0), which measures the answer with no reasoning
    at all. Locations are bounded by the actual reasoning length, so no probe can
    contain a final-answer token.

    When explicit ``checkpoints`` are given (e.g. 256/512/1024/1536), only those
    that fit within the reasoning length are used, plus the empty prefix and the
    full reasoning prefix; a trajectory shorter than a checkpoint simply does not
    receive that checkpoint. Otherwise a uniform grid of ``n_probes`` locations
    is used.
    """
    max_prefix = max(0, n_reasoning)
    if checkpoints is not None:
        checkpoint_lengths: set[int] = {0, max_prefix}
        for checkpoint in checkpoints:
            if 0 <= int(checkpoint) <= max_prefix:
                checkpoint_lengths.add(int(checkpoint))
        return sorted(checkpoint_lengths)
    if n_probes <= 0:
        return []
    lengths = [0]
    if n_reasoning > min_prefix:
        import numpy as np

        grid = np.linspace(min_prefix, max_prefix, num=n_probes)
        for value in grid:
            length = int(round(float(value)))
            if length not in lengths:
                lengths.append(length)
    return sorted(lengths)
