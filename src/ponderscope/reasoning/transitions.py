"""Answer-transition classification for reasoning trajectories.

Given correctness at a sequence of reasoning prefixes (from forced-finalization
probes) and the natural final correctness, classify the trajectory. Labels are
accompanied by independent boolean flags and by *both* an array index and an
actual prefix-token count, so trajectory-category labels are never conflated
with sufficiency-point measurement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

STATES = (
    "initially_correct",
    "wrong_to_correct",
    "correct_to_wrong",
    "multiple_flips",
    "stable_correct",
    "never_correct",
)


@dataclass
class TrajectoryState:
    primary: str
    flips: int
    first_correct_probe_index: int | None
    first_correct_prefix_tokens: int | None
    stable_sufficient_prefix_tokens: int | None
    final_correct: bool
    flags: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def answer_flips(answers: list[str | None]) -> int:
    """Number of changes between consecutive non-None answers."""
    seq = [a for a in answers if a is not None]
    return sum(1 for a, b in zip(seq, seq[1:], strict=False) if a != b)


def stable_sufficient_index(prefix_correct: list[bool], final_correct: bool) -> int | None:
    """Earliest tested prefix that is correct and stays correct through the end.

    Requires the natural final answer to be correct as well, so a probe-that-looks
    correct but whose trajectory ends wrong is not counted as sufficient.
    """
    if not final_correct:
        return None
    for i in range(len(prefix_correct)):
        if prefix_correct[i] and all(prefix_correct[i:]):
            return i
    return None


def classify_transitions(
    final_correct: bool,
    prefix_correct: list[bool],
    prefix_token_lengths: list[int] | None = None,
) -> TrajectoryState:
    full = list(prefix_correct) + [final_correct]
    flips = sum(1 for a, b in zip(full, full[1:], strict=False) if a != b)
    first_idx = next((i for i, c in enumerate(prefix_correct) if c), None)
    suff_idx = stable_sufficient_index(prefix_correct, final_correct)

    def tokens(idx: int | None) -> int | None:
        if idx is None:
            return None
        if prefix_token_lengths is not None and idx < len(prefix_token_lengths):
            return prefix_token_lengths[idx]
        return idx

    flags = dict.fromkeys(STATES, False)
    flags["never_correct"] = not any(full)
    flags["stable_correct"] = all(full)
    if prefix_correct:
        flags["initially_correct"] = bool(prefix_correct[0])
    if flips == 1:
        flags["wrong_to_correct"] = not full[0] and full[-1]
        flags["correct_to_wrong"] = full[0] and not full[-1]
    if flips >= 2:
        flags["multiple_flips"] = True

    if flags["multiple_flips"]:
        primary = "multiple_flips"
    elif flags["never_correct"]:
        primary = "never_correct"
    elif flags["stable_correct"]:
        primary = "stable_correct"
    elif flags["wrong_to_correct"]:
        primary = "wrong_to_correct"
    elif flags["correct_to_wrong"]:
        primary = "correct_to_wrong"
    elif flags["initially_correct"]:
        primary = "initially_correct"
    else:
        primary = "never_correct"

    return TrajectoryState(
        primary=primary,
        flips=flips,
        first_correct_probe_index=first_idx,
        first_correct_prefix_tokens=tokens(first_idx),
        stable_sufficient_prefix_tokens=tokens(suff_idx),
        final_correct=final_correct,
        flags=flags,
    )


def prefix_lengths(n_reasoning: int, n_probes: int, min_prefix: int = 16) -> list[int]:
    """Prefix token counts for forced-finalization probing.

    Includes the empty prefix (0), which measures the answer with no reasoning
    at all. Locations are bounded by the actual reasoning length, so no probe can
    contain a final-answer token.
    """
    if n_probes <= 0:
        return []
    max_prefix = max(0, n_reasoning)
    lengths = [0]
    if n_reasoning > min_prefix:
        import numpy as np

        grid = np.linspace(min_prefix, max_prefix, num=n_probes)
        for value in grid:
            length = int(round(float(value)))
            if length not in lengths:
                lengths.append(length)
    return sorted(lengths)
