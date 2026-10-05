"""Reasoning trajectory metrics computed from generated token ids and text.

All metrics are deterministic functions of the trace. No external models are
used. Repetition here is token-level and textual; semantic similarity is not
attempted in this pass.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RepetitionMetrics:
    n_tokens: int
    unique_tokens: int
    unique_token_ratio: float
    distinct_1: float
    distinct_2: float
    distinct_4: float
    max_token_run: int
    most_common_token_count: int
    repeated_ngram_fraction_4: float
    repeated_ngram_fraction_8: float
    text_char_len: int
    text_repeat_ratio: float  # 1 - unique_lines/total_lines

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ngrams(seq: list[int], n: int) -> list[tuple[int, ...]]:
    if len(seq) < n or n <= 0:
        return []
    return [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]


def _distinct_ratio(seq: list[int], n: int) -> float:
    grams = _ngrams(seq, n)
    if not grams:
        return 1.0
    return len(set(grams)) / len(grams)


def _repeated_fraction(seq: list[int], n: int) -> float:
    grams = _ngrams(seq, n)
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(grams)


def _max_run(seq: list[int]) -> int:
    if not seq:
        return 0
    best = 1
    cur = 1
    for prev, cur_tok in zip(seq, seq[1:], strict=False):
        if prev == cur_tok:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def compute_repetition(token_ids: list[int], text: str) -> RepetitionMetrics:
    n = len(token_ids)
    counts = Counter(token_ids)
    most_common = max(counts.values()) if counts else 0
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    line_counts = Counter(lines)
    text_repeat = 0.0
    if lines:
        text_repeat = 1.0 - (len(line_counts) / len(lines))
    return RepetitionMetrics(
        n_tokens=n,
        unique_tokens=len(counts),
        unique_token_ratio=(len(counts) / n) if n else 1.0,
        distinct_1=_distinct_ratio(token_ids, 1),
        distinct_2=_distinct_ratio(token_ids, 2),
        distinct_4=_distinct_ratio(token_ids, 4),
        max_token_run=_max_run(token_ids),
        most_common_token_count=most_common,
        repeated_ngram_fraction_4=_repeated_fraction(token_ids, 4),
        repeated_ngram_fraction_8=_repeated_fraction(token_ids, 8),
        text_char_len=len(text),
        text_repeat_ratio=text_repeat,
    )


@dataclass
class TraceMetrics:
    reasoning_tokens: int
    answer_tokens: int
    total_tokens: int
    termination: dict[str, Any] = field(default_factory=dict)
    repetition: RepetitionMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reasoning_tokens": self.reasoning_tokens,
            "answer_tokens": self.answer_tokens,
            "total_tokens": self.total_tokens,
            "termination": self.termination,
            "repetition": self.repetition.to_dict() if self.repetition else None,
        }
