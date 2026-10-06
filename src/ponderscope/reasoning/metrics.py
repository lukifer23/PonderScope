"""Reasoning trajectory metrics computed from generated token ids and text.

All metrics are deterministic functions of the trace. No external models are
used. Repetition here is token-level and textual; semantic similarity is not
attempted in this pass.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

_SPECIAL_PIECE = re.compile(r"<\|[^|]*\|>| thinking|</think>")


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
class LoopDiagnostics:
    """Descriptive loop/degeneration structure of one generation.

    These are *descriptive* diagnostics with an explicitly documented onset
    rule, not a validated universal loop detector.
    """

    longest_run_length: int
    longest_run_token_id: int | None
    longest_run_token_piece: str | None
    longest_run_is_special: bool
    longest_run_start: int | None
    longest_run_end: int | None
    top_motif_n: int
    top_motif: list[int]
    top_motif_count: int
    top_motif_span_tokens: int
    top_motif_piece: str | None
    rolling_window: int
    rolling_unique_ratio_min: float | None
    degeneration_onset_index: int | None
    onset_rule: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _longest_run(seq: list[int]) -> tuple[int, int | None, int | None, int | None]:
    best_len, best_tok, best_start, best_end = 0, None, None, None
    i = 0
    while i < len(seq):
        j = i
        while j + 1 < len(seq) and seq[j + 1] == seq[i]:
            j += 1
        length = j - i + 1
        if length > best_len:
            best_len, best_tok, best_start, best_end = length, seq[i], i, j
        i = j + 1
    return best_len, best_tok, best_start, best_end


def _top_motif(seq: list[int], n: int) -> tuple[list[int], int]:
    if len(seq) < n or n <= 0:
        return [], 0
    counts = Counter(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))
    gram, count = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    return list(gram), count


def _rolling_unique_ratio_min(seq: list[int], window: int) -> float | None:
    if not seq:
        return None
    if len(seq) < window:
        return len(set(seq)) / len(seq)
    return min(len(set(seq[i : i + window])) / window for i in range(len(seq) - window + 1))


def _degeneration_onset(seq: list[int], window: int, threshold: float) -> int | None:
    """Earliest window start from which repeated-4-gram fraction never drops below
    ``threshold`` again. Purely descriptive; ``None`` when no such suffix exists.
    """
    if len(seq) < window:
        return None
    reps = [_repeated_fraction(seq[i : i + window], 4) for i in range(len(seq) - window + 1)]
    onset: int | None = None
    for i in range(len(reps) - 1, -1, -1):
        if reps[i] >= threshold:
            onset = i
        else:
            break
    return onset


def compute_loop_diagnostics(
    token_ids: list[int],
    *,
    decode: Callable[[list[int]], str] | None = None,
    special_token_ids: set[int] | None = None,
    motif_n: int = 4,
    window: int = 64,
    threshold: float = 0.5,
) -> LoopDiagnostics:
    """Compute descriptive loop structure from a generated token sequence."""
    special_token_ids = special_token_ids or set()
    run_len, run_tok, run_start, run_end = _longest_run(token_ids)

    piece: str | None = None
    is_special = False
    if run_tok is not None:
        if decode is not None:
            piece = decode([run_tok])
            is_special = bool(_SPECIAL_PIECE.search(piece))
        is_special = is_special or run_tok in special_token_ids

    motif, motif_count = _top_motif(token_ids, motif_n)
    motif_piece = decode(motif) if (decode is not None and motif) else None

    return LoopDiagnostics(
        longest_run_length=run_len,
        longest_run_token_id=run_tok,
        longest_run_token_piece=piece,
        longest_run_is_special=is_special,
        longest_run_start=run_start,
        longest_run_end=run_end,
        top_motif_n=motif_n,
        top_motif=motif,
        top_motif_count=motif_count,
        top_motif_span_tokens=motif_count * motif_n,
        top_motif_piece=motif_piece,
        rolling_window=window,
        rolling_unique_ratio_min=_rolling_unique_ratio_min(token_ids, window),
        degeneration_onset_index=_degeneration_onset(token_ids, window, threshold),
        onset_rule=(
            f"earliest window start (window={window}) from which the repeated-4-gram "
            f"fraction stays >= {threshold} for every later window; descriptive only"
        ),
    )


@dataclass
class TraceMetrics:
    reasoning_tokens: int
    answer_tokens: int
    total_tokens: int
    termination: dict[str, Any] = field(default_factory=dict)
    repetition: RepetitionMetrics | None = None
    loop: LoopDiagnostics | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reasoning_tokens": self.reasoning_tokens,
            "answer_tokens": self.answer_tokens,
            "total_tokens": self.total_tokens,
            "termination": self.termination,
            "repetition": self.repetition.to_dict() if self.repetition else None,
            "loop": self.loop.to_dict() if self.loop else None,
        }
