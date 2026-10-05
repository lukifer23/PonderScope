"""Parse a generated trace into reasoning and final-answer channels.

Channels are delimited at **token boundaries** where possible: the reasoning
channel is the token ids before the model's native think-end token and the final
channel is everything after it. Because ``tokenizer.decode`` renders special
tokens literally (e.g. ``<|im_end|>``), we keep three distinct representations
and never collapse them:

- the raw decoded text (special tokens included) for evidence;
- the token-level channel boundaries;
- a sanitized semantic final channel used for scoring.

If the closing delimiter was never emitted (capped / censored), there is no
final channel and this is recorded explicitly.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

THINK_END_STR = "</think>"
_ANSWER_CUE = re.compile(r"answer\s*[:=]", re.IGNORECASE)
_SPECIAL = re.compile(r"<\|[^|]*\|>|<think>|</think>")


def strip_special_tokens(text: str) -> str:
    """Remove decoded special-token markers so they cannot reach a scorer."""
    return _SPECIAL.sub("", text)


def reasoning_prefix_ids(token_ids: list[int], think_end_id: int | None) -> list[int]:
    """Token ids strictly before the first validated think-end boundary.

    Anything at or after the native think-end token belongs to the final-answer
    channel (or EOS) and must never enter a probe prefix. If the token is absent,
    the whole generation is (censored) reasoning.
    """
    if think_end_id is not None and think_end_id in token_ids:
        return list(token_ids[: token_ids.index(think_end_id)])
    return list(token_ids)


@dataclass(frozen=True)
class ParsedReasoning:
    reasoning: str
    final: str
    answer_raw: str | None
    closed: bool  # think-end delimiter observed

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ParsedTrace:
    """Full three-way channel split of one generation."""

    reasoning_token_ids: list[int]
    final_token_ids: list[int]
    token_closed: bool
    reasoning: str
    final_raw: str
    final_semantic: str
    answer_raw: str | None
    closed: bool
    eos_observed: bool
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_channels(text: str, think_end_reached: bool) -> tuple[str, str, bool]:
    """Return (reasoning, final, closed) from decoded text."""
    idx = text.rfind(THINK_END_STR)
    if idx != -1:
        return text[:idx], text[idx + len(THINK_END_STR) :], True
    return text, "", think_end_reached


def extract_answer(final_text: str) -> str | None:
    """Extract the raw answer segment after the final 'Answer:' cue.

    Special-token markers are stripped first so an EOS string such as
    ``<|im_end|>`` can never be read as answer content. Returns the substring
    after the cue if present, else the last non-empty line.
    """
    cleaned = strip_special_tokens(final_text)
    matches = list(_ANSWER_CUE.finditer(cleaned))
    if matches:
        return cleaned[matches[-1].end() :].strip()
    for line in reversed(cleaned.splitlines()):
        if line.strip():
            return line.strip()
    return None


def parse_reasoning(text: str, think_end_reached: bool) -> ParsedReasoning:
    reasoning, final, closed = split_channels(text, think_end_reached)
    answer_raw = extract_answer(final) if final.strip() else None
    return ParsedReasoning(
        reasoning=reasoning,
        final=final,
        answer_raw=answer_raw,
        closed=closed,
    )


def parse_trace(
    text: str,
    token_ids: list[int],
    *,
    think_end_id: int | None,
    eos_ids: set[int] | frozenset[int] = frozenset(),
) -> ParsedTrace:
    """Token-boundary-aware channel split of a full generation.

    ``token_ids`` are the generated tokens only. The final channel cannot
    contain any reasoning token, and a decoded EOS marker can never be mistaken
    for answer content because the semantic channel is sanitized.
    """
    token_closed = think_end_id is not None and think_end_id in token_ids
    if token_closed:
        idx = token_ids.index(think_end_id)  # type: ignore[arg-type]
        reasoning_ids = list(token_ids[:idx])
        final_ids = list(token_ids[idx + 1 :])
    else:
        reasoning_ids = list(token_ids)
        final_ids = []

    str_idx = text.find(THINK_END_STR)
    if str_idx != -1:
        reasoning_text = text[:str_idx]
        final_raw = text[str_idx + len(THINK_END_STR) :]
    else:
        reasoning_text = text
        final_raw = ""

    final_semantic = strip_special_tokens(final_raw).strip()
    answer_raw = extract_answer(final_semantic) if final_semantic else None
    return ParsedTrace(
        reasoning_token_ids=reasoning_ids,
        final_token_ids=final_ids,
        token_closed=token_closed,
        reasoning=reasoning_text,
        final_raw=final_raw,
        final_semantic=final_semantic,
        answer_raw=answer_raw,
        closed=token_closed or str_idx != -1,
        eos_observed=bool(eos_ids) and any(t in eos_ids for t in token_ids),
        raw_text=text,
    )
