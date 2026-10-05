"""Parse a generated trace into reasoning and final-answer channels.

The reasoning channel is the text the model produced inside its thinking block.
Because the chat template already opened the block in the prompt, the generated
text contains the reasoning followed by the model's closing delimiter. If the
closing delimiter was never emitted (capped / censored), there is no final
channel and this is recorded explicitly.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

THINK_END_STR = "</think>"
THINK_END_TOKEN_IDS = (248069,)
_ANSWER_CUE = re.compile(r"answer\s*[:=]", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedReasoning:
    reasoning: str
    final: str
    answer_raw: str | None
    closed: bool  # think-end delimiter observed

    def to_dict(self) -> dict:
        return asdict(self)


def split_channels(text: str, think_end_reached: bool) -> tuple[str, str, bool]:
    """Return (reasoning, final, closed)."""
    idx = text.rfind(THINK_END_STR)
    if idx != -1:
        return text[:idx], text[idx + len(THINK_END_STR) :], True
    return text, "", think_end_reached


def extract_answer(final_text: str) -> str | None:
    """Extract the raw answer segment after the final 'Answer:' cue.

    Returns the substring after the cue if present, else the last non-empty line.
    Canonicalization is the scorer's job, not this function's.
    """
    matches = list(_ANSWER_CUE.finditer(final_text))
    if matches:
        return final_text[matches[-1].end() :].strip()
    for line in reversed(final_text.splitlines()):
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
