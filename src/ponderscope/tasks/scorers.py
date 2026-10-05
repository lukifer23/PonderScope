"""Exact scorers and answer normalization.

Scoring is mechanical. No LLM judge is used. Normalization extracts the
family's canonical answer form from a raw model-produced string; scoring then
compares canonical forms.
"""

from __future__ import annotations

import re

_INT_RE = re.compile(r"-?\d+")
_SM_RE = re.compile(r"\bs?(\d+)\b")
_BOOL_RE = re.compile(r"\b(true|false)\b", re.IGNORECASE)
_ORDER_RE = re.compile(r"[A-Za-z]+")
_ANSWER_CUE = re.compile(r"answer\s*[:=]", re.IGNORECASE)


def _answer_segment(raw: str) -> str:
    """Restrict normalization to the text after the final answer cue, if any."""
    matches = list(_ANSWER_CUE.finditer(raw))
    return raw[matches[-1].end() :] if matches else raw


def normalize(family: str, raw: str) -> str | None:
    """Return the canonical answer string, or None if none can be found."""
    if raw is None:
        return None
    raw = _answer_segment(raw)
    if family in ("arith", "path"):
        matches = _INT_RE.findall(raw)
        if not matches:
            return None
        return str(int(matches[-1]))
    if family == "order":
        words = [w.lower() for w in _ORDER_RE.findall(raw)]
        if not words:
            return None
        return ",".join(words)
    if family == "logic":
        values = [m.lower() for m in _BOOL_RE.findall(raw)]
        if not values:
            return None
        return ", ".join(values)
    if family == "sm":
        match = _SM_RE.search(raw)
        if not match:
            return None
        return f"s{int(match.group(1))}"
    raise ValueError(f"unknown family: {family!r}")


def canonical_answer(family: str, answer: str) -> str:
    """Canonicalize a ground-truth answer to match :func:`normalize` output."""
    a = answer.strip()
    if family == "order":
        return ",".join(w.lower() for w in re.split(r"[,\s]+", a) if w)
    if family == "logic":
        return ", ".join(x.strip().lower() for x in a.split(","))
    if family == "sm":
        match = _SM_RE.search(a)
        return f"s{int(match.group(1))}" if match else a.lower()
    return a


def score(family: str, raw_prediction: str, answer: str) -> bool:
    normalized = normalize(family, raw_prediction)
    return normalized is not None and normalized == canonical_answer(family, answer)
