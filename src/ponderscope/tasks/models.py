"""Task records for the procedural task pack."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class TaskError(ValueError):
    """Raised when a generated task fails structural validation."""


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    pack: str
    version: str
    split: str
    seed: int
    index: int
    variant: str | None
    prompt: str
    answer: str  # canonical normalized answer
    prompt_policy: str = "pp-v1"
    difficulty: dict[str, Any] = field(default_factory=dict)
    structural: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        return cls(**data)


def validate_task(task: Task) -> None:
    """Invariants every generated task must satisfy.

    The prompt may contain the answer as incidental problem data (a state name
    in a transition table, an operand, an edge weight). What must never happen
    is the prompt explicitly stating the answer at an answer cue.
    """
    import re

    if not task.prompt or not task.prompt.strip():
        raise TaskError(f"{task.task_id}: empty prompt")
    if not task.answer or not task.answer.strip():
        raise TaskError(f"{task.task_id}: empty answer")
    answer = task.answer.strip()
    embedded = re.search(
        r"(?:answer\s*[:=]\s*|=\s*)" + re.escape(answer) + r"(?![A-Za-z0-9])",
        task.prompt,
        flags=re.IGNORECASE,
    )
    if embedded:
        raise TaskError(f"{task.task_id}: answer {answer!r} stated at an answer cue")
