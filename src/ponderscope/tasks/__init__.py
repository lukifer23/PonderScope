"""Versioned procedural task pack."""

from .families import (
    ALL_FAMILIES,
    DEFAULT_PROMPT_POLICY,
    DIFFICULTY_LEVELS,
    PROMPT_POLICIES,
    make_family_task,
)
from .generator import (
    GENERATOR_VERSION,
    PACK_REGISTRY,
    PACK_VERSION,
    SPLITS,
    PackSpec,
    generate_pack,
    generate_pack_metadata,
    get_pack,
    make_task,
    task_id_for,
)
from .invariants import (
    collision_audit,
    signature_hash,
    structural_signature,
    verify_invariants,
)
from .models import Task, TaskError, validate_task
from .scorers import canonical_answer, normalize, score

__all__ = [
    "ALL_FAMILIES",
    "DEFAULT_PROMPT_POLICY",
    "DIFFICULTY_LEVELS",
    "GENERATOR_VERSION",
    "PACK_REGISTRY",
    "PACK_VERSION",
    "PROMPT_POLICIES",
    "SPLITS",
    "PackSpec",
    "Task",
    "TaskError",
    "canonical_answer",
    "collision_audit",
    "generate_pack",
    "generate_pack_metadata",
    "get_pack",
    "make_family_task",
    "make_task",
    "normalize",
    "score",
    "signature_hash",
    "structural_signature",
    "task_id_for",
    "validate_task",
    "verify_invariants",
]
