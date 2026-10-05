"""Deterministic, versioned procedural task generation with an explicit registry.

Task identity depends only on the declared generation fields, never on the
rendered wording. Regenerating the same fields always yields byte-identical
tasks, and changing a prompt policy does not change a structural task id.

A pack version is looked up in an explicit registry and fails closed if unknown;
it can never silently reuse another pack's generator semantics. ``tasks-v1`` is
frozen and reproducible forever.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

from .families import (
    ALL_FAMILIES,
    DEFAULT_PROMPT_POLICY,
    DIFFICULTY_LEVELS,
    PROMPT_POLICIES,
    make_family_task,
)
from .models import Task, validate_task

PACK_VERSION = "tasks-v1"
GENERATOR_VERSION = "generator-v1"
SPLITS = ("train", "calibration", "dev", "test")

_SPLIT_OFFSET = {"train": 0, "calibration": 1_000_003, "dev": 2_000_003, "test": 3_000_003}


@dataclass(frozen=True)
class PackSpec:
    """Immutable contract for one task-pack version."""

    pack_version: str
    generator_version: str
    default_prompt_policy: str
    structural_params: dict[str, Any] = field(default_factory=dict)
    provenance: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_version": self.pack_version,
            "generator_version": self.generator_version,
            "default_prompt_policy": self.default_prompt_policy,
            "structural_params": self.structural_params,
            "provenance": self.provenance,
        }


# Explicit registry. Unknown packs fail closed rather than silently running.
PACK_REGISTRY: dict[str, PackSpec] = {
    "tasks-v1": PackSpec(
        pack_version="tasks-v1",
        generator_version="generator-v1",
        default_prompt_policy=DEFAULT_PROMPT_POLICY,
        structural_params={
            "families": list(ALL_FAMILIES),
            "difficulty_levels": list(DIFFICULTY_LEVELS),
            "splits": list(SPLITS),
        },
        provenance="Phase 1.0 initial procedural pack; frozen after first live evidence.",
    ),
}


def get_pack(pack: str) -> PackSpec:
    if pack not in PACK_REGISTRY:
        raise ValueError(
            f"unknown task pack: {pack!r} (known: {sorted(PACK_REGISTRY)}); refusing to "
            "silently reuse another pack's generator semantics"
        )
    return PACK_REGISTRY[pack]


def task_id_for(
    pack: str,
    family: str,
    split: str,
    seed: int,
    index: int,
    variant: str | None,
    generator_version: str | None = None,
) -> str:
    """Wording-independent, stable structural task id."""
    gen = generator_version or get_pack(pack).generator_version
    key = "|".join([pack, gen, family, split, str(seed), str(index), variant or ""])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def make_task(
    family: str,
    index: int,
    *,
    pack: str = PACK_VERSION,
    split: str = "dev",
    seed: int = 0,
    variant: str | None = None,
    difficulty: str | None = None,
    prompt_policy: str | None = None,
) -> Task:
    pack_spec = get_pack(pack)
    policy = prompt_policy or pack_spec.default_prompt_policy
    if policy not in PROMPT_POLICIES:
        raise ValueError(f"unknown prompt policy: {policy!r}")
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split!r}")
    task_seed = seed + _SPLIT_OFFSET[split] + index
    rng = random.Random(f"{pack}|{pack_spec.generator_version}|{family}|{task_seed}|{variant}")
    diff = difficulty or DIFFICULTY_LEVELS[index % len(DIFFICULTY_LEVELS)]
    data = make_family_task(family, rng, diff, policy)
    task = Task(
        task_id=task_id_for(pack, family, split, seed, index, variant, pack_spec.generator_version),
        family=family,
        pack=pack,
        version=pack_spec.pack_version,
        split=split,
        seed=seed,
        index=index,
        variant=variant,
        prompt=data["prompt"],
        answer=data["answer"],
        prompt_policy=policy,
        difficulty={**data["difficulty"], "level": diff},
        structural=data["structural"],
    )
    validate_task(task)
    from .invariants import verify_invariants

    verify_invariants(task)
    return task


def generate_pack(
    families: list[str] | None = None,
    n_per_family: int = 4,
    *,
    pack: str = PACK_VERSION,
    split: str = "dev",
    seed: int = 0,
    variant: str | None = None,
    prompt_policy: str | None = None,
) -> list[Task]:
    get_pack(pack)  # fail closed before doing any work
    families = list(families) if families else list(ALL_FAMILIES)
    for family in families:
        if family not in ALL_FAMILIES:
            raise ValueError(f"unknown family: {family!r}")
    tasks: list[Task] = []
    for family in families:
        for index in range(n_per_family):
            tasks.append(
                make_task(
                    family,
                    index,
                    pack=pack,
                    split=split,
                    seed=seed,
                    variant=variant,
                    prompt_policy=prompt_policy,
                )
            )
    # deterministic ordering
    tasks.sort(key=lambda t: (t.family, t.index))
    return tasks


def generate_pack_metadata(
    tasks: list[Task],
    *,
    families: list[str] | None,
    n_per_family: int,
    split: str,
    seed: int,
    pack: str,
    prompt_policy: str | None = None,
) -> dict[str, Any]:
    """Provenance of exactly what was generated (not merely requested)."""
    pack_spec = get_pack(pack)
    task_ids = sorted(t.task_id for t in tasks)
    digest = hashlib.sha256("\n".join(task_ids).encode("utf-8")).hexdigest()
    actual_families = sorted({t.family for t in tasks})
    actual_policies = sorted({t.prompt_policy for t in tasks})
    return {
        "pack_version": pack_spec.pack_version,
        "generator_version": pack_spec.generator_version,
        "prompt_policy": prompt_policy or pack_spec.default_prompt_policy,
        "prompt_policies_present": actual_policies,
        "requested_pack": pack,
        "structural_params": pack_spec.structural_params,
        "provenance": pack_spec.provenance,
        "families": actual_families,
        "n_per_family": n_per_family,
        "split": split,
        "seed": seed,
        "n_tasks": len(tasks),
        "task_ids_sha256": digest,
    }
