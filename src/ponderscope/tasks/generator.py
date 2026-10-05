"""Deterministic, versioned procedural task generation.

Task identity depends only on the declared generation fields, never on the
rendered wording. Regenerating the same fields always yields byte-identical
tasks, and changing a wording template would not change a task id.
"""

from __future__ import annotations

import hashlib
import random

from .families import ALL_FAMILIES, DIFFICULTY_LEVELS, make_family_task
from .models import Task, validate_task

PACK_VERSION = "tasks-v1"
GENERATOR_VERSION = "generator-v1"
SPLITS = ("train", "calibration", "dev", "test")

_SPLIT_OFFSET = {"train": 0, "calibration": 1_000_003, "dev": 2_000_003, "test": 3_000_003}


def task_id_for(
    pack: str,
    family: str,
    split: str,
    seed: int,
    index: int,
    variant: str | None,
) -> str:
    """Wording-independent, stable task id."""
    key = "|".join([pack, GENERATOR_VERSION, family, split, str(seed), str(index), variant or ""])
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
) -> Task:
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split!r}")
    task_seed = seed + _SPLIT_OFFSET[split] + index
    rng = random.Random(f"{pack}|{GENERATOR_VERSION}|{family}|{task_seed}|{variant}")
    diff = difficulty or DIFFICULTY_LEVELS[index % len(DIFFICULTY_LEVELS)]
    data = make_family_task(family, rng, diff)
    task = Task(
        task_id=task_id_for(pack, family, split, seed, index, variant),
        family=family,
        pack=pack,
        version=PACK_VERSION,
        split=split,
        seed=seed,
        index=index,
        variant=variant,
        prompt=data["prompt"],
        answer=data["answer"],
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
) -> list[Task]:
    families = list(families) if families else list(ALL_FAMILIES)
    for family in families:
        if family not in ALL_FAMILIES:
            raise ValueError(f"unknown family: {family!r}")
    tasks: list[Task] = []
    for family in families:
        for index in range(n_per_family):
            tasks.append(
                make_task(family, index, pack=pack, split=split, seed=seed, variant=variant)
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
) -> dict:
    """Provenance of exactly what was generated (not merely requested)."""
    task_ids = sorted(t.task_id for t in tasks)
    digest = hashlib.sha256("\n".join(task_ids).encode("utf-8")).hexdigest()
    actual_families = sorted({t.family for t in tasks})
    return {
        "pack_version": PACK_VERSION,
        "generator_version": GENERATOR_VERSION,
        "requested_pack": pack,
        "families": actual_families,
        "n_per_family": n_per_family,
        "split": split,
        "seed": seed,
        "n_tasks": len(tasks),
        "task_ids_sha256": digest,
    }
