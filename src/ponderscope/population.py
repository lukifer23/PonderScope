"""Frozen task-population locks.

A population lock is a committed, versioned record of the exact task and
presentation identities declared for an experiment, so that a change to the
generator or its parameters cannot silently alter the held-out task instances
under the same nominal specification. The lock reflects an *already declared*
population; it never introduces a new one.

Enforcement is fail-closed: when a lock matching a specification's population is
found, the freshly generated population must match it exactly (task ids,
presentation ids, structural signatures, family/difficulty counts, digests) or
the run/preflight is refused. Historical specifications without a lock remain
supported.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any

from .config.identity import configuration_id
from .config.schema import ExperimentSpec
from .tasks import generate_pack, generate_pack_metadata
from .tasks.invariants import signature_hash

LOCK_SCHEMA = "ponderscope-population-lock/1"
LOCK_GLOB = "*population-lock*.json"


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def population_key(spec: ExperimentSpec) -> str:
    """Stable key of the *population-defining* fields of a specification."""
    return configuration_id(
        {
            "task_pack": spec.task_pack,
            "split": spec.split,
            "task_seed": spec.task_seed,
            "prompt_policy": spec.prompt_policy,
            "families": sorted(spec.families)
            if spec.families
            else ["arith", "logic", "order", "path", "sm"],
            "n_per_family": spec.n_per_family,
        },
        prefix="pop",
    )


def expected_executions(spec: ExperimentSpec) -> dict[str, int]:
    """Declared task/draw/execution counts.

    Distinguishes unique stochastic draws from technical executions: technical
    repeats multiply executions, never draws.
    """
    n_families = len(spec.families) if spec.families else 5
    n_tasks = n_families * spec.n_per_family
    sampled_draws = n_tasks * len(spec.sampled_seeds)
    sampled_executions = sampled_draws * spec.sampled_repeats_per_seed
    greedy_executions = n_tasks * spec.greedy_repeats
    return {
        "n_tasks": n_tasks,
        "greedy_draws": n_tasks if spec.greedy_repeats else 0,
        "greedy_executions": greedy_executions,
        "sampled_draws": sampled_draws,
        "sampled_executions": sampled_executions,
        "total_draws": (n_tasks if spec.greedy_repeats else 0) + sampled_draws,
        "total_executions": greedy_executions + sampled_executions,
    }


def _structural_signature_digest(tasks: list[Any]) -> tuple[str, dict[str, str]]:
    per_task = {t.task_id: signature_hash(t) for t in tasks}
    payload = "\n".join(f"{tid}:{sig}" for tid, sig in sorted(per_task.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), per_task


def build_lock(
    spec: ExperimentSpec,
    *,
    spec_hashes: dict[str, str] | None = None,
    code_sha: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Build a population lock from the declared specification (pure)."""
    tasks = generate_pack(
        families=spec.families or None,
        n_per_family=spec.n_per_family,
        pack=spec.task_pack,
        split=spec.split,
        seed=spec.task_seed,
        prompt_policy=spec.prompt_policy,
    )
    meta = generate_pack_metadata(
        tasks,
        families=spec.families or None,
        n_per_family=spec.n_per_family,
        split=spec.split,
        seed=spec.task_seed,
        pack=spec.task_pack,
        prompt_policy=spec.prompt_policy,
    )
    family_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    for t in tasks:
        family_counts[t.family] = family_counts.get(t.family, 0) + 1
        level = str(t.difficulty.get("level", "?"))
        difficulty_counts[level] = difficulty_counts.get(level, 0) + 1
    sig_digest, per_task_sigs = _structural_signature_digest(tasks)
    return {
        "schema": LOCK_SCHEMA,
        "created_utc": _utc_now(),
        "population_key": population_key(spec),
        "spec_hashes": spec_hashes or {},
        "task_pack": spec.task_pack,
        "generator_version": meta["generator_version"],
        "split": spec.split,
        "task_seed": spec.task_seed,
        "prompt_policy": spec.prompt_policy,
        "families": sorted(family_counts),
        "n_per_family": spec.n_per_family,
        "n_tasks": meta["n_tasks"],
        "family_counts": dict(sorted(family_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "task_ids": meta["structural_task_ids"],
        "presentation_ids": meta["presentation_ids"],
        "task_ids_sha256": meta["task_ids_sha256"],
        "presentation_ids_sha256": meta["presentation_ids_sha256"],
        "structural_signature_sha256": sig_digest,
        "structural_signatures": per_task_sigs,
        "expected_executions": expected_executions(spec),
        "code": {"git_sha": code_sha, "generator_version": meta["generator_version"]},
        "notes": notes,
    }


def load_lock(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    if data.get("schema") != LOCK_SCHEMA:
        raise ValueError(f"unsupported population-lock schema {data.get('schema')!r} in {path}")
    for key in ("population_key", "task_ids_sha256", "presentation_ids_sha256"):
        if key not in data:
            raise ValueError(f"population lock missing required key {key!r} in {path}")
    return data


def find_lock_for_spec(spec: ExperimentSpec, specs_dir: str | Path = "specs") -> Path | None:
    """Locate the lock whose population matches the specification, if any."""
    directory = Path(specs_dir)
    if not directory.exists():
        return None
    wanted = population_key(spec)
    matches: list[Path] = []
    for path in sorted(directory.glob(LOCK_GLOB)):
        try:
            lock = load_lock(path)
        except (ValueError, OSError):
            continue
        if lock.get("population_key") == wanted:
            matches.append(path)
    if len(matches) > 1:
        raise ValueError(f"multiple population locks match the specification: {matches}")
    return matches[0] if matches else None


def verify_lock(
    spec: ExperimentSpec,
    lock: dict[str, Any],
    *,
    tasks: list[Any] | None = None,
) -> dict[str, Any]:
    """Verify a freshly generated population against a lock (fail closed)."""
    if lock.get("population_key") != population_key(spec):
        return {"ok": False, "mismatches": ["population key does not match the specification"]}
    if tasks is None:
        tasks = generate_pack(
            families=spec.families or None,
            n_per_family=spec.n_per_family,
            pack=spec.task_pack,
            split=spec.split,
            seed=spec.task_seed,
            prompt_policy=spec.prompt_policy,
        )
    meta = generate_pack_metadata(
        tasks,
        families=spec.families or None,
        n_per_family=spec.n_per_family,
        split=spec.split,
        seed=spec.task_seed,
        pack=spec.task_pack,
        prompt_policy=spec.prompt_policy,
    )
    family_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    for t in tasks:
        family_counts[t.family] = family_counts.get(t.family, 0) + 1
        difficulty_counts[str(t.difficulty.get("level", "?"))] = (
            difficulty_counts.get(str(t.difficulty.get("level", "?")), 0) + 1
        )
    sig_digest, _ = _structural_signature_digest(tasks)

    mismatches: list[str] = []
    if meta["task_ids_sha256"] != lock.get("task_ids_sha256"):
        mismatches.append("task-id digest differs")
    if meta["presentation_ids_sha256"] != lock.get("presentation_ids_sha256"):
        mismatches.append("presentation-id digest differs")
    if sig_digest != lock.get("structural_signature_sha256"):
        mismatches.append("structural-signature digest differs")
    if dict(sorted(family_counts.items())) != lock.get("family_counts"):
        mismatches.append("family counts differ")
    if dict(sorted(difficulty_counts.items())) != lock.get("difficulty_counts"):
        mismatches.append("difficulty distribution differs")
    if meta["n_tasks"] != lock.get("n_tasks"):
        mismatches.append("task count differs")
    for field, value in (
        ("task_pack", spec.task_pack),
        ("generator_version", meta["generator_version"]),
        ("split", spec.split),
        ("task_seed", spec.task_seed),
        ("prompt_policy", spec.prompt_policy),
        ("n_per_family", spec.n_per_family),
    ):
        if lock.get(field) != value:
            mismatches.append(f"{field} differs ({lock.get(field)!r} != {value!r})")
    # ``expected_executions`` is recorded for the primary design but is a
    # sampling-design property, not a population property: a different design
    # (e.g. a greedy probe run) over the same frozen population is legitimate and
    # must not fail the population lock.
    return {"ok": not mismatches, "mismatches": mismatches}


def enforce_population_lock(
    spec: ExperimentSpec,
    *,
    lock_path: str | Path | None = None,
    specs_dir: str | Path = "specs",
    tasks: list[Any] | None = None,
) -> dict[str, Any]:
    """Locate (or load) the lock and verify the declared population.

    Returns ``{"locked": bool, "path": str|None, "ok": bool, "mismatches": [...]}``.
    An unlocked specification is reported as ``locked=False`` and is not an error.
    """
    path: Path | None
    if lock_path is not None:
        path = Path(lock_path)
        if not path.exists():
            return {
                "locked": True,
                "path": str(path),
                "ok": False,
                "mismatches": ["lock file missing"],
            }
    else:
        path = find_lock_for_spec(spec, specs_dir)
    if path is None:
        return {"locked": False, "path": None, "ok": True, "mismatches": []}
    lock = load_lock(path)
    result = verify_lock(spec, lock, tasks=tasks)
    result.update({"locked": True, "path": str(path)})
    return result
