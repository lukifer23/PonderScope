"""Experiment preflight: verify preconditions before a real run.

A small, reusable validation function (not an orchestration framework). It
performs no model inference: it checks cached source availability, derived
artifact provenance, disk and memory headroom, runtime versions, the frozen task
population (including disjointness from an earlier population), the expected
generation count, a compute estimate, duplicate runs, and any active/incomplete
run directories. It is advisory: a failed check is reported, never silently
ignored, and never mutates anything.
"""

from __future__ import annotations

import platform
import shutil
import time
from pathlib import Path
from typing import Any

from .config.schema import ExperimentSpec
from .population import enforce_population_lock, expected_executions
from .tasks import generate_pack, generate_pack_metadata
from .tasks.invariants import collision_audit, signature_hash


def _expected_generations(spec: ExperimentSpec) -> dict[str, int]:
    return expected_executions(spec)


def run_preflight(
    spec: ExperimentSpec,
    *,
    model_repo: str,
    model_revision: str,
    artifact_path: str | None = None,
    runs_dir: str | Path = "runs",
    per_generation_seconds: float | None = None,
    comparison_population: tuple[str, int, int] | None = None,
    population_lock: str | None = None,
    specs_dir: str | Path = "specs",
) -> dict[str, Any]:
    """Return a preflight report. ``comparison_population`` = (split, seed, n_per_family)."""
    checks: dict[str, Any] = {}
    problems: list[str] = []

    # 1. Task population (deterministic, frozen).
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
    checks["task_population"] = {
        "n_tasks": meta["n_tasks"],
        "family_counts": family_counts,
        "difficulty_counts": difficulty_counts,
        "task_ids_sha256": meta["task_ids_sha256"],
        "presentation_ids_sha256": meta["presentation_ids_sha256"],
    }

    # 1b. Frozen population lock (fail closed when a lock exists).
    lock = enforce_population_lock(
        spec, lock_path=population_lock, specs_dir=specs_dir, tasks=tasks
    )
    checks["population_lock"] = lock
    if lock["locked"] and not lock["ok"]:
        problems.append(f"frozen population lock does not match: {lock['mismatches']}")
    if comparison_population is not None:
        split, seed, n = comparison_population
        other = generate_pack(
            families=spec.families or None,
            n_per_family=n,
            pack=spec.task_pack,
            split=split,
            seed=seed,
            prompt_policy=spec.prompt_policy,
        )
        id_overlap = {t.task_id for t in tasks} & {t.task_id for t in other}
        sig_overlap = {signature_hash(t) for t in tasks} & {signature_hash(t) for t in other}
        audit = collision_audit({"primary": tasks, "comparison": other})
        checks["disjointness"] = {
            "comparison": {"split": split, "seed": seed, "n_per_family": n},
            "task_id_overlap": len(id_overlap),
            "structural_overlap": len(sig_overlap),
            "collision_clean": audit["clean"],
        }
        if id_overlap or sig_overlap:
            problems.append("task population overlaps the comparison population")

    # 2. Cached source availability.
    snapshot = None
    try:
        from .backends.mlx_backend import resolve_local_snapshot

        snapshot = resolve_local_snapshot(model_repo, model_revision)
        checks["source_cache"] = {"available": True, "path": str(snapshot)}
    except Exception as exc:
        checks["source_cache"] = {"available": False, "error": str(exc)}
        problems.append(f"source snapshot not cached: {exc}")

    # 3. Derived artifact provenance.
    if artifact_path is not None:
        from .conversion import (
            load_variant_provenance,
            verify_variant_artifact,
        )

        try:
            prov = load_variant_provenance(artifact_path)
            verified = verify_variant_artifact(artifact_path, prov)
            src = prov["source_artifact"]
            lineage_ok = src.get("repo_id") == model_repo and src.get("revision") == model_revision
            checks["artifact"] = {
                "representation": prov.get("representation"),
                "precision": prov.get("precision"),
                "quantization": prov.get("quantization"),
                "bits_per_weight": prov.get("bits_per_weight"),
                "source_lineage_ok": lineage_ok,
                "derived_hashes_ok": verified["ok"],
            }
            if not lineage_ok:
                problems.append("derived artifact source lineage does not match the request")
            if not verified["ok"]:
                problems.append("derived artifact hashes do not verify")
        except Exception as exc:
            checks["artifact"] = {"error": str(exc)}
            problems.append(f"derived artifact provenance invalid: {exc}")

    # 4. Disk and memory.
    disk = shutil.disk_usage(str(Path(runs_dir).resolve().parent))
    checks["disk"] = {"free_bytes": disk.free, "total_bytes": disk.total}
    checks["memory_bytes"] = _physical_memory()

    # 5. Runtime versions.
    try:
        import mlx_lm

        checks["runtime"] = {
            "mlx_lm": getattr(mlx_lm, "__version__", "unknown"),
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
    except Exception as exc:  # pragma: no cover - host dependent
        checks["runtime"] = {"error": str(exc)}
        problems.append(f"runtime not importable: {exc}")

    # 6. Expected generations and compute estimate.
    expected = _expected_generations(spec)
    checks["expected_generations"] = expected
    if per_generation_seconds is not None:
        seconds = expected["total_executions"] * per_generation_seconds
        checks["compute_estimate"] = {
            "per_generation_seconds": per_generation_seconds,
            "total_seconds": seconds,
            "total_hours": round(seconds / 3600, 2),
        }

    # 7. Duplicate and active/incomplete runs (live vs stale).
    runs_path = Path(runs_dir)
    duplicates: list[str] = []
    live: list[str] = []
    stale: list[str] = []
    now = time.time()
    if runs_path.exists():
        for manifest_path in runs_path.glob("*/manifest.json"):
            try:
                import json

                manifest = json.loads(manifest_path.read_text())
            except Exception:
                continue
            run_id = manifest.get("run_id", manifest_path.parent.name)
            status = manifest.get("status", {}).get("run")
            if status in ("RUNNING", "IN_PROGRESS", "PARTIAL", "FAILED"):
                age = now - manifest_path.stat().st_mtime
                entry = f"{run_id} ({status}, age={int(age)}s)"
                (live if age < 600 else stale).append(entry)
            same_spec = manifest.get("spec_hash") == spec.canonical_hash()
            recorded_path = manifest.get("weight_variant", {}).get("local_path") or manifest.get(
                "artifact", {}
            ).get("local_path")
            same_variant = (artifact_path is None and recorded_path is None) or (
                artifact_path is not None
                and recorded_path is not None
                and Path(recorded_path) == Path(artifact_path)
            )
            if same_spec and same_variant:
                duplicates.append(run_id)
    checks["duplicate_runs"] = duplicates
    checks["active_runs"] = live
    checks["stale_runs"] = stale
    checks["active_or_incomplete_runs"] = live + stale

    # 8. Code state (advisory: preflight never mutates).
    from .evidence.environment import capture_code_state

    code_state = capture_code_state(include_diff_hash=False)
    checks["code_state"] = {
        "git_sha": code_state.get("git_sha"),
        "tracked_dirty": code_state.get("tracked_dirty"),
    }

    # 9. Gate evaluation: PASS / WARNING / BLOCK.
    gates: dict[str, str] = {}
    blocks: list[str] = []
    warnings: list[str] = []

    def gate(name: str, status: str, detail: str = "") -> None:
        gates[name] = status
        if status == "BLOCK":
            blocks.append(f"{name}: {detail}" if detail else name)
        elif status == "WARNING":
            warnings.append(f"{name}: {detail}" if detail else name)

    src = checks["source_cache"]
    gate("source_cache", "PASS" if src.get("available") else "BLOCK", src.get("error", ""))

    if "artifact" in checks:
        art = checks["artifact"]
        if art.get("error"):
            gate("artifact", "BLOCK", art["error"])
        elif art.get("source_lineage_ok") and art.get("derived_hashes_ok"):
            gate("artifact", "PASS")
        else:
            gate("artifact", "BLOCK", "lineage or derived-hash verification failed")

    lock = checks["population_lock"]
    if not lock.get("locked"):
        gate("population_lock", "PASS", "no lock declared for this specification")
    elif lock.get("ok"):
        gate("population_lock", "PASS")
    else:
        gate("population_lock", "BLOCK", str(lock.get("mismatches")))

    required_bytes = max(1 << 30, expected["total_executions"] * 200_000)
    checks["required_disk_bytes"] = required_bytes
    free = checks["disk"]["free_bytes"]
    if free < required_bytes:
        gate("disk", "BLOCK", f"free={free} < required={required_bytes}")
    elif free < 5 * required_bytes:
        gate("disk", "WARNING", f"free={free} is within 5x required={required_bytes}")
    else:
        gate("disk", "PASS")

    mem = checks["memory_bytes"]
    if mem is None:
        gate("memory", "WARNING", "physical memory unknown")
    elif mem < 12 * (1 << 30):
        gate("memory", "BLOCK", f"{mem} < 12 GiB")
    elif mem < 16 * (1 << 30):
        gate("memory", "WARNING", f"{mem} < 16 GiB")
    else:
        gate("memory", "PASS")

    rt = checks["runtime"]
    if rt.get("error"):
        gate("runtime", "BLOCK", rt["error"])
    else:
        py = str(rt.get("python", ""))
        if not py.startswith("3.12"):
            gate("runtime", "BLOCK", f"python {py} is not 3.12.x")
        elif rt.get("mlx_lm") != "0.32.0":
            gate("runtime", "WARNING", f"mlx-lm {rt.get('mlx_lm')} != pinned 0.32.0")
        else:
            gate("runtime", "PASS")

    gate("expected_generations", "PASS")

    if duplicates:
        gate("duplicate_runs", "BLOCK", str(duplicates))
    else:
        gate("duplicate_runs", "PASS")

    if live:
        gate("active_runs", "BLOCK", f"live experiment(s): {live}")
    elif stale:
        gate("active_runs", "WARNING", f"stale partial run(s): {stale}")
    else:
        gate("active_runs", "PASS")

    if code_state.get("tracked_dirty"):
        gate("code_state", "WARNING", "tracked worktree is dirty (run would be exploratory)")
    else:
        gate("code_state", "PASS")

    return {
        "ok": not blocks,
        "blocks": blocks,
        "warnings": warnings,
        "gates": gates,
        # backward-compatible alias: problems == blocking conditions
        "problems": blocks,
        "checks": checks,
        "spec_hash": spec.canonical_hash(),
        "spec_name": spec.name,
    }


def _physical_memory() -> int | None:
    import subprocess

    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5, check=False
        )
        return int(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
