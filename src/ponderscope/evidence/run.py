"""Run directories: create-once, immutable, atomic evidence storage.

Raw evidence (tasks, traces, probes, final seal) is create-once and never
overwritten. Derived analysis/report artifacts may be regenerated and are marked
as derived by name.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config.conditions import condition_identities
from ..config.identity import DecodingPolicy, Deployment, sha256_file
from ..config.schema import ExperimentSpec
from .environment import capture_code_state, capture_environment
from .store import (
    JsonlWriter,
    atomic_write_json,
    atomic_write_text,
    read_json,
    read_jsonl,
)

RUNS_DIRNAME = "runs"
MANIFEST = "manifest.json"
ENVIRONMENT = "environment.json"
TASKS = "tasks.jsonl"
TRACES = "traces.jsonl"
PROBES = "probes.jsonl"
EVIDENCE = "evidence.json"
ANALYSIS = "analysis.json"
SUMMARY_MD = "summary.md"
SUMMARY_HTML = "summary.html"

RAW_FILES = (ENVIRONMENT, TASKS, TRACES, PROBES)


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in text]
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "run"


@dataclass
class RunStore:
    """Handle to a single immutable run directory."""

    path: Path
    manifest: dict[str, Any]

    # -- ids -----------------------------------------------------------------
    @property
    def run_id(self) -> str:
        return self.manifest["run_id"]

    @property
    def source_artifact_id(self) -> str:
        return self.manifest.get("source_artifact_id") or self.manifest.get("artifact_id", "")

    @property
    def weight_variant_id(self) -> str:
        return self.manifest.get("weight_variant_id") or self.manifest.get("artifact_id", "")

    @property
    def artifact_id(self) -> str:
        # Backwards-compatible alias for the executable weight variant.
        return self.weight_variant_id

    @property
    def deployment_id(self) -> str:
        return self.manifest["deployment_id"]

    @property
    def condition_ids(self) -> list[str]:
        return [c["condition_id"] for c in self.manifest.get("conditions", [])]

    @property
    def is_sealed(self) -> bool:
        return (self.path / EVIDENCE).exists()

    # -- creation ------------------------------------------------------------
    @classmethod
    def create(
        cls,
        deployment: Deployment,
        spec: ExperimentSpec,
        runs_dir: str | Path = RUNS_DIRNAME,
        created_utc: str | None = None,
        run_suffix: str | None = None,
        conditions: list[DecodingPolicy] | None = None,
        code_state: dict[str, Any] | None = None,
        prompt_policy: str | None = None,
    ) -> RunStore:
        runs_dir = Path(runs_dir)
        created_utc = created_utc or _utc_now()
        suffix = f"-{_slug(run_suffix)}" if run_suffix else ""
        run_id = f"{created_utc}-{_slug(spec.name)}{suffix}-{deployment.deployment_id}"
        path = runs_dir / run_id
        if path.exists():
            raise FileExistsError(f"run already exists and will not be overwritten: {path}")
        path.mkdir(parents=True)
        environment = capture_environment()
        conds = conditions if conditions is not None else condition_identities(spec)
        manifest: dict[str, Any] = {
            "run_id": run_id,
            "created_utc": created_utc,
            "name": spec.name,
            "spec": spec.to_dict(),
            "spec_hash": spec.canonical_hash(),
            "source_artifact": deployment.model.source.to_dict(),
            "source_artifact_id": deployment.source_artifact_id,
            "weight_variant": deployment.model.to_dict(),
            "weight_variant_id": deployment.weight_variant_id,
            # Backwards-compatible alias for the executable variant.
            "artifact": deployment.model.to_dict(),
            "artifact_id": deployment.artifact_id,
            "runtime": deployment.runtime.to_dict(),
            "deployment_id": deployment.deployment_id,
            "deployment_description": deployment.describe(),
            "conditions": [
                {"condition_id": d.condition_id, "decoding": d.identity_dict()} for d in conds
            ],
            "code": code_state if code_state is not None else capture_code_state(),
            "prompt_policy": prompt_policy,
            "timestamps": {"started_utc": created_utc, "completed_utc": None},
            "status": {"run": "IN_PROGRESS", "observations": [], "error": None},
        }
        atomic_write_json(path / MANIFEST, manifest)
        atomic_write_json(path / ENVIRONMENT, environment)
        return cls(path=path, manifest=manifest)

    # -- loading -------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> RunStore:
        path = Path(path)
        manifest = read_json(path / MANIFEST)
        return cls(path=path, manifest=manifest)

    @classmethod
    def latest(cls, runs_dir: str | Path = RUNS_DIRNAME) -> RunStore:
        runs_dir = Path(runs_dir)
        candidates = [p for p in runs_dir.iterdir() if (p / MANIFEST).exists()]
        if not candidates:
            raise FileNotFoundError(f"no runs found in {runs_dir}")
        candidates.sort(key=lambda p: (p.stat().st_mtime_ns, p.name))
        return cls.load(candidates[-1])

    # -- status --------------------------------------------------------------
    def _write_manifest(self) -> None:
        if self.is_sealed:
            raise RuntimeError(f"run {self.run_id} is sealed; refusing to mutate manifest")
        atomic_write_json(self.path / MANIFEST, self.manifest)

    def update_status(self, status: str, observations: list[str] | None = None) -> None:
        self.manifest["status"]["run"] = status
        if observations:
            self.manifest["status"]["observations"].extend(observations)
        self._write_manifest()

    def add_observation(self, text: str) -> None:
        self.manifest["status"]["observations"].append(text)
        self._write_manifest()

    def mark_failed(self, error_type: str, message: str, partial: bool = True) -> None:
        """Record a failed/partial run without discarding partial evidence."""
        self.manifest["status"]["run"] = "PARTIAL" if partial else "FAILED"
        self.manifest["status"]["error"] = {"type": error_type, "message": message}
        self.manifest["timestamps"]["completed_utc"] = _utc_now()
        atomic_write_json(self.path / MANIFEST, self.manifest)

    # -- raw evidence (create-once) ------------------------------------------
    def write_tasks(self, tasks: list[dict[str, Any]]) -> None:
        from .store import write_jsonl

        write_jsonl(self.path / TASKS, tasks, overwrite=False)

    def read_tasks(self) -> list[dict[str, Any]]:
        return list(read_jsonl(self.path / TASKS))

    def open_traces(self) -> JsonlWriter:
        return JsonlWriter(self.path / TRACES).open()

    def read_traces(self) -> list[dict[str, Any]]:
        return list(read_jsonl(self.path / TRACES))

    def open_probes(self) -> JsonlWriter:
        return JsonlWriter(self.path / PROBES).open()

    def read_probes(self) -> list[dict[str, Any]]:
        path = self.path / PROBES
        return list(read_jsonl(path)) if path.exists() else []

    # -- sealing -------------------------------------------------------------
    def seal(self) -> dict[str, Any]:
        """Finalize the manifest, then write a create-once evidence seal last.

        Ordering invariant (Phase 1.2 fix): the manifest is fully finalized and
        atomically written *before* it is hashed into ``evidence.json``. Once
        ``evidence.json`` exists the manifest and all raw evidence are immutable,
        and the seal's ``manifest_sha256`` matches the on-disk manifest.
        """
        if self.is_sealed:
            raise FileExistsError(f"run {self.run_id} already sealed")
        # 1. finalize all manifest content first.
        sealed_utc = _utc_now()
        self.manifest["status"]["run"] = "EVIDENCE_COMPLETE"
        self.manifest["timestamps"]["completed_utc"] = sealed_utc
        self.manifest["seal_schema"] = "ponderscope-seal/2"
        # 2. atomically write the final manifest.
        atomic_write_json(self.path / MANIFEST, self.manifest)
        # 3. hash the FINAL manifest and all raw evidence.
        hashes: dict[str, str] = {}
        for name in RAW_FILES:
            p = self.path / name
            if p.exists():
                hashes[name] = sha256_file(str(p))
        manifest_path = self.path / MANIFEST
        seal = {
            "run_id": self.run_id,
            "artifact_id": self.artifact_id,
            "deployment_id": self.deployment_id,
            "spec_hash": self.manifest.get("spec_hash"),
            "code": self.manifest.get("code"),
            "files": hashes,
            "manifest_sha256": sha256_file(str(manifest_path)) if manifest_path.exists() else None,
            "sealed_utc": sealed_utc,
            "seal_schema": "ponderscope-seal/2",
        }
        # 4. write the seal LAST.
        atomic_write_json(self.path / EVIDENCE, seal)
        return seal

    def read_seal(self) -> dict[str, Any]:
        return read_json(self.path / EVIDENCE)

    def verify(self) -> dict[str, Any]:
        return verify_run(self.path)

    # -- derived artifacts (regenerable) -------------------------------------
    def write_analysis(self, analysis: dict[str, Any]) -> None:
        atomic_write_json(self.path / ANALYSIS, analysis)

    def read_analysis(self) -> dict[str, Any]:
        return read_json(self.path / ANALYSIS)

    def write_summary(self, markdown: str, html: str) -> None:
        atomic_write_text(self.path / SUMMARY_MD, markdown)
        atomic_write_text(self.path / SUMMARY_HTML, html)

    def has_analysis(self) -> bool:
        return (self.path / ANALYSIS).exists()

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "run_id": self.run_id,
            "artifact_id": self.artifact_id,
            "deployment_id": self.deployment_id,
        }


def verify_run(path: str | Path) -> dict[str, Any]:
    """Recompute every raw evidence hash and the final manifest hash.

    Read-only: this never writes or repairs evidence. Returns a PASS/FAIL report
    with per-file detail. A run whose seal is missing, or whose manifest no
    longer matches ``manifest_sha256``, FAILS.
    """
    path = Path(path)
    errors: list[str] = []
    checks: list[dict[str, Any]] = []
    evidence_path = path / EVIDENCE
    if not evidence_path.exists():
        return {
            "path": str(path),
            "run_id": None,
            "pass": False,
            "errors": ["no evidence seal (evidence.json missing)"],
            "checks": [],
        }
    seal = read_json(evidence_path)
    for name in RAW_FILES:
        p = path / name
        expected = seal.get("files", {}).get(name)
        actual = sha256_file(str(p)) if p.exists() else None
        ok = expected is not None and actual == expected
        checks.append({"file": name, "expected": expected, "actual": actual, "ok": ok})
        if not ok:
            errors.append(
                f"{name}: hash mismatch or missing (expected={expected}, actual={actual})"
            )
    manifest_path = path / MANIFEST
    manifest_actual = sha256_file(str(manifest_path)) if manifest_path.exists() else None
    manifest_expected = seal.get("manifest_sha256")
    manifest_ok = manifest_expected is not None and manifest_actual == manifest_expected
    checks.append(
        {
            "file": MANIFEST,
            "expected": manifest_expected,
            "actual": manifest_actual,
            "ok": manifest_ok,
        }
    )
    if not manifest_ok:
        errors.append(
            "manifest.json: hash mismatch — seal does not cover the final manifest "
            f"(expected={manifest_expected}, actual={manifest_actual})"
        )
    return {
        "path": str(path),
        "run_id": seal.get("run_id"),
        "pass": not errors,
        "errors": errors,
        "checks": checks,
    }


def load_manifest_text(path: str | Path) -> str:
    return json.dumps(read_json(Path(path) / MANIFEST), indent=2)
